#!/usr/bin/env python3
"""
Janus V6 Training Pipeline — google/gemma-4-26B-A4B-it (27B MoE)
==================================================================
CORRECT model: 26B total, 4B active per token (128 experts, top-8)
Standard HuggingFace + PEFT + bitsandbytes. NO Unsloth.

Phase 0: Download model (if not cached)
Phase 1: SFT with QLoRA (manual training loop)
Phase 2: DPO with QLoRA (manual training loop)
Phase 3: Merge LoRA → bf16 on CPU (uses swap)
Phase 4: GGUF Q4_K_M via llama.cpp
Phase 5: Ollama deploy + smoke test

Usage:
  python3 v6_26b_pipeline.py --phase 0       # Download only
  python3 v6_26b_pipeline.py --phase 1       # SFT
  python3 v6_26b_pipeline.py --phase 2       # DPO (requires SFT adapter)
  python3 v6_26b_pipeline.py --phase 3       # Merge
  python3 v6_26b_pipeline.py --phase 4       # GGUF
  python3 v6_26b_pipeline.py --phase 5       # Ollama
  python3 v6_26b_pipeline.py --phase all     # Run all phases
  python3 v6_26b_pipeline.py --phase diag    # Architecture diagnostics only
"""
import os, sys, json, time, glob, re, random, subprocess, gc, argparse

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# ===== CONFIGURATION =====
MODEL_NAME = "google/gemma-4-26B-A4B-it"
BASE_DIR = os.path.expanduser("~/project")
ADAPTER_DIR = f"{BASE_DIR}/adapters/janus_v6"
MERGED_DIR = f"{BASE_DIR}/merged_models/gemma4_janus_v6"
GGUF_DIR = f"{BASE_DIR}/merged_models"
LLAMA_CPP = f"{BASE_DIR}/llama.cpp"

MAX_SEQ_LENGTH = 512   # SFT: safe for 24GB VRAM
DPO_SEQ_LENGTH = 768   # DPO: longer for ReAct/tool-chains (only 2 fwd per pair with cache)
LORA_R = 16
LORA_ALPHA = 16

# For 26B MoE: attention-only LoRA by default (safe for 24GB VRAM)
# 128 experts × 30 layers × 3 proj = 11520 modules — too many for LoRA
# Attention: 30 layers × 4 proj = 120 modules — manageable
# MLP LoRA: gate/up/down_proj in router MLP (NOT expert params!)
#   adds ~2.6GB but improves tool-calling generation quality
TARGET_MODULES_ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
TARGET_MODULES_FULL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
INCLUDE_MLP_LORA = False    # SFT: attention-only (safe)
DPO_INCLUDE_MLP_LORA = True # DPO: add MLP for better tool output generation
DPO_MLP_LORA_R = 8          # Lower rank for MLP — less risk of destabilizing MoE routing

# SFT Config
SFT_EPOCHS = 2
SFT_LR = 2e-5
SFT_BATCH_SIZE = 1
SFT_GRAD_ACCUM = 16
SFT_WARMUP_RATIO = 0.05
SFT_MAX_GRAD_NORM = 1.0

# DPO Config
DPO_EPOCHS = 3
DPO_LR = 2e-6
DPO_BETA = 0.1
DPO_BATCH_SIZE = 1
DPO_GRAD_ACCUM = 16
DPO_HARD_TARGET_RATIO = 0.30  # Upsample HARD examples to 30% of DPO dataset

# Data Files — adjust paths to your training data location
SFT_FILES = [
    f"{BASE_DIR}/data/training_data.jsonl",
]
DPO_FILE = f"{BASE_DIR}/data/dpo_pairs.jsonl"

LOG_FILE = "/tmp/v6_training.log"
CHECKPOINT_FILE = "/tmp/v6_checkpoint.json"

# ===== LOGGING =====
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_checkpoint(phase, step=0, extra=None):
    data = {"phase": phase, "step": step, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        data.update(extra)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

# ===== PHASE 0: DOWNLOAD MODEL =====
def phase_download():
    log("=" * 60)
    log("PHASE 0: DOWNLOAD google/gemma-4-26B-A4B-it")
    log("=" * 60)
    
    from transformers import AutoTokenizer, AutoConfig
    
    # Check if already cached
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--google--gemma-4-26B-A4B-it")
    
    if os.path.exists(model_cache):
        snapshots = os.path.join(model_cache, "snapshots")
        if os.path.exists(snapshots) and os.listdir(snapshots):
            snap = os.listdir(snapshots)[0]
            snap_dir = os.path.join(snapshots, snap)
            safetensors = glob.glob(os.path.join(snap_dir, "*.safetensors"))
            if len(safetensors) >= 5:
                total_size = sum(os.path.getsize(f) for f in safetensors)
                log(f"  Model already cached: {len(safetensors)} shards, {total_size/1e9:.1f} GB")
                log(f"  Location: {snap_dir}")
                return True
    
    log("  Model not in cache. Starting download (~52 GB)...")
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    text_cfg = config.text_config
    log(f"  Hidden size: {text_cfg.hidden_size}")
    log(f"  Num layers: {text_cfg.num_hidden_layers}")
    log(f"  Num experts: {text_cfg.num_experts}")
    log(f"  Top-k experts: {text_cfg.top_k_experts}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    log(f"  Vocab size: {tokenizer.vocab_size}")
    
    from huggingface_hub import snapshot_download
    start = time.time()
    local_dir = snapshot_download(MODEL_NAME, ignore_patterns=["*.gguf", "*.bin"])
    elapsed = time.time() - start
    
    safetensors = glob.glob(os.path.join(local_dir, "*.safetensors"))
    total_size = sum(os.path.getsize(f) for f in safetensors) if safetensors else 0
    log(f"  Download complete in {elapsed:.0f}s — {len(safetensors)} shards, {total_size/1e9:.1f} GB")
    
    save_checkpoint("download_complete")
    return True


# ===== DATA LOADING =====

def text_to_messages(text):
    """Convert pre-formatted Gemma3/4 text to messages list."""
    messages = []
    ROLE_MAP = {
        "developer": "system", "model": "assistant",
        "user": "user", "system": "system", "assistant": "assistant",
    }
    
    for pattern in [r'<start_of_turn>(\w+)\n(.*?)<end_of_turn>',
                    r'<\|turn>(\w+)\n(.*?)<turn\|>']:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for role, content in matches:
                mapped_role = ROLE_MAP.get(role, role)
                messages.append({"role": mapped_role, "content": content.strip()})
            return messages if messages else None
    return None


def load_sft_data(tokenizer):
    """Load all SFT files, convert to Gemma4 format via apply_chat_template."""
    all_texts = []
    skipped = 0
    
    for sf in SFT_FILES:
        if not os.path.exists(sf):
            log(f"  SKIP (not found): {sf}")
            continue
        loaded = 0
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    messages = None
                    
                    if "messages" in d:
                        messages = d["messages"]
                    elif "text" in d:
                        messages = text_to_messages(d["text"])
                    elif "instruction" in d and "output" in d:
                        inp = d.get("input", "")
                        user_content = f"{d['instruction']}\n{inp}".strip() if inp else d["instruction"]
                        messages = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": d["output"]}
                        ]
                    elif "prompt" in d and "response" in d:
                        messages = [
                            {"role": "user", "content": d["prompt"]},
                            {"role": "assistant", "content": d["response"]}
                        ]
                    
                    if messages and any(m.get("role") == "assistant" for m in messages):
                        if all(m.get("content", "").strip() for m in messages):
                            g4_text = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=False
                            )
                            all_texts.append(g4_text)
                            loaded += 1
                    else:
                        skipped += 1
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        log(f"  Loaded {os.path.basename(sf)}: {loaded} examples")
    
    log(f"  Total SFT: {len(all_texts)} examples, skipped: {skipped}")
    return all_texts


def load_dpo_data(tokenizer):
    """Load DPO pairs. Upsample HARD examples with augmentation."""
    pairs = []
    hard_pairs = []
    easy_pairs = []
    hard_raw = []
    
    if not os.path.exists(DPO_FILE):
        log(f"  DPO file not found: {DPO_FILE}")
        return pairs
    
    with open(DPO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                prompt, chosen, rejected = d["prompt"], d["chosen"], d["rejected"]
                difficulty = d.get("difficulty", "easy")
                
                chosen_msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
                rejected_msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
                
                entry = {
                    "prompt_text": tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
                    "chosen_text": tokenizer.apply_chat_template(
                        chosen_msgs, tokenize=False, add_generation_prompt=False),
                    "rejected_text": tokenizer.apply_chat_template(
                        rejected_msgs, tokenize=False, add_generation_prompt=False),
                }
                
                if difficulty == "hard":
                    hard_pairs.append(entry)
                    hard_raw.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                else:
                    easy_pairs.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    
    orig_hard = len(hard_pairs)
    orig_total = len(hard_pairs) + len(easy_pairs)
    orig_ratio = orig_hard / max(orig_total, 1)
    
    # Upsample HARD pairs to target ratio with augmentation
    if orig_hard > 0 and orig_ratio < DPO_HARD_TARGET_RATIO:
        target_hard = int(DPO_HARD_TARGET_RATIO * len(easy_pairs) / (1 - DPO_HARD_TARGET_RATIO))
        augmented_hard = list(hard_pairs)
        
        # Augmentation: prompt prefix variations + tool corruption
        augment_prefixes = ["", "Please, ", "Help me: ", "Need help — ", "Urgently: ", "Question: "]
        tool_corruptions = [
            ("gpu_status()", "docker_ps()"), ("docker_logs(", "disk_usage("),
            ("docker_restart(", "answer("), ("shell(", "http_get("),
            ("http_get(", "shell("), ("memory_usage()", "uptime()"),
        ]
        
        aug_idx = 1
        corruption_idx = 0
        while len(augmented_hard) < target_hard:
            for raw, orig_entry in zip(hard_raw, hard_pairs):
                if len(augmented_hard) >= target_hard:
                    break
                
                if aug_idx % 3 == 0 and corruption_idx < len(tool_corruptions):
                    # Tool corruption: teach model to prefer correct tool usage
                    pattern, replacement = tool_corruptions[corruption_idx % len(tool_corruptions)]
                    corrupted = raw["chosen"].replace(pattern, replacement, 1) if pattern in raw["chosen"] else raw["rejected"]
                    
                    tc_entry = {
                        "prompt_text": tokenizer.apply_chat_template(
                            [{"role": "user", "content": raw["prompt"]}], tokenize=False, add_generation_prompt=True),
                        "chosen_text": tokenizer.apply_chat_template(
                            [{"role": "user", "content": raw["prompt"]}, {"role": "assistant", "content": raw["chosen"]}],
                            tokenize=False, add_generation_prompt=False),
                        "rejected_text": tokenizer.apply_chat_template(
                            [{"role": "user", "content": raw["prompt"]}, {"role": "assistant", "content": corrupted}],
                            tokenize=False, add_generation_prompt=False),
                    }
                    augmented_hard.append(tc_entry)
                    corruption_idx += 1
                else:
                    # Prefix augmentation
                    prefix = augment_prefixes[aug_idx % len(augment_prefixes)]
                    if prefix:
                        aug_prompt = prefix + raw["prompt"]
                        aug_entry = {
                            "prompt_text": tokenizer.apply_chat_template(
                                [{"role": "user", "content": aug_prompt}], tokenize=False, add_generation_prompt=True),
                            "chosen_text": tokenizer.apply_chat_template(
                                [{"role": "user", "content": aug_prompt}, {"role": "assistant", "content": raw["chosen"]}],
                                tokenize=False, add_generation_prompt=False),
                            "rejected_text": tokenizer.apply_chat_template(
                                [{"role": "user", "content": aug_prompt}, {"role": "assistant", "content": raw["rejected"]}],
                                tokenize=False, add_generation_prompt=False),
                        }
                        augmented_hard.append(aug_entry)
                    else:
                        augmented_hard.append(orig_entry)
            aug_idx += 1
        
        pairs = easy_pairs + augmented_hard
        final_hard = len(augmented_hard)
    else:
        pairs = easy_pairs + hard_pairs
        final_hard = orig_hard
    
    final_ratio = final_hard / max(len(pairs), 1)
    log(f"  Loaded DPO: {orig_total} pairs ({orig_hard} HARD = {100*orig_ratio:.1f}%)")
    if final_hard != orig_hard:
        log(f"  Augmented HARD: {orig_hard} → {final_hard} ({100*final_ratio:.1f}% of {len(pairs)} total)")
    return pairs


# ===== TOKENIZATION =====

def tokenize_and_mask(text, tokenizer, max_len):
    """Tokenize and mask: ONLY the LAST assistant response is trained.
    Earlier assistant responses are masked (-100) for cleaner gradient signal.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_len:
        ids = ids[:max_len]
    
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = input_ids.clone()
    
    turn_token_id = tokenizer.convert_tokens_to_ids("<|turn>")
    if turn_token_id == tokenizer.unk_token_id:
        turn_token_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    
    model_token_ids = tokenizer.encode("model", add_special_tokens=False)
    
    model_starts = []
    for i in range(len(ids) - len(model_token_ids)):
        if ids[i] == turn_token_id:
            match = all(i + 1 + j < len(ids) and ids[i + 1 + j] == mt
                       for j, mt in enumerate(model_token_ids))
            if match:
                model_starts.append(i)
    
    if model_starts:
        last_model_start = model_starts[-1]
        mask_end = last_model_start + 1 + len(model_token_ids)
        while mask_end < len(ids) and ids[mask_end] in tokenizer.encode("\n", add_special_tokens=False):
            mask_end += 1
        
        # Skip truncated examples: < 10 trainable tokens = bad signal
        trainable_tokens = len(ids) - mask_end
        if trainable_tokens < 10:
            labels[:] = -100
            return input_ids, labels
        
        labels[:mask_end] = -100
    
    return input_ids, labels


from torch.utils.data import Dataset, DataLoader

class SFTDataset(Dataset):
    def __init__(self, all_input_ids, all_labels):
        self.all_input_ids = all_input_ids
        self.all_labels = all_labels
    def __len__(self):
        return len(self.all_input_ids)
    def __getitem__(self, idx):
        return {"input_ids": self.all_input_ids[idx], "labels": self.all_labels[idx]}


# ===== MODEL LOADING (custom NF4 expert quantization) =====

def _nf4_expert_forward(self, hidden_states, top_k_index, top_k_weights):
    """Patched forward for Gemma4TextExperts with per-expert NF4 + TOKEN-CENTRIC batching.

    SELECTIVE DEQUANT: only top-k active experts dequanted (~8/128).
    TOKEN-CENTRIC: flatten all (token, expert) pairs, group by expert via
    torch.unique_consecutive, then batch-dequant + batched matmul per group.
    Eliminates the standard per-expert loop with torch.where.
    
    Flow:
    1. Flatten routing into (token_idx, expert_id, weight) triples
    2. Sort by expert → unique_consecutive → per-expert groups
    3. Batch-dequant gate_up & down for active experts only
    4. Per group: gather tokens → matmul → scatter results via index_add_
    """
    import bitsandbytes.functional as BF

    final_hidden_states = torch.zeros_like(hidden_states)
    dtype = hidden_states.dtype
    num_tokens = hidden_states.shape[0]
    k = top_k_index.shape[1]

    # Flatten routing: [num_tokens, k] → flat arrays
    flat_token_idx = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, k).reshape(-1)
    flat_expert_idx = top_k_index.reshape(-1)
    flat_weights = top_k_weights.reshape(-1)

    # Group by expert using sort + unique_consecutive (pure torch, no Python dict)
    sorted_expert, sort_order = flat_expert_idx.sort()
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_weights = flat_weights[sort_order]

    unique_experts, counts = sorted_expert.unique_consecutive(return_counts=True)
    if len(unique_experts) == 0:
        return final_hidden_states

    split_sizes = counts.tolist()
    token_groups = sorted_token_idx.split(split_sizes)
    weight_groups = sorted_weights.split(split_sizes)

    for grp_idx, eidx_t in enumerate(unique_experts):
        eidx = eidx_t.item()
        if eidx >= self.num_experts:
            continue

        grp_token_idx = token_groups[grp_idx]
        grp_weights = weight_groups[grp_idx]

        # Dequant only this expert's weights (NF4 → BF16)
        gu = BF.dequantize_4bit(
            self._expert_gate_up_nf4[eidx], self._expert_gate_up_qs[eidx]
        ).reshape(self._expert_gate_up_shape).to(dtype)

        dw = BF.dequantize_4bit(
            self._expert_down_nf4[eidx], self._expert_down_qs[eidx]
        ).reshape(self._expert_down_shape).to(dtype)

        # Gather tokens → MLP → scatter
        current_state = hidden_states[grp_token_idx]
        gate_val, up_val = torch.nn.functional.linear(current_state, gu).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate_val) * up_val
        current_hidden_states = torch.nn.functional.linear(current_hidden_states, dw)
        current_hidden_states = current_hidden_states * grp_weights.unsqueeze(-1)
        final_hidden_states.index_add_(0, grp_token_idx, current_hidden_states.to(final_hidden_states.dtype))

        del gu, dw

    return final_hidden_states


def quantize_experts(model):
    """Quantize Gemma4TextExperts to NF4 — PER-EXPERT for selective dequant.

    Standard bitsandbytes cannot quantize nn.Parameter tensors (only nn.Linear).
    MoE expert weights are stored as nn.Parameter [128, ...] stacked tensors.
    
    We quantize each expert individually with bnb.functional.quantize_4bit(),
    allowing _nf4_expert_forward to dequantize only the top-k active experts
    (~8 out of 128 per token) → ~16x less dequant work per layer.
    """
    import types
    import bitsandbytes.functional as BF

    layer_count = 0
    for name, module in model.named_modules():
        if type(module).__name__ != 'Gemma4TextExperts':
            continue

        layer_count += 1
        num_experts = module.gate_up_proj.data.shape[0]
        
        # Quantize gate_up_proj per-expert: [num_experts, 2*intermediate, hidden]
        gate_data = module.gate_up_proj.data
        expert_gate_up_shape = gate_data.shape[1:]
        expert_gate_up_nf4, expert_gate_up_qs = [], []
        
        for eidx in range(num_experts):
            e_flat = gate_data[eidx].reshape(-1).to('cuda:0').contiguous()
            nf4, qs = BF.quantize_4bit(e_flat, quant_type='nf4', compress_statistics=True)
            expert_gate_up_nf4.append(nf4)
            expert_gate_up_qs.append(qs)
            del e_flat
        del gate_data
        
        module._expert_gate_up_nf4 = expert_gate_up_nf4
        module._expert_gate_up_qs = expert_gate_up_qs
        module._expert_gate_up_shape = expert_gate_up_shape

        # Quantize down_proj per-expert: [num_experts, hidden, intermediate]
        down_data = module.down_proj.data
        expert_down_shape = down_data.shape[1:]
        expert_down_nf4, expert_down_qs = [], []
        
        for eidx in range(num_experts):
            e_flat = down_data[eidx].reshape(-1).to('cuda:0').contiguous()
            nf4, qs = BF.quantize_4bit(e_flat, quant_type='nf4', compress_statistics=True)
            expert_down_nf4.append(nf4)
            expert_down_qs.append(qs)
            del e_flat
        del down_data
        
        module._expert_down_nf4 = expert_down_nf4
        module._expert_down_qs = expert_down_qs
        module._expert_down_shape = expert_down_shape

        # Replace BF16 parameters with tiny placeholders to free memory
        module.gate_up_proj = torch.nn.Parameter(torch.empty(1, device='cpu'), requires_grad=False)
        module.down_proj = torch.nn.Parameter(torch.empty(1, device='cpu'), requires_grad=False)

        # Patch forward
        module.forward = types.MethodType(_nf4_expert_forward, module)

        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        log(f"  Layer {layer_count}/30: experts → NF4. GPU: {alloc_gb:.2f} GiB")

        gc.collect()
        torch.cuda.empty_cache()

    log(f"  Expert quantization complete: {layer_count} layers processed.")


def load_model_q4():
    """Load 26B model with manual NF4 expert quantization.

    Strategy: Load entire model on CPU (BF16, no BnB auto-quant), manually
    quantize expert nn.Parameter weights to NF4 on GPU one layer at a time,
    then move the whole model to GPU.

    This bypasses the BnB limitation where expert weights (nn.Parameter, not
    nn.Linear) are NOT quantized by BnB's automatic 4-bit loading.

    Memory budget (24 GiB GPU):
    - Expert NF4 (30 layers): ~11.3 GiB
    - Attention BF16 (30 layers): ~2.4 GiB
    - Embedding + LM head BF16: ~2.9 GiB
    - CUDA context: ~0.5 GiB
    - Total: ~17.1 GiB → fits with ~7 GiB headroom
    """
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading 26B on CPU (BF16, no BnB auto-quant)...")
    log("  Needs ~52 GiB virtual memory (RAM + swap).")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, attn_implementation="eager",
    )
    log("  CPU load complete.")

    log("Quantizing expert weights to NF4 on GPU...")
    quantize_experts(model)

    log("Moving non-expert modules to GPU (BF16)...")
    model = model.to('cuda:0')

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        log(f"  GPU after full model on CUDA: {alloc:.2f} GB")

    torch.cuda.empty_cache()
    gc.collect()
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapter to model."""
    from peft import get_peft_model, LoraConfig, TaskType

    # Unwrap ClippableLinear if present
    clippable = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Gemma4ClippableLinear":
            clippable += 1
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                setattr(parent, parts[1], module.linear)

    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = TARGET_MODULES_FULL if INCLUDE_MLP_LORA else TARGET_MODULES_ATTENTION
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=target_modules,
        lora_dropout=0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


# ===== PHASE 1: SFT TRAINING =====

def phase_sft():
    log("=" * 60)
    log("PHASE 1: SFT TRAINING (26B-A4B QLoRA)")
    log("=" * 60)
    
    model, tokenizer = load_model_q4()
    model = apply_lora(model)
    
    log("Loading SFT data...")
    texts = load_sft_data(tokenizer)
    if not texts:
        log("ERROR: No SFT data loaded!")
        return False
    random.shuffle(texts)
    
    log(f"Tokenizing {len(texts)} examples (max_seq={MAX_SEQ_LENGTH})...")
    all_input_ids, all_labels = [], []
    too_short, all_masked = 0, 0
    for text in texts:
        ids, labels = tokenize_and_mask(text, tokenizer, MAX_SEQ_LENGTH)
        if len(ids) < 10:
            too_short += 1; continue
        if (labels == -100).all():
            all_masked += 1; continue
        all_input_ids.append(ids)
        all_labels.append(labels)
    log(f"  Tokenized: {len(all_input_ids)} examples, too short: {too_short}, all-masked: {all_masked}")
    
    dataset = SFTDataset(all_input_ids, all_labels)
    dataloader = DataLoader(dataset, batch_size=SFT_BATCH_SIZE, shuffle=True,
                           collate_fn=lambda batch: {k: v.unsqueeze(0) for k, v in batch[0].items()})
    
    from torch.optim import AdamW
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=SFT_LR, weight_decay=0.01, betas=(0.9, 0.999))
    
    total_steps = len(dataloader) * SFT_EPOCHS
    warmup_steps = int(total_steps * SFT_WARMUP_RATIO)
    
    def get_lr(step):
        if step < warmup_steps:
            return SFT_LR * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return SFT_LR * max(0.1, 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress)))
    
    log(f"\n--- SFT TRAINING ---")
    log(f"  Examples: {len(dataset)}, Epochs: {SFT_EPOCHS}")
    log(f"  Batch: {SFT_BATCH_SIZE}, Grad accum: {SFT_GRAD_ACCUM}, Effective: {SFT_BATCH_SIZE * SFT_GRAD_ACCUM}")
    log(f"  Total steps: {total_steps}, Optimizer steps: {total_steps // SFT_GRAD_ACCUM}")
    log(f"  LR: {SFT_LR}, Warmup: {warmup_steps}")
    
    model.train()
    global_step, best_loss, running_loss, loss_count, nan_count = 0, float("inf"), 0, 0, 0
    sft_dir = os.path.join(ADAPTER_DIR, "sft_final")
    os.makedirs(sft_dir, exist_ok=True)
    start_time = time.time()
    
    for epoch in range(SFT_EPOCHS):
        epoch_loss, epoch_steps = 0, 0
        for batch in dataloader:
            global_step += 1
            input_ids = batch["input_ids"].to("cuda")
            labels_t = batch["labels"].to("cuda")
            mm_types = torch.zeros_like(input_ids)
            
            outputs = model(input_ids=input_ids, labels=labels_t,
                          mm_token_type_ids=mm_types, attention_mask=torch.ones_like(input_ids))
            loss = outputs.loss.float() / SFT_GRAD_ACCUM
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1; continue
            
            loss.backward()
            loss_val = outputs.loss.item()
            running_loss += loss_val; loss_count += 1
            epoch_loss += loss_val; epoch_steps += 1
            
            if global_step % SFT_GRAD_ACCUM == 0:
                lr = get_lr(global_step)
                for pg in optimizer.param_groups: pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_MAX_GRAD_NORM)
                optimizer.step(); optimizer.zero_grad()
                
                opt_step = global_step // SFT_GRAD_ACCUM
                if opt_step % 50 == 0:
                    avg_loss = running_loss / max(loss_count, 1)
                    elapsed = time.time() - start_time
                    eta = (total_steps - global_step) / (global_step / elapsed) if elapsed > 0 else 0
                    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    log(f"  Epoch {epoch+1}/{SFT_EPOCHS} | Step {opt_step}/{total_steps//SFT_GRAD_ACCUM} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | VRAM: {vram:.1f}GB | NaN: {nan_count} | ETA: {eta/60:.0f}min")
                    running_loss, loss_count = 0, 0
                    save_checkpoint("sft", opt_step, {"loss": avg_loss, "epoch": epoch+1})
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        log(f"  Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(sft_dir); tokenizer.save_pretrained(sft_dir)
    
    total_time = time.time() - start_time
    log(f"\nSFT COMPLETE — Best loss: {best_loss:.4f}, Time: {total_time/60:.1f} min")
    model.save_pretrained(sft_dir); tokenizer.save_pretrained(sft_dir)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    return True


# ===== PHASE 2: DPO TRAINING =====

def phase_dpo():
    log("=" * 60)
    log("PHASE 2: DPO TRAINING (26B-A4B QLoRA, cached ref logprobs)")
    log("=" * 60)
    
    sft_dir = os.path.join(ADAPTER_DIR, "sft_final")
    if not os.path.exists(sft_dir):
        log(f"ERROR: SFT adapter not found. Run phase 1 first!")
        return False
    
    model, tokenizer = load_model_q4()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sft_dir, is_trainable=True)
    
    # Add MLP LoRA for DPO phase (better tool-calling generation)
    if DPO_INCLUDE_MLP_LORA and not INCLUDE_MLP_LORA:
        from peft import LoraConfig
        mlp_config = LoraConfig(
            r=DPO_MLP_LORA_R, lora_alpha=DPO_MLP_LORA_R,
            target_modules=["gate_proj", "up_proj", "down_proj"], lora_dropout=0, bias="none",
        )
        model.add_adapter("mlp_dpo", mlp_config)
        model.set_adapter(["default", "mlp_dpo"])
    
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    
    pairs = load_dpo_data(tokenizer)
    if not pairs:
        log("ERROR: No DPO data!"); return False
    
    # Precompute reference logprobs (1-time cost → 2x DPO speedup)
    log(f"Precomputing reference logprobs for {len(pairs)} pairs...")
    
    def compute_ref_logprobs(text):
        ids = tokenizer.encode(text, add_special_tokens=False)[:DPO_SEQ_LENGTH]
        t = torch.tensor([ids], device="cuda")
        model.eval()
        with torch.no_grad():
            model.disable_adapter_layers()
            out = model(input_ids=t, mm_token_type_ids=torch.zeros_like(t), attention_mask=torch.ones_like(t))
            model.enable_adapter_layers()
        logits = out.logits[:, :-1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(2, t[:, 1:].unsqueeze(-1)).squeeze(-1).mean().item()
    
    ref_chosen_list, ref_rejected_list = [], []
    ref_start = time.time()
    for idx, pair in enumerate(pairs):
        ref_chosen_list.append(compute_ref_logprobs(pair["chosen_text"]))
        ref_rejected_list.append(compute_ref_logprobs(pair["rejected_text"]))
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - ref_start
            log(f"  Ref cache: {idx+1}/{len(pairs)} ({elapsed/60:.1f}min)")
    
    # Pinned CPU tensors for async GPU transfer
    ref_chosen_cache = torch.tensor(ref_chosen_list, dtype=torch.float32).pin_memory()
    ref_rejected_cache = torch.tensor(ref_rejected_list, dtype=torch.float32).pin_memory()
    log(f"  Reference cache complete in {(time.time()-ref_start)/60:.1f} min")
    log(f"  Each DPO step: 2 forward (policy only) instead of 4")
    
    model.train()
    indices = list(range(len(pairs))); random.shuffle(indices)
    pairs = [pairs[i] for i in indices]
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    ref_chosen_cache = ref_chosen_cache[idx_tensor]
    ref_rejected_cache = ref_rejected_cache[idx_tensor]
    
    from torch.optim import AdamW
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=DPO_LR, weight_decay=0.01)
    total_steps = len(pairs) * DPO_EPOCHS
    
    def get_policy_logprobs(text):
        ids = tokenizer.encode(text, add_special_tokens=False)[:DPO_SEQ_LENGTH]
        t = torch.tensor([ids], device="cuda")
        out = model(input_ids=t, mm_token_type_ids=torch.zeros_like(t), attention_mask=torch.ones_like(t))
        logits = out.logits[:, :-1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(2, t[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
    
    global_step, running_loss, loss_count, best_loss, nan_count = 0, 0, 0, float("inf"), 0
    dpo_dir = os.path.join(ADAPTER_DIR, "dpo_final")
    os.makedirs(dpo_dir, exist_ok=True)
    start_time = time.time()
    
    for epoch in range(DPO_EPOCHS):
        indices = list(range(len(pairs))); random.shuffle(indices)
        epoch_pairs = [pairs[i] for i in indices]
        idx_t = torch.tensor(indices, dtype=torch.long)
        # Async copy with non_blocking — avoids CPU stall
        epoch_ref_chosen = ref_chosen_cache[idx_t].to("cuda", non_blocking=True)
        epoch_ref_rejected = ref_rejected_cache[idx_t].to("cuda", non_blocking=True)
        
        epoch_loss, epoch_steps = 0, 0
        for i, pair in enumerate(epoch_pairs):
            global_step += 1
            chosen_logp = get_policy_logprobs(pair["chosen_text"])
            rejected_logp = get_policy_logprobs(pair["rejected_text"])
            
            chosen_rewards = DPO_BETA * (chosen_logp - epoch_ref_chosen[i])
            rejected_rewards = DPO_BETA * (rejected_logp - epoch_ref_rejected[i])
            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards) / DPO_GRAD_ACCUM
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1; continue
            
            loss.backward()
            running_loss += loss.item() * DPO_GRAD_ACCUM; loss_count += 1
            epoch_loss += loss.item() * DPO_GRAD_ACCUM; epoch_steps += 1
            
            if global_step % DPO_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
                
                opt_step = global_step // DPO_GRAD_ACCUM
                if opt_step % 25 == 0:
                    avg_loss = running_loss / max(loss_count, 1)
                    elapsed = time.time() - start_time
                    eta = (total_steps - global_step) / (global_step / elapsed) if elapsed > 0 else 0
                    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    log(f"  Epoch {epoch+1}/{DPO_EPOCHS} | Step {opt_step}/{total_steps//DPO_GRAD_ACCUM} | "
                        f"DPO Loss: {avg_loss:.4f} | NaN: {nan_count} | VRAM: {vram:.1f}GB | ETA: {eta/60:.0f}min")
                    running_loss, loss_count = 0, 0
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        log(f"  DPO Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(dpo_dir); tokenizer.save_pretrained(dpo_dir)
    
    log(f"\nDPO COMPLETE — Best loss: {best_loss:.4f}, NaN: {nan_count}, Time: {(time.time()-start_time)/60:.1f} min")
    model.save_pretrained(dpo_dir); tokenizer.save_pretrained(dpo_dir)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    return True


# ===== PHASE 3: MERGE TO BF16 =====

def phase_merge():
    log("=" * 60)
    log("PHASE 3: MERGE LoRA → BF16 (CPU, will use swap)")
    log("=" * 60)
    
    dpo_dir = os.path.join(ADAPTER_DIR, "dpo_final")
    sft_dir = os.path.join(ADAPTER_DIR, "sft_final")
    adapter_dir = dpo_dir if os.path.exists(dpo_dir) else sft_dir
    if not os.path.exists(adapter_dir):
        log("ERROR: No adapter found!"); return False
    
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from peft import PeftModel
    
    log("  Loading base model on CPU (bf16)...")
    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    
    log("  Loading & merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    
    model.save_pretrained(MERGED_DIR, max_shard_size="2GB", safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(MERGED_DIR)
    
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Saved! {total_params/1e9:.2f}B params")
    del model, base_model; gc.collect()
    return True


# ===== PHASE 4: GGUF CONVERSION =====

def phase_gguf():
    log("=" * 60)
    log("PHASE 4: GGUF Q4_K_M CONVERSION")
    log("=" * 60)
    
    if not os.path.exists(MERGED_DIR):
        log("ERROR: Merged model not found!"); return False
    
    convert_script = os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(LLAMA_CPP, "build", "bin", "llama-quantize")
    os.makedirs(GGUF_DIR, exist_ok=True)
    
    bf16_gguf = os.path.join(GGUF_DIR, "model-bf16.gguf")
    q4_gguf = os.path.join(GGUF_DIR, "model-q4_k_m.gguf")
    
    log("  Converting HF → BF16 GGUF...")
    result = subprocess.run(
        ["python3", convert_script, MERGED_DIR, "--outfile", bf16_gguf, "--outtype", "bf16"],
        capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr[-500:]}"); return False
    
    log(f"  BF16 GGUF: {os.path.getsize(bf16_gguf)/1e9:.1f} GB")
    
    log("  Quantizing BF16 → Q4_K_M...")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(LLAMA_CPP, "llama-quantize")
    
    result = subprocess.run([quantize_bin, bf16_gguf, q4_gguf, "Q4_K_M"],
                          capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr[-500:]}"); return False
    
    log(f"  Q4_K_M GGUF: {os.path.getsize(q4_gguf)/1e9:.1f} GB")
    os.remove(bf16_gguf)  # Clean up ~50GB intermediate
    return True


# ===== PHASE 5: OLLAMA DEPLOY =====

def phase_ollama():
    log("=" * 60)
    log("PHASE 5: OLLAMA DEPLOY")
    log("=" * 60)
    
    q4_gguf = os.path.join(GGUF_DIR, "model-q4_k_m.gguf")
    if not os.path.exists(q4_gguf):
        log("ERROR: GGUF not found!"); return False
    
    modelfile = f"""FROM {q4_gguf}
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 4096
PARAMETER num_ctx 8192
PARAMETER num_gpu 99
"""
    modelfile_path = os.path.join(BASE_DIR, "Modelfile.gemma4-v6")
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    
    result = subprocess.run(["ollama", "create", "gemma4-finetuned", "-f", modelfile_path],
                          capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr}"); return False
    
    log("PHASE 5 COMPLETE — Model deployed to Ollama")
    return True


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 26B MoE Fine-Tuning Pipeline")
    parser.add_argument("--phase", required=True,
                        choices=["0", "1", "2", "3", "4", "5", "all", "diag"],
                        help="Phase: 0=download, 1=SFT, 2=DPO, 3=merge, 4=GGUF, 5=ollama, all, diag")
    parser.add_argument("--include-mlp", action="store_true",
                        help="Include MLP projections in LoRA (needs more VRAM)")
    args = parser.parse_args()
    
    global INCLUDE_MLP_LORA
    if args.include_mlp:
        INCLUDE_MLP_LORA = True
    
    log("=" * 60)
    log("GEMMA 4 26B MoE FINE-TUNING PIPELINE")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Phase: {args.phase}")
    log(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    log(f"  SFT seq: {MAX_SEQ_LENGTH}, DPO seq: {DPO_SEQ_LENGTH}")
    log(f"  DPO MLP LoRA: {DPO_INCLUDE_MLP_LORA} (r={DPO_MLP_LORA_R})")
    log(f"  DPO HARD target: {DPO_HARD_TARGET_RATIO:.0%}")
    log("=" * 60)
    
    phases = {"0": phase_download, "1": phase_sft, "2": phase_dpo, "3": phase_merge, "4": phase_gguf, "5": phase_ollama}
    
    if args.phase == "all":
        for p in ["0", "1", "2", "3", "4", "5"]:
            if not phases[p]():
                log(f"PHASE {p} FAILED"); sys.exit(1)
    elif args.phase in phases:
        if not phases[args.phase]():
            log(f"PHASE {args.phase} FAILED"); sys.exit(1)
    
    log("\nALL REQUESTED PHASES COMPLETE!")


if __name__ == "__main__":
    main()
