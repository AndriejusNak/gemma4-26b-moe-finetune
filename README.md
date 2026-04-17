# Gemma 4 26B MoE Fine-Tuning on a Single RTX 3090

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GPU: RTX 3090](https://img.shields.io/badge/GPU-RTX%203090%2024GB-green.svg)](https://www.nvidia.com/)
[![VRAM: 18.4 GB](https://img.shields.io/badge/Peak%20VRAM-18.4%20GB-orange.svg)](#vram-budget-actual-measurements)
[![Model: Gemma 4 26B](https://img.shields.io/badge/Model-Gemma%204%2026B%20MoE-red.svg)](https://ai.google.dev/gemma)

> Training a 26-billion parameter Mixture-of-Experts model on consumer hardware —
> and making it work.

## ⚡ Quickstart

```bash
# Clone
git clone https://github.com/AndriejusNak/gemma4-26b-moe-finetune.git
cd gemma4-26b-moe-finetune

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (requires RTX 3090 24GB or similar)
python v6_26b_pipeline.py
```

> **Prerequisites:** NVIDIA GPU with 24 GB VRAM, CUDA 12.8+, Python 3.10+

## 🎯 The Challenge

Google's `gemma-4-26B-A4B-it` is a 25.81B parameter MoE model with 128 experts per layer, top-8 routing, across 30 transformer layers. The conventional wisdom says you need multiple A100s or H100s to fine-tune this. We did it on a single RTX 3090.

## 📊 Results

### SFT Phase — COMPLETE ✅

| Metric | Value |
|--------|-------|
| Training time | 687 min (11.45 hours) |
| Best loss | **1.2146** (epoch 2 average) |
| Lowest step loss | 1.1653 (step 700) |
| Total steps | 750 (2 epochs) |
| NaN explosions | **0** |
| Peak VRAM | 18.4 GB |
| Training data | 6,119 examples (multi-source) |

<details>
<summary>Full SFT Loss Curve (click to expand)</summary>

```
Step  50 | Loss: 6.8887 | LR: 2.00e-5 | VRAM: 18.0GB
Step 100 | Loss: 2.8130 | LR: 1.96e-5 | VRAM: 18.4GB  (-59%)
Step 150 | Loss: 1.7163 | LR: 1.88e-5 | VRAM: 18.4GB  (-39%)
Step 200 | Loss: 1.5580 | LR: 1.75e-5 | VRAM: 18.4GB  (-9%)
Step 250 | Loss: 1.4741 | LR: 1.59e-5 | VRAM: 17.9GB  (-5.4%)
Step 300 | Loss: 1.3663 | LR: 1.40e-5 | VRAM: 18.3GB  (-7.3%)
Step 350 | Loss: 1.3284 | LR: 1.19e-5 | VRAM: 18.0GB  (-2.8%)
--- Epoch 1 avg: 2.3724 --- Saved! ---
Step 400 | Loss: 1.2911 | LR: 9.74e-6 | VRAM: 18.4GB  (-2.8%)
Step 450 | Loss: 1.2570 | LR: 7.56e-6 | VRAM: 18.0GB  (-2.6%)
Step 500 | Loss: 1.2168 | LR: 5.50e-6 | VRAM: 18.4GB  (-3.2%)
Step 550 | Loss: 1.2311 | LR: 3.66e-6 | VRAM: 18.4GB  (+1.2%)
Step 600 | Loss: 1.2355 | LR: 2.12e-6 | VRAM: 18.4GB  (+0.4%)
Step 650 | Loss: 1.2046 | LR: 2.00e-6 | VRAM: 18.3GB  (-2.5%)
Step 700 | Loss: 1.1653 | LR: 2.00e-6 | VRAM: 18.0GB  (-3.3%) ← lowest
Step 750 | Loss: 1.1688 | LR: 2.00e-6 | VRAM: 18.4GB  (+0.3%)
--- Epoch 2 avg: 1.2146 --- BEST! Saved! ---

SFT COMPLETE — Best loss: 1.2146, Time: 687 min, 0 NaN
```
</details>

### DPO Phase — IN PROGRESS 🔄

| Metric | Value |
|--------|-------|
| Dataset | 2,708 pairs (30% HARD augmented) |
| Reference cache | 2,708 pairs pre-computed (79.3 min) |
| Cache effect | 2 forward passes instead of 4 (**2× speedup**) |
| Beta | 0.1 |
| Step 25/507 | **Loss: 0.7241** (sweet spot: 0.5–1.0) |
| Step 50/507 | **Loss: 0.7308** (stable, no divergence) |
| NaN | **0** |
| VRAM | 17.9 GB |
| ETA | ~13 hours |

## 🔧 Key Technical Innovations

### 1. Custom NF4 Expert Quantization

Standard `bitsandbytes` `load_in_4bit` **cannot quantize** MoE expert weights — they're stored as `nn.Parameter`, not `nn.Linear`. We built a custom solution:

```python
# Per-expert NF4 quantization (128 experts × 30 layers × 3 projections)
for layer in range(30):
    gate_up = model.layers[layer].block_sparse_moe.experts  # [128, 2*704, 2816]
    for expert_idx in range(128):
        weight = gate_up[expert_idx]
        quantized, quant_state = bnb.functional.quantize_nf4(weight.data.cuda())
        # Store NF4 weight + state, replace original with tiny placeholder
```

- 11,520 expert weight matrices quantized individually
- Selective dequantization: only top-8 active experts dequantized per token
- Result: **26B model in ~17 GB VRAM** with full gradient support

### 2. Token-Centric Expert Forward Pass

Standard MoE forward pass = Python `for` loop over experts (slow). Our approach:

```python
# Instead of: for expert_id in range(128): if tokens_for_expert...
# We do:
flat_expert_idx = routing_indices.flatten()        # [num_tokens × k]
sorted_idx = flat_expert_idx.argsort()
_, counts = torch.unique_consecutive(flat_expert_idx[sorted_idx], return_counts=True)
# → Each expert's tokens are batched together → single matmul per expert group
```

- Eliminates `torch.where` and `one_hot` per expert
- GPU stays saturated (no fragmented work)
- **~2-4× speedup** vs expert-centric approach

### 3. DPO with CPU-Cached Reference Model

DPO requires 4 forward passes per step (policy + reference × chosen + rejected). On 24 GB, this is impossible with two models loaded.

```python
# One-time cost: pre-compute all reference logprobs
ref_cache = []
with torch.no_grad():
    for chosen, rejected in dataset:
        ref_chosen_logp = compute_logprobs(model, chosen)   # model = frozen policy
        ref_rejected_logp = compute_logprobs(model, rejected)
        ref_cache.append((ref_chosen_logp.cpu(), ref_rejected_logp.cpu()))

# During training: only 2 forward passes needed (policy only)
# ref_cache loaded with non_blocking=True for async CPU→GPU transfer
```

- Pre-computation: 79.3 minutes for 2,708 pairs
- Training: **2 forward passes** instead of 4 = **2× faster**
- Pinned CPU memory (21.2 KB) for zero-copy transfer

### 4. HARD Adversarial DPO Training

Standard DPO: "good answer vs bad answer." Our DPO:

| Category | Count | What it teaches |
|----------|-------|------------------|
| `tool_confusion` | Dynamic | Right tool vs wrong tool for query |
| `multi_tool` | 42 | Multi-step diagnosis chains |
| `overthinking` | 24 | Short efficient answer vs verbose rambling |
| `adversarial` | 18 | Resist user suggesting wrong tool |
| `recovery` | 15 | Adapt when first tool fails |
| `hallucination_guard` | 24 | Factual check vs making up services |
| `wrong_params` | 12 | Correct tool, wrong arguments |
| `edge_case` | 9 | Typos, empty queries, non-existent services |
| + 5 more categories | ... | ... |

**Tool corruption augmentation**: Every 3rd HARD sample has correct reasoning but wrong tool call in the rejected sample — teaching **reasoning correctness**, not just output quality.

### 5. Iterative DPO Pipeline

Post-training failure analysis feeds the next iteration:

```
Train DPO → Deploy → Benchmark → Analyze failures
     ↑                                    ↓
     └── Build new dataset ← Generate targeted HARD pairs
```

Scripts included:
- `failure_to_hard.py` — Classify benchmark failures → targeted HARD DPO pairs
- `tool_confusion_matrix.py` — Analyze expected vs actual tool usage → find systematic errors
- `build_dpo_iteration.py` — Combine datasets with configurable HARD ratio (target: 40-50%)

## 🏗️ Architecture Details

| | |
|---|---|
| Model | google/gemma-4-26B-A4B-it |
| Total Parameters | 25.81B |
| Active per Token | ~4B (top-8 of 128 experts) |
| Hidden Size | 2816 |
| Expert Layers | 30 layers × 128 experts |
| Expert MLP dim | 704 per expert |
| Sliding Window | 1024 |
| Quantization | Custom NF4 (per-expert, manual dequant) |
| LoRA (SFT) | r=16, α=16 — attention only (q,k,v,o_proj) |
| LoRA (DPO) | + r=8, α=8 — MLP (gate,up,down_proj) |
| Hardware | Single RTX 3090 24GB |

### VRAM Budget (Actual Measurements)

```
Expert NF4 (30 layers × 128 experts):     11.3 GiB
Attention BF16 (30 layers):                 2.4 GiB
Embedding + LM head BF16:                  2.9 GiB
CUDA context:                               0.5 GiB
────────────────────────────────────────────────
Model total:                               17.1 GiB
LoRA weights (r=16):                       ~22 MB
AdamW optimizer states:                    ~22 MB
Activations (seq=512, batch=1):           1-3 GiB
────────────────────────────────────────────────
Training peak:                            18.4 GiB
Headroom:                                  5.6 GiB ✅
```

## 🚀 Training Pipeline (6 Phases)

```
Phase 0: Download & Cache
  └─ google/gemma-4-26B-it (~52 GB BF16 safetensors)

Phase 1: SFT with QLoRA                              ✅ COMPLETE
  └─ Manual NF4 expert quantization (11,520 expert matrices)
  └─ Attention-only LoRA (r=16) on 30 layers
  └─ 2 epochs, batch=1, grad_accum=16, LR=2e-5 cosine
  └─ Best loss: 1.2146, 750 steps, 687 min, 0 NaN

Phase 2: DPO with QLoRA                              🔄 IN PROGRESS
  └─ Loads SFT adapter → adds MLP LoRA (r=8)
  └─ Reference logprobs pre-cached on CPU (79.3 min)
  └─ 3 epochs, LR=2e-6, β=0.1, 2708 pairs (30% HARD)
  └─ Step 50: loss 0.7308, 0 NaN, 17.9 GB VRAM

Phase 3: Merge LoRA → BF16 on CPU
  └─ Full model reload + LoRA merge (~54 GB RAM, uses swap)
  └─ 2 GB safetensors shards

Phase 4: GGUF Conversion
  └─ llama.cpp convert_hf_to_gguf.py → F16 GGUF
  └─ llama-quantize → Q4_K_M (~17 GB)

Phase 5: Ollama Deploy + Smoke Test
  └─ Custom Modelfile with calibrated stop tokens
  └─ ollama create → production inference
```

## 🛠️ Technical Challenges Solved

| # | Problem | Solution |
|---|---------|----------|
| 1 | `bitsandbytes` can't quantize `nn.Parameter` | Manual per-expert NF4 with `quantize_nf4()`/`dequantize_nf4()` |
| 2 | MoE expert loop is pure Python (slow) | Token-centric batching with `unique_consecutive` — 2-4× speedup |
| 3 | DPO doubles VRAM (need reference model) | CPU-cached reference logprobs with async `non_blocking` transfer |
| 4 | NaN loss explosions at step ~350 | Three fixes: all-masked filter + FP32 loss + NaN skip counter |
| 5 | DPO length bias (`.sum()` rewards) | Changed to `.mean()` — reward proportional to quality, not length |
| 6 | Multi-turn masking trained on all turns | Mask all except LAST assistant response |
| 7 | PEFT `set_adapter()` rejects lists | Bypass via `model.base_model.set_adapter()` |
| 8 | 26B model in 24 GB | Gradient checkpointing + NF4 experts + attention-only LoRA = 18.4 GB peak |
| 9 | Ollama safetensors converter crashes | Used llama.cpp `convert_hf_to_gguf.py` pipeline |
| 10 | HARD overfitting (repeated templates) | Prefix augmentation + dynamic generation from tool × confusion matrix |

## 🤖 Ops Agent Integration

The fine-tuned models power **Janus Auto** — a self-healing AI operations agent:

- 🔧 69 tools for infrastructure management (Docker, networking, diagnostics)
- 🧠 ReAct agent with native tool calling
- 🔀 SmartRouter with 5 routing paths (FSC, MultiTool, InstantTool, Theory, Full)
- 📚 RAG system with Qdrant (17,159 indexed chunks)
- 🇱🇹 Lithuanian-first AI assistant for homelab infrastructure

## 📈 Previous Results (E4B — 8B dense model)

Before tackling the 26B MoE, we fine-tuned the smaller `gemma-4-E4B-it` (8B):

- SFT + DPO: 932 steps, final DPO loss: 0.0876
- GGUF: BF16 → Q4_K_M (5.07 GB), deployed via Ollama
- Benchmark: **100/100** on internal Hard80 evaluation suite
- Inference: 31-34 tokens/s on RTX 3060

## 💻 Hardware Setup

```
Server: Ubuntu, 27GB RAM + 33GB swap
GPU: NVIDIA RTX 3090 24GB (CUDA 12.8, Driver 570.211.01)
Storage: NVMe SSD
Model cache: ~52 GB on disk (BF16 safetensors)
```

## 📋 Version History

| Version | Key Changes |
|---------|-------------|
| v20 | First working 26B pipeline with manual NF4 (NaN issues) |
| v21 | NaN fix: all-masked filter + FP32 loss + NaN skip |
| v22 | DPO: cached ref logprobs, MLP LoRA, HARD 30%, seq=768 |
| v23 | `.mean()` fix, multi-turn masking, prefix augmentation, per-expert NF4 |
| v24 | Vectorized expert forward (batch-dequant), pinned ref cache |
| **v25** | **Token-centric forward (2-4× speedup), non_blocking DPO, tool corruption HARD, truncated skip** |

## 📁 Files

| File | Description |
|------|-------------|
| `v6_26b_pipeline.py` | Full 6-phase training pipeline (~1700 lines) |
| `data/sft_example.jsonl` | SFT training data format example |
| `data/dpo_example.jsonl` | DPO training data format example |
| `Modelfile.example` | Ollama Modelfile with calibrated stop tokens |
| `requirements.txt` | Python dependencies |

## 📄 License

The training pipeline code is provided for educational purposes. The base model
(`gemma-4-26B-A4B-it`) is subject to [Google's Gemma license](https://ai.google.dev/gemma/terms).

Built with PyTorch, HuggingFace Transformers, PEFT, bitsandbytes, and a healthy
disregard for VRAM requirements.