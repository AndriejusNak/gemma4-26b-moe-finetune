# Gemma 4 26B MoE Fine-Tuning on a Single RTX 3090

> **Training a 26-billion parameter Mixture-of-Experts model on consumer hardware — and making it work.**

## The Challenge

Google's `gemma-4-26B-A4B-it` is a **25.81B parameter MoE model** with 128 experts per layer, top-8 routing, across 30 transformer layers. The conventional wisdom says you need multiple A100s or H100s to fine-tune this. We did it on a **single NVIDIA RTX 3090 (24 GB VRAM)**.

## Key Achievements

### Custom NF4 Expert Quantization
- Standard `bitsandbytes` cannot quantize MoE expert weights stored as `nn.Parameter` tensors
- Built **custom per-expert NF4 quantization** that dequantizes individual experts on-the-fly during forward pass
- Each expert (128 per layer × 30 layers = 3,840 expert groups) stored as NF4 with per-block quantization
- Result: **26B model loads in ~18 GB VRAM** with full gradient checkpointing

### Token-Centric Expert Forward Pass
- Standard MoE forward: loop over experts, gather tokens → slow Python loop
- Our approach: **flatten all routing decisions → `torch.unique_consecutive` sort → per-group batched matmul**
- Tokens destined for the same expert are batched together, eliminating the expert-loop overhead
- Custom `_nf4_expert_forward()` replaces the model's native expert dispatch

### DPO with Reference Model Caching
- DPO requires forward passes through a frozen reference model — doubles VRAM requirements
- Solution: **pre-compute all reference log-probabilities on CPU**, cache as tensors
- During training, load cached refs with `non_blocking=True` for async CPU→GPU transfer
- Enables DPO on 24 GB hardware without model swapping

### HARD Tool Corruption Augmentation
- Training data augmented with intentionally corrupted tool calls
- Model learns to recover from malformed JSON, truncated arguments, wrong tool names
- **30% of DPO dataset** is hard-negative examples with corrupted tool patterns

## Architecture Details

| Component | Specification |
|-----------|---------------|
| Model | `google/gemma-4-26B-A4B-it` |
| Total Parameters | 25.81B |
| Active per Token | ~4B (top-8 of 128 experts) |
| Expert Layers | 30 layers × 128 experts |
| Quantization | Custom NF4 (per-expert, manual dequant) |
| LoRA Rank | r=16, α=16 (attention), r=8 (MLP in DPO) |
| Hardware | Single RTX 3090 24GB |
| VRAM Usage | 18.0–20.9 GB (with gradient checkpointing) |
| Training Data | 6,006 SFT examples, multi-source |

## Training Pipeline (5 Phases)

```
Phase 1: SFT with QLoRA
  └─ Custom NF4 loading → attention-only LoRA → manual training loop
  └─ 2 epochs, batch=1, grad_accum=16, effective_batch=16
  └─ 750 optimizer steps, LR=2e-5 with cosine schedule

Phase 2: DPO with QLoRA
  └─ Loads SFT adapter → adds MLP LoRA (r=8) for tool generation
  └─ Reference model cached on CPU (non_blocking transfer)
  └─ 3 epochs, LR=2e-6, β=0.1, HARD augmentation 30%

Phase 3: Merge LoRA → BF16 on CPU
  └─ Full model reload in BF16 (uses 27GB RAM + swap)
  └─ LoRA weights merged into base model

Phase 4: GGUF Conversion (Q4_K_M)
  └─ llama.cpp convert_hf_to_gguf.py → BF16 GGUF
  └─ llama-quantize → Q4_K_M (~5 GB final)

Phase 5: Ollama Deploy + Smoke Test
  └─ Custom Modelfile → ollama create → production ready
```

## Live Training Progress (v25)

```
[2026-04-16 16:34:40] Step  50/750 | Loss: 6.8887 | VRAM: 18.0GB | NaN: 0
[2026-04-16 17:20:59] Step 100/750 | Loss: 2.8130 | VRAM: 18.4GB | NaN: 0
```

**Loss dropped from 6.89 → 2.81 in the first 100 steps.** Zero NaN explosions. Stable VRAM. Training is running right now.

## Previous Gemma 4 Results (E4B model)

Before tackling the 26B MoE, we successfully fine-tuned the smaller `gemma4:e4b` (9.6B) model:

- **SFT + DPO training**: 932 steps, final DPO loss: 0.0876
- **Manual CPU merge**: 442/442 LoRA pairs → 14.89 GB BF16 safetensors
- **GGUF pipeline**: BF16 → Q4_K_M (5.07 GB), deployed via Ollama
- **Benchmark**: 100/100 on our internal Hard80 evaluation suite
- **Inference**: 31-34 tokens/s on RTX 3060, Lithuanian language fluency preserved

## Ops Agent Integration

The fine-tuned models are deployed in **Janus Auto** — a self-healing AI operations agent:

- **69 tools** for infrastructure management (Docker, networking, diagnostics)
- **ReAct agent** with native tool calling
- **SmartRouter** with 5 routing paths (FSC, MultiTool, InstantTool, Theory, Full)
- **RAG system** with Qdrant (17,159 indexed chunks)
- Lithuanian-first AI assistant for homelab infrastructure

## Technical Challenges Solved

1. **`bitsandbytes` can't auto-quantize `nn.Parameter`**: Built manual NF4 codec with `bnb.functional.quantize_nf4()` / `dequantize_nf4()`
2. **Ollama's safetensors converter crashes on Gemma4**: Used llama.cpp `convert_hf_to_gguf.py` pipeline instead
3. **Gemma4 thinking mode eats all tokens**: Disabled with `"think": False` in Ollama payload
4. **MoE expert loop is pure Python (slow)**: Token-centric batching with `unique_consecutive` eliminates the loop
5. **DPO doubles VRAM**: CPU-cached reference logprobs with async transfer
6. **26B in 24GB**: Gradient checkpointing + NF4 experts + attention-only LoRA = 18 GB peak

## Hardware Setup

```
Server: Ubuntu, 27GB RAM + 33GB swap
GPU: NVIDIA RTX 3090 24GB (CUDA 12.8)
Storage: NVMe SSD
Model cache: ~52 GB on disk (BF16 safetensors)
```

## Version History

| Version | Key Changes |
|---------|-------------|
| v20 | Initial 26B pipeline with custom NF4 |
| v21 | Expert forward optimization |
| v22 | DPO HARD augmentation |
| v23 | Reference model CPU caching |
| v24 | Stability improvements |
| v25 | Token-centric forward, non_blocking DPO, tool corruption augmentation, truncated skip |

## License

The training pipeline code is provided for educational purposes. The base model (`gemma-4-26B-A4B-it`) is subject to [Google's Gemma license](https://ai.google.dev/gemma/terms).

---

*Built with PyTorch, HuggingFace Transformers, PEFT, bitsandbytes, and a healthy disregard for VRAM requirements.*
