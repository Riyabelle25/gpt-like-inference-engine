# GPT-2 Inference Engine from Scratch (C++)
Despite being an avid reader, I've always learned best by _doing_ rather than poring over heavy theory without end. 

First up in my series of **TWIL aka "This Week I Learnt"**: Building a complete GPT-2 inference engine in C++, one milestone at a time.


## Milestones

| # | What | Status |
|---|------|--------|
| 1 | Tensor class + math primitives (matmul, softmax, layernorm, gelu) | ✅ Done |
| 2 | GPT-2 forward pass (full architecture, random weights, shape-verified) | ✅ Done |
| 3 | Weight loading, BPE tokeniser, KV-cache, autoregressive decoding | ✅ Done |

---

## Build

```bash
make all      # builds test_tensor, test_gpt2, and the gpt2 inference binary
make test     # runs all unit tests (53/53)
```

---

## Run

### 1. Download real GPT-2 weights

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai-community/gpt2', local_dir='./gpt2-weights')
"
```

This downloads `model.safetensors`, `Merges.txt`, and `vocab.json` into `./gpt2-weights/`.

### 2. Generate text

```bash
# Greedy decoding
./gpt2 ./gpt2-weights "The history of artificial intelligence" 60

# Top-k sampling (k=40, temperature=0.8)
./gpt2 ./gpt2-weights "The history of artificial intelligence" 60 topk 40 0.8

# Top-p (nucleus) sampling (p=0.9)
./gpt2 ./gpt2-weights "The history of artificial intelligence" 60 topp 0.9 1.0
```

---

## Architecture overview

```
Prompt text
    ↓  [tokenizer.h]  BPE encode -> token IDs
    ↓  [loader.h]     prefill: full forward pass, populate KV-cache
    ↓  [kvcache.h]    decode loop: one token at a time, O(N) via cache
    ↓  [tokenizer.h]  BPE decode -> text
Generated text (streamed token by token)
```

---

## File map

| File | What it does |
|------|-------------|
| `tensor.h` | Flat float Tensor + matmul, softmax, layernorm, gelu |
| `gpt2.h` | Full GPT-2 forward pass: embeddings, attention, FFN, LM head |
| `loader.h` | Parse safetensors binary format; load + transpose Conv1D weights |
| `tokenizer.h` | BPE tokeniser: byte encoder, merge rules, encode/decode |
| `kvcache.h` | KV-cache, prefill(), decode_step(), greedy/top-k/top-p sampling |
| `main.cpp` | CLI: load -> tokenize -> prefill -> decode loop |

---

## Key design decisions

**Weight transpose (`loader.h`)**
OpenAI trained GPT-2 with TensorFlow's `Conv1D`, which stores weight matrices as `{in_dim, out_dim}`. Our `matvec(W, x)` convention requires `{out_dim, in_dim}`. Four weight tensors per block are transposed after loading: `c_attn.weight`, `c_proj.weight`, `mlp.c_fc.weight`, `mlp.c_proj.weight`. Getting this wrong produces numerically plausible but completely incorrect outputs — one of the most common bugs in from-scratch GPT-2 implementations.

**KV-cache memory layout (`kvcache.h`)**
Each layer's cache is a pre-allocated `{max_seq, d_model}` tensor (one for K, one for V). New tokens append one row; the cached rows are never moved. For GPT-2 small: `2 × 12 × 1024 × 768 × 4 bytes ≈ 75 MB`, allocated once at startup. Zero heap traffic on the hot decode path.

**Prefill vs. decode split**
Prefill runs the full multi-token `attention()` (M2's implementation) to process the prompt in one shot, which is more efficient than decoding token-by-token. It simultaneously populates the KV-cache. Decode then runs `attention_cached()` which only computes Q for the new token and scores it against all cached K/V.

**BPE tokeniser (`tokenizer.h`)**
Loads `encoder.json` (token->ID map) and `vocab.bpe` (50,000 merge rules). Applies the byte encoder so all 256 raw byte values map to printable characters before BPE, then greedily applies merges in priority order.

---

## References
- [GPT-2 Paper — Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated GPT-2 — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- [Karpathy, "Let's Reproduce GPT-2"](https://towardsdatascience.com/line-by-line-lets-reproduce-gpt-2-section-1)
- [safetensors format spec](https://huggingface.co/docs/safetensors)