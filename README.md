# GPT-2 Inference Engine from Scratch (C++)
Despite being an avid reader, I've always learned best by _doing_ rather than poring over heavy theory without end. 

First up in my series of **TWIL aka "This Week I Learnt"**: Building a complete GPT-2 inference engine in C++, one milestone at a time.


## Milestones

| # | What | Status |
|---|------|--------|
| 1 | Tensor class + math primitives (matmul, softmax, layernorm, gelu) | ✅ Done |
| 2 | GPT-2 forward pass (full architecture, random weights, shape-verified) | ✅ Done |
| 3 | Real weight loading + BPE tokeniser + end-to-end text generation | TODO |

---

## Milestone 1: Tensor & Math Primitives

During the inital MLsys classes, it took me a hot minute (+pen/paper) to grasp optimization techniques for matrix multiplication properly. Therefore, it felt befitting to start with the basics and flesh out the underlying ops as a first step towards writing this inference engine.

`tensor.h` — A `Tensor` struct (flat `float` storage + shape) and the four math ops every GPT-2 layer needs: `matmul`, `softmax`, `layernorm`, `gelu`. Every function is tested against hand-verifiable expected values.

```bash
make test_tensor   # 22/22 tests
```

---

## Milestone 2: GPT-2 Forward Pass

`gpt2.h` — The complete GPT-2 (small) forward pass, built from the M1 primitives:

```
token_ids  →  embedding lookup + positional encoding
           →  12 × TransformerBlock [LayerNorm → Attention → Residual → LayerNorm → FFN → Residual]
           →  final LayerNorm
           →  LM head (x @ wte.T, weight-tied)
           →  logits {seq_len, vocab_size=50257}
```

Key things implemented and verified:
- **Multi-head self-attention** with causal mask (future tokens blocked by −∞ before softmax)
- **Scaled dot-product attention**: scores divided by √d_head for numerical stability  
- **Weight tying**: LM head reuses the token embedding matrix transposed
- **Pre-LN**: LayerNorm applied *before* each sub-layer (GPT-2's training-stable variant)
- **Residual connections** on both attention and FFN sub-layers

```bash
make test_gpt2     # 31/31 tests — causal mask, shape checks, 124M param count
```

### Key architectural constants (GPT-2 small)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `vocab_size` | 50257 | BPE vocabulary |
| `d_model` | 768 | Token vector dimension |
| `n_layers` | 12 | Stacked transformer blocks |
| `n_heads` | 12 | Parallel attention heads |
| `d_head` | 64 | Per-head dimension (`d_model / n_heads`) |
| `d_ff` | 3072 | FFN inner dimension (`4 × d_model`) |
| `max_seq` | 1024 | Maximum context length |

### Build & run all tests
```bash
make test
```

### References
- [GPT-2 Paper — Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated GPT-2 — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- [GPT-2 architecture hyperparameters — Keras Hub](https://keras.io/keras_hub/api/models/gpt2/gpt2_backbone/)
