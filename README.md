# Small Language Model — 110M Parameters

> A decoder-only transformer language model built entirely from scratch, trained on publicly available English text corpora with no pretrained weights, no pretrained tokenizer, and no external model dependencies of any kind.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Tokenizer](#tokenizer)
- [Training Data](#training-data)
- [Training Configuration](#training-configuration)
- [Hardware and Infrastructure](#hardware-and-infrastructure)
- [Weight Initialization](#weight-initialization)
- [Training Dynamics](#training-dynamics)
- [Checkpointing and Persistence](#checkpointing-and-persistence)
- [Inference and Generation](#inference-and-generation)
- [Results and Evaluation](#results-and-evaluation)
- [File Structure](#file-structure)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Reproducing This Work](#reproducing-this-work)
- [License](#license)
- Way to Run it Locally
  

---

## Overview

This repository contains a 110-million-parameter autoregressive language model trained entirely from scratch on publicly available English text. The project demonstrates that a coherent, functional language model can be built without any pretrained weights, pretrained vocabulary, or proprietary datasets — using only open data, open-source tooling, and commodity GPU hardware.

The architecture follows the GPT-2 family of decoder-only transformers with pre-normalization, causal self-attention, and weight-tied input/output embeddings. Training uses mixed-precision (FP16) with automatic loss scaling, gradient accumulation, and a cosine learning rate schedule with linear warmup.

This repository covers **Phase 1: Base Pre-Training** — raw language modelling on 60 million tokens drawn from TinyStories, WikiText-103, and C4. The model at this stage has no name and no instruction-following behaviour. It is a foundational language model — a general-purpose next-token predictor trained to model the statistical structure of English text.

---

## Model Architecture

The model is a standard decoder-only GPT-style transformer. Every component was implemented from scratch in PyTorch — no HuggingFace `transformers` model classes, no external model libraries, no borrowed architectural code.

### Architectural Hyperparameters

| Hyperparameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Number of layers | 12 |
| Hidden dimension (`n_embd`) | 768 |
| Attention heads | 12 |
| Head dimension | 64 (= 768 / 12) |
| FFN intermediate dimension | 3072 (= 4 × 768) |
| Context window (`block_size`) | 512 tokens |
| Vocabulary size | 32,000 |
| Dropout | 0.0 |
| Bias terms | False (all linear layers are bias-free) |
| Normalization | Pre-LayerNorm (applied before each sub-layer) |
| Activation | GELU |
| Weight tying | Input embedding tied to output projection |

### Parameter Count Breakdown

| Component | Shape | Parameters |
|---|---|---|
| Token embedding (`wte`) | 32,000 × 768 | 24,576,000 |
| Position embedding (`wpe`) | 512 × 768 | 393,216 |
| Attention projection (`c_attn`) × 12 | 768 × 2304 | 21,233,664 |
| Attention output (`c_proj`) × 12 | 768 × 768 | 7,077,888 |
| FFN up-projection (`c_fc`) × 12 | 768 × 3072 | 28,311,552 |
| FFN down-projection (`c_proj`) × 12 | 3072 × 768 | 28,311,552 |
| LayerNorm weights × 25 | 768 each | 19,200 |
| LM head | tied to `wte` | 0 (shared) |
| **Total** | | **~110M** |

Weight tying between `wte` and `lm_head` saves 24.6M parameters and typically improves perplexity by forcing the model to learn a unified representation space for input and output tokens. The LM head weight tensor is literally the same memory allocation as the token embedding weight — any gradient update to one propagates to both.

### Attention Mechanism

Causal self-attention is implemented using PyTorch's `scaled_dot_product_attention` with `is_causal=True`, which automatically dispatches to Flash Attention on supported hardware. This avoids materializing the full N×N attention matrix and significantly reduces memory consumption for longer sequences.

For environments where `scaled_dot_product_attention` is unavailable (PyTorch < 2.0), a manual fallback is provided that uses an explicit lower-triangular causal mask registered as a non-parameter buffer:

```python
self.register_buffer(
    "causal_mask",
    torch.tril(torch.ones(block_size, block_size))
          .view(1, 1, block_size, block_size)
)
```

Registering the mask as a buffer rather than a plain tensor ensures it moves to the correct device automatically when `model.to(device)` is called and is excluded from the parameter count.

### Feed-Forward Network

Each transformer block contains a two-layer MLP with a 4× expansion factor:

```
x → Linear(768, 3072) → GELU → Linear(3072, 768) → Dropout → output
```

All linear layers in the FFN are bias-free (`bias=False`). Empirically, removing biases from large linear layers has a negligible effect on model quality while slightly simplifying the parameter count and reducing memory footprint.

### Residual Scaling

Following the GPT-2 technical report, residual projection weights — `c_proj` in the attention module and the down-projection in the FFN — are initialized with a scaled standard deviation:

```python
std = 0.02 / math.sqrt(2 * n_layers)
nn.init.normal_(param, mean=0.0, std=std)
```

With 12 layers, this gives `std ≈ 0.00408`. The factor of 2 accounts for the two residual connections per block (one after attention, one after FFN). This initialization prevents the residual stream variance from growing linearly with depth at initialization, which would otherwise destabilize early training.

---

## Tokenizer

The tokenizer is trained entirely from scratch. No pretrained vocabulary, merge table, or tokenizer weights from any external source are used.

| Property | Value |
|---|---|
| Algorithm | Byte Pair Encoding (BPE) |
| Vocabulary size | 32,000 |
| Pre-tokenizer | ByteLevel (prefix space) |
| Decoder | ByteLevel |
| Minimum token frequency | 2 |
| Special tokens | `<unk>`, `<pad>`, `<bos>`, `<eos>` |
| Training corpus | 150,000 documents sampled from the pre-training datasets |
| Implementation | HuggingFace `tokenizers` library (Rust backend) |

ByteLevel pre-tokenization operates on raw UTF-8 bytes rather than Unicode characters. This means the tokenizer assigns a consistent prefix-space byte to each word boundary and treats every byte value as a valid token building block. The effective coverage is 100% — any input string, regardless of encoding or script, can be tokenized without producing unknown tokens.

A vocabulary of 32,000 is large enough to avoid excessive fragmentation of common English words while remaining small enough that the embedding table does not dominate total parameter count. At 32,000 × 768 = 24.6M, the embedding table accounts for roughly 22% of total parameters — a reasonable proportion for this model size.

---

## Training Data

Three streaming datasets are interleaved in a round-robin fashion to form the pre-training corpus. Documents are tokenized on-the-fly and written into a single memory-mapped binary file using NumPy `memmap` with `uint16` dtype. Memory-mapped access means the training process reads token batches directly from disk without loading the entire corpus into RAM, keeping memory overhead minimal regardless of corpus size.

| Dataset | Description | Role |
|---|---|---|
| `roneneldan/TinyStories` | Synthetically generated short English stories | Grammatical fluency, sentence structure, narrative coherence |
| `wikitext-103-raw-v1` | Raw Wikipedia article text | Factual language, formal prose, encyclopaedic register |
| `allenai/c4` (en) | Colossal Cleaned Crawled Corpus | Scale, diversity, informal and semi-formal web register |

### Corpus Statistics

```
Total tokens tokenized  : 60,000,000
Training split          : 59,400,000  (99%)
Validation split        :    600,000  (1%)
Token dtype             : uint16  (supports vocabulary ≤ 65,535)
Storage format          : Memory-mapped NumPy array
On-disk size            : ~120 MB
```

### Document Packing

Each document is prepended with `<bos>` (beginning-of-sequence) and appended with `<eos>` (end-of-sequence) before tokenization. Documents are concatenated end-to-end without padding tokens. The training context window slides over this continuous token stream, sampling random offsets at batch construction time. This means every token in every batch is a real content token — no compute is wasted on padding.

---

## Training Configuration

### Optimizer

| Parameter | Value |
|---|---|
| Algorithm | AdamW |
| Peak learning rate | 3e-4 |
| Minimum learning rate | 3e-5 |
| LR schedule | Cosine decay with linear warmup |
| Warmup steps | 200 |
| β₁ | 0.9 |
| β₂ | 0.95 |
| ε | 1e-8 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (global L2 norm) |

Weight decay is applied exclusively to parameters with `ndim >= 2` — weight matrices in linear and embedding layers. Biases, LayerNorm scale and shift parameters, and 1D tensors are placed in a separate parameter group with `weight_decay = 0.0`. L2 regularization on these parameters is generally not beneficial and can actively interfere with normalization layer behaviour.

### Learning Rate Schedule

The learning rate follows a cosine decay curve after a linear warm-up phase:

```
lr(step) =
    lr_max × (step / warmup_steps)                          if step < warmup_steps
    lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t))   if warmup_steps ≤ step < max_steps
    lr_min                                                   if step ≥ max_steps

where  t = (step - warmup_steps) / (max_steps - warmup_steps)
```

The warmup prevents large parameter updates in the first 200 steps when the model is far from any reasonable solution. The cosine decay smoothly anneals the learning rate to one-tenth of its peak value by the end of training, encouraging convergence to a tighter minimum.

### Batching

| Parameter | Value |
|---|---|
| Batch size per step | 24 sequences |
| Gradient accumulation steps | 4 |
| Effective batch size (sequences) | 96 |
| Effective batch size (tokens) | 49,152 |
| Sequence length | 512 tokens |

Gradient accumulation divides each logical optimizer step into 4 micro-steps. Each micro-step computes a forward and backward pass with `loss / 4` as the training signal, accumulating gradients in-place across all 4 micro-steps before calling the optimizer. The final accumulated gradient is numerically equivalent to a single forward-backward pass over a batch of 96 sequences — allowing large effective batch sizes without a proportional increase in GPU memory consumption.

### Mixed Precision Training

Training uses PyTorch's Automatic Mixed Precision framework:

- **Forward pass and loss computation** — `torch.float16` for all matrix multiplications and activations
- **Master weights** — maintained in `torch.float32` to preserve numerical precision during optimizer updates
- **Gradient scaling** — `torch.amp.GradScaler` dynamically scales the loss to prevent FP16 gradient underflow, then unscales before gradient clipping

FP16 tensor operations deliver substantially higher throughput than FP32 on modern GPU hardware due to dedicated half-precision execution units. The GradScaler handles the numerical stability issues that would otherwise arise from FP16's limited dynamic range during backpropagation.

### Multi-GPU Parallelism

Training uses `torch.nn.DataParallel` to distribute computation across two GPUs. The input batch is split along the batch dimension, each GPU processes its half independently through a model replica, and gradients are reduced (averaged) across GPUs before the optimizer step. The `loss.mean()` call on the returned loss tensor correctly aggregates the per-GPU scalar losses into a single training signal.

---

## Hardware and Infrastructure

| Specification | Value |
|---|---|
| Number of GPUs | 2 |
| GPU parallelism | `torch.nn.DataParallel` |
| Training duration | 110 minutes |
| Approximate throughput | ~540,000 tokens per minute |

All training is fully resumable from checkpoint — if the training process is interrupted for any reason, it can be restarted and will automatically continue from the last saved step without loss of progress.

---

## Weight Initialization

| Component | Initialization |
|---|---|
| Linear weight matrices | Normal(μ=0, σ=0.02) |
| Embedding matrices | Normal(μ=0, σ=0.02) |
| Residual projections | Normal(μ=0, σ=0.02/√(2×N_layers)) |
| Bias terms | Zeros (where applicable; most layers use `bias=False`) |
| LayerNorm weight (γ) | Ones |
| LayerNorm bias (β) | Zeros |
| LM head | Tied to `wte`; shares initialization |

The standard deviation of 0.02 for weight matrices is inherited from the original GPT-2 implementation. It keeps the initial activations in a reasonable range given the residual architecture — small enough to prevent saturation, large enough to break symmetry.

LayerNorm is initialized with γ=1 and β=0, the identity transformation. Any deviation from identity is learned during training and represents a genuine signal rather than an initialization artifact.

---

## Training Dynamics

### Loss Progression

```
Iteration     0  →  loss 10.5443  (≈ ln(32,000) — theoretical random baseline)
Iteration    80  →  loss 6.6877   (basic n-gram patterns emerging)
Iteration   500  →  loss ~5.x     (first checkpoint; word-level coherence)
Iteration  3000+ →  loss 3.2209   (final; sentence-level coherence achieved)
```

A randomly initialized model over a vocabulary of V tokens produces cross-entropy loss of approximately `ln(V)`. For a 32,000-token vocabulary, this is `ln(32,000) ≈ 10.37`. The observed starting loss of 10.54 is consistent with this theoretical baseline — confirming that no pretrained weights were silently incorporated at initialization.

The loss drop from 10.54 to 6.69 in the first 80 steps reflects the model rapidly learning high-frequency unigram and bigram statistics. The slower descent from iteration 500 onward corresponds to the model learning longer-range syntactic and semantic patterns that require more parameter updates to encode.

### Perplexity

```
Perplexity = exp(cross_entropy_loss)

Iteration     0  →  ppl 37,891   (random baseline)
Iteration  final →  ppl 25.05    (trained)
```

Perplexity of 25.05 on a 32,000-vocabulary model means the model assigns the true next token a probability roughly equivalent to choosing uniformly from 25 candidates — down from 32,000 candidates at initialization. This represents genuine learned compression of the English language distribution.

---

## Checkpointing and Persistence

Every checkpoint stores the complete training state required for exact resumption:

```python
{
    "model"    : raw_model.state_dict(),     # learned weights
    "optimizer": optimizer.state_dict(),     # Adam moment estimates
    "scaler"   : scaler.state_dict(),        # AMP loss scale history
    "config"   : vars(cfg),                  # architectural hyperparameters
    "iter"     : current_step,               # step counter
    "val_loss" : best_val_loss,              # best observed validation loss
}
```

Checkpoints are written conditionally (only when validation loss improves) at every 500 training steps, plus an unconditional final save when the training time budget is exhausted. The conditional write prevents a degraded checkpoint from overwriting a better one in the event of late-stage overfitting.

On resumption, the training cell detects an existing checkpoint and restores all six fields before entering the training loop — including optimizer momentum buffers and AMP scaler state. A resumed run is numerically indistinguishable from an uninterrupted run past the first ~50 warm-up steps for the optimizer.

---

## Inference and Generation

Token sampling at inference time supports four composable decoding strategies:

**Temperature scaling**
```
logits_adjusted = logits / temperature
```
Temperature < 1.0 sharpens the distribution, concentrating probability mass on high-confidence tokens and producing more deterministic, factual output. Temperature > 1.0 flattens the distribution, increasing diversity and creativity at the cost of coherence.

**Top-k filtering**
Restricts sampling to the k tokens with highest logit values. All tokens outside the top-k are assigned `-inf` before softmax, effectively removing them from the candidate set. Top-k provides a hard floor on candidate quality regardless of the overall distribution shape.

**Nucleus (top-p) sampling**
Restricts sampling to the smallest subset of tokens whose cumulative probability mass exceeds p. Unlike top-k, the size of the candidate set adapts dynamically — on a peaked distribution (high-confidence prediction), only a few tokens are needed to exceed the threshold; on a flat distribution, many are included. This avoids both over-restriction and over-diversity across varying prediction confidences.

**Repetition penalty**
Divides the logit of any token that has appeared in the last 64 generated positions by a penalty factor greater than 1.0. This discourages the repetitive output loops common in smaller language models without hard-constraining the vocabulary.

### Default Generation Parameters

| Parameter | Value |
|---|---|
| Temperature | 0.80 |
| Top-k | 40 |
| Max new tokens | 150 |

---

## Results and Evaluation

### Quantitative

| Metric | Value |
|---|---|
| Final training loss | 3.2209 |
| Final training perplexity | 25.05 |
| Training tokens | 60,000,000 |
| Training duration | 110 minutes |
| Theoretical random baseline loss | 10.37 |
| Total loss reduction | 7.15 nats |

### Qualitative Observations

The model at this stage — without any instruction fine-tuning — exhibits the following characteristics:

**Strengths:**
- Grammatically coherent English sentences across diverse topics
- Correct sentence-level and paragraph-level flow
- Reasonable topic continuity within a passage
- Clean handling of punctuation, capitalisation, and basic syntax

**Weaknesses:**
- Significant factual hallucination, particularly on specific numerical claims, named entities, and domain-specific facts
- No instruction-following capability — the model continues text rather than responding to prompts
- Repetitive phrasing in longer outputs, especially past 100 generated tokens
- Vocabulary co-occurrence artifacts from the training distribution (words that frequently co-occur in the training data appear together even when semantically inappropriate)

These characteristics are expected and consistent with a base language model at this parameter count and training token volume. The factual weaknesses are addressed in subsequent fine-tuning phases (not included in this repository release).

### Comparison

| Model | Parameters | Training tokens | Notes |
|---|---|---|---|
| GPT-2 Small | 117M | 40 billion | OpenAI, 2019 |
| **This model** | **110M** | **60 million** | **From scratch, this repo** |

The parameter count is comparable to GPT-2 Small. The training token count is approximately 667× smaller, which explains the quality gap on knowledge-intensive tasks. The architecture and training pipeline are sound — the limiting factor is data volume, not model design.

---

## File Structure

```
project root
│
├── base_training_resumable.py    # Complete training script (single cell)
│
├── model.pt                      # Trained checkpoint
│   └── keys: model, optimizer, scaler, config, iter, val_loss
│
├── tokenizer.json                # BPE tokenizer (32,000 vocab)
│
└── data/
    ├── train.bin                 # Token corpus (60M tokens, uint16)
    └── train.bin.meta            # Corpus metadata (token count, doc count)
```

---

## Limitations

**Training data volume.** 60 million tokens is insufficient to encode broad world knowledge reliably. GPT-2 was trained on 40 billion tokens — 667× more. Topics that appear rarely or not at all in the training corpus will produce hallucinated output.

**Model scale.** At 110M parameters, this model is below the empirical threshold at which multi-step reasoning, arithmetic, and structured planning reliably emerge. These capabilities generally appear at 1B+ parameters with sufficient training data. Do not rely on this model for arithmetic, logical deduction, or code generation.

**No instruction tuning.** This is a base language model — a text continuation engine. It does not follow instructions, answer questions in a structured way, or refuse harmful requests. Instruction-following behaviour requires a separate supervised fine-tuning phase not included in this release.

**Fixed context window.** The 512-token context window limits utility for long-document tasks, extended dialogue, and retrieval-augmented generation patterns that require fitting long passages in context.

**Hallucination.** The model produces confident-sounding text regardless of factual accuracy. All factual claims in model output should be independently verified. This is an inherent property of base language models at this scale, not a bug.

**No safety alignment.** The model has not undergone any safety training, constitutional AI, or RLHF process. It will complete any text prefix including harmful content. It is released for research and educational purposes only.

---

## Roadmap

- [ ] Extend context window to 1024 tokens via positional embedding interpolation
- [ ] Scale pre-training to 500M+ tokens across additional data sources
- [ ] Supervised fine-tuning phase — instruction following and factual grounding
- [ ] Continued pre-training on Wikipedia and long-form book corpora
- [ ] Gradient checkpointing to enable larger batch sizes within memory constraints
- [ ] Standard benchmark evaluation — HellaSwag, PIQA, BoolQ, WinoGrande
- [ ] Rotary Position Embeddings (RoPE) to improve generalisation to longer sequences
- [ ] Web-based inference interface

---

## Reproducing This Work

### Requirements

```
Python        >= 3.8
PyTorch       >= 2.0
datasets      >= 2.14.0
tokenizers    >= 0.5.0
numpy
```

Two GPUs with at least 8 GB VRAM each are recommended. The training script will run on a single GPU with reduced throughput — increase `grad_accum` to maintain the same effective batch size.

### Steps

1. Clone this repository
2. Install dependencies: `pip install torch datasets tokenizers numpy`
3. Paste or run `base_training_resumable.py`
4. Training runs for approximately 110 minutes and saves `model.pt` on completion
5. The script automatically loads the trained model and opens a text generation interface when training finishes

The script is fully self-contained. It handles tokenizer training, corpus assembly, model construction, training, checkpointing, and inference loading in sequence. If interrupted at any point, re-running the script will resume from the last saved checkpoint automatically.

```
Expected output at completion:
  ✓ Final model saved  → model.pt
  ✓ Tokenizer          → tokenizer.json
  ✓ Stopped at iter    : ~3200
  ✓ Best val loss      : 3.22
```

---

## License

This project and all associated training code are released under the MIT License. Model weights are released for research and educational use only. No warranties are made regarding the factual accuracy, safety, or fitness for purpose of any outputs produced by this model.

---

*Built entirely from scratch. No pretrained weights. No pretrained tokenizer. No proprietary data.*
## Road map to run it locally.
First downlaod the files "model.pt", "run.py" and "tokenizer.json" then cretae a separte folder with these three components and install pythion in your pc and then install all required packages and then
open terminal in that specific folder and run this command "python run.py" wai for some time and you can test how the model predicts coherent 
words.
