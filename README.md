# LoRA from Scratch — Understanding Low-Rank Adaptation at First Principles

*Built from scratch to understand how LoRA works at the matrix level — no HuggingFace, no PEFT library, just PyTorch and linear algebra.*

---

I built this to understand LoRA from absolute first principles — not just the theory, but how the weights actually get trained, saved, and reused. No libraries, no abstractions, just raw PyTorch and matrix math.

## What is LoRA?

Large language models have billions of parameters. Fine-tuning all of them is expensive and slow. LoRA solves this by **freezing the original model weights** and adding two small trainable matrices (A and B) alongside existing layers. These matrices take the same input, compute a small correction, and add it back to the original output. Because they are "low-rank" (very skinny), they have 99% fewer parameters to train, making the process fast and memory-efficient.

```
delta W = B @ A
```

- **A**: shape `(rank, in_features)` — compresses the input into a small bottleneck
- **B**: shape `(out_features, rank)` — expands it back to the original dimension
- **rank**: a tiny number (e.g., 8) compared to the original dimensions (e.g., 4096)

## How the Code Works

### 1. The LoRA Layer

```python
self.linear = nn.Linear(in_f, out_f)       # Frozen base layer
self.linear.weight.requires_grad = False    # No gradients for base weights

self.lora_A = nn.Parameter(torch.randn(rank, in_f))   # Trainable matrix A
self.lora_B = nn.Parameter(torch.randn(out_f, rank))   # Trainable matrix B
```

**Forward pass:**
```python
base_output = self.linear(x)                          # Original frozen computation
lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) # LoRA path
return base_output + lora_output * self.scaling        # Combined result
```

### 2. Why We Use Transpose (`.t()`) Instead of Flipping the Shapes

One of the first things I questioned: why not just define A as `(in_f, rank)` and B as `(rank, out_f)` to skip the transpose entirely?

**Reason 1 — Initialization correctness.** PyTorch's `kaiming_uniform_` assumes the second dimension is the input size ("fan-in") to calculate how large the random numbers should be. If A is `(in_f, rank)`, PyTorch thinks the fan-in is `rank` (e.g., 8) instead of `in_features` (e.g., 4096). The random numbers would be ~22x too large (`sqrt(4096)/sqrt(8)`), causing exploding gradients before training even starts.

**Reason 2 — Clean weight merging.** The base weight W is stored as `(out_features, in_features)`. With our shapes, `B @ A` naturally produces `(out_features, in_features)` — a direct match. No extra reshaping needed for `W_new = W_base + B @ A`.

### 3. Initialization Strategy

```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Smart random init
nn.init.zeros_(self.lora_B)                              # Zeros
```

- **Matrix A** gets Kaiming initialization — prevents signals from exploding or vanishing through layers.
- **Matrix B** starts at **zeros**, so `B @ A = 0` at the start. The LoRA layer initially has zero effect. The model begins from its original behavior and gradually learns the adaptation.

### 4. The Scaling Factor — Why It Exists

This was a big "why" for me — why not just add `B @ A` directly to the output?

```python
self.scaling = self.lora_alpha / rank  # e.g., 16 / 8 = 2.0
```

**The problem it solves:** When you change the rank, the magnitude of the LoRA output changes too. With `rank=8`, you sum 8 products to get each value. With `rank=100`, you sum 100 products — the output is naturally much "louder." Without scaling, this would shock the model's layers and cause NaN values or exploding gradients.

**A concrete example:** Assume input `x = 1.0`, and all values in A and B are initialized to `0.1`:

| Rank | Raw LoRA Output | Scaling (`16/r`) | Final Output |
|------|-----------------|-------------------|--------------|
| r=2  | `(0.1*0.1)*2 = 0.02` | 8.0 | **0.16** |
| r=100 | `(0.1*0.1)*100 = 1.0` | 0.16 | **0.16** |

The final output stays **0.16** regardless of rank. The scaling factor keeps the "volume" stable so you can change the rank without retuning your learning rate.

### 5. Rank vs Alpha — Two Independent Knobs

Another thing I initially got wrong: I assumed higher rank = higher output values, so why bother scaling it down? Turns out rank and alpha control completely different things.

**Rank (r)** controls *what* the model can learn (capacity/resolution):
- `r=8`: Like an 8x8 pixel image — captures broad patterns (tone, sentiment)
- `r=100`: Like a 100x100 pixel image — captures fine details (specific syntax, medical terms)

Both images can have the same brightness (scaling). Higher rank = more detail, not louder signal.

**Alpha** controls *how much* to trust the new learning (strength):

| Situation | Adjust | Why |
|---|---|---|
| Underfitting — not learning the task | Increase alpha | Trust new data more than pre-trained weights |
| Overfitting — forgetting general knowledge | Decrease alpha | Keep new learning as a light suggestion |
| Very different domain (e.g., English to Medical) | High alpha + High rank | Need both capacity and strength to override base patterns |

**Starting point:** Set `alpha = 2 * rank` (e.g., `r=16, alpha=32`). Tune from there.

### 6. Dimension Walkthrough (From This Implementation)

I traced every shape through the forward pass to make sure I understood where the squeeze and expand actually happens. Using the actual values from the code: `embed_dim=128`, `hidden_dim=64`, `rank=8`, `batch=4`:

| Matrix | Shape | Dimensions | Parameters |
|---|---|---|---|
| x (input) | [batch, in_features] | [4, 128] | — |
| W (frozen base) | [out_features, in_features] | [64, 128] | 8,192 |
| A (adapter) | [rank, in_features] | [8, 128] | 1,024 |
| B (adapter) | [out_features, rank] | [64, 8] | 512 |

**The math flow — a squeeze-and-expand funnel:**

```
Step 1: x @ A.T     →  [4, 128] @ [128, 8]  →  [4, 8]   (squeezed into rank)
Step 2: result @ B.T →  [4, 8]   @ [8, 64]   →  [4, 64]  (expanded back)
Step 3: base + lora  →  [4, 64]  + [4, 64]   →  [4, 64]  (shapes match, add them)
```

**Parameter savings:** Full weights = 8,192. LoRA weights = 1,536. That's **81% fewer trainable parameters**. In a real LLM, this jumps to training ~0.1% of total parameters.

### 7. Training — What Actually Gets Updated

```python
loss.backward()    # Gradients ONLY for requires_grad=True parameters
optimizer.step()   # Updates ONLY A, B, and classifier — base weights untouched
```

### 8. Saving and Loading the Adapter

**Save only the tiny LoRA weights:**
```python
lora_weights = {name: param for name, param in model.named_parameters() if "lora_" in name}
torch.save(lora_weights, "lora_adapter.pt")  # KBs instead of GBs
```

**Load onto a fresh model:**
```python
new_model = BaseModel(vocab_size=1000, embed_dim=128, hidden_dim=64)
loaded_weights = torch.load("lora_adapter.pt")
new_model.load_state_dict(loaded_weights, strict=False)  # strict=False ignores missing base keys
```

### 9. Inference

```python
new_model.eval()        # Turns off dropout/batchnorm for deterministic output
with torch.no_grad():   # Skips gradient tracking to save memory
    logits = new_model(test)
```

## Appendix Lite

| Concept | Takeaway |
|---|---|
| **Frozen weights** | `requires_grad=False` excludes parameters from backward pass |
| **Low-rank decomposition** | A large weight update approximated by two small matrices |
| **Transpose convention** | Keeps shapes aligned with PyTorch internals and enables clean weight merging |
| **Kaiming init for A** | Prevents vanishing/exploding gradients using correct fan-in |
| **Zero init for B** | LoRA starts with zero effect — training begins from original model behavior |
| **Scaling factor (alpha/r)** | Normalizes LoRA output magnitude so rank changes don't break training |
| **Rank = capacity** | Higher rank = more expressive power, not louder signal |
| **Alpha = strength** | Controls how much the adapter overrides the base model |
| **strict=False loading** | Enables partial weight injection for adapter swapping |


## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

---
