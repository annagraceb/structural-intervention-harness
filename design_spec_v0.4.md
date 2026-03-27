# Structural Intervention Experiment Harness

## Design Spec v0.4 (Final)

---

## Changelog from v0.3

- **Tier 2 symmetry:** Added Tier 2-destructive for large degradation effects (not just constructive). Both directions are scientifically valuable.
- **BH dual-pass semantics:** Clarified that coarse-sweep BH is provisional (budget reallocation only). Final BH across all trials is authoritative.
- **Degenerate trials excluded from BH pool.** Prevents corrupted p-values from incomplete evaluations.
- **W5 `rank` defined:** `rank = min(rows, cols)`. Invalid k values (≤0 or ≥rank) are skipped.
- **Tier 2 threshold realism:** Raised minimum benchmark size to 700 items. At N=500, the 3pp threshold is knife-edge with BH correction over ~515 tests (requires 15+ perfectly one-directional flips to survive as top-ranked result). At N=700, 3pp = 21 items, giving more room.
- **Vacuous c≥5 condition removed** from Tier 2(b). Any intervention surviving BH will trivially exceed 5 flips.
- **Pinned ambiguities:** multi-token log-prob aggregation formula, A2 hook insertion point, Q/K coupling rule for W1, A1 calibration vector freeze, `bh_significant` semantics, item-property taxonomy timing.
- **McNemar null model clarified:** tests directional asymmetry of discordant pairs, not total flip count. Symmetric disruption (equal flips both ways) will not reach Tier 1.

## Changelog from v0.2

- **Benchmark:** Replaced open-ended candidate list with BBH multiple-choice tasks evaluated via log-probabilities. GSM8K demoted to optional secondary validation.
- **Evaluation mode:** Committed to deterministic greedy decoding. Stage 2 reduced to a single verification run + determinism confirmation.
- **Model guidance:** Narrowed candidates to 1.5-2B parameter range. Phi-3-mini removed (VRAM-infeasible). Gemma-2 flagged for GQA complications.
- **Statistical framework:** Simplified to single BH correction across all interventions. Removed non-standard two-stage BH→Holm-Bonferroni procedure. Added precommitted power analysis requirement.
- **W1 permutations:** Added explicit coupled/uncoupled distinction and coupling graph requirement.
- **Tier 3 criteria:** Replaced subjective "no precedent" / "contradicts priors" with pre-registered prediction violations.
- **Infrastructure:** Added weight management protocol, intervention serialization schema requirement, crash recovery, prompt template specification, sanity check intervention, and per-intervention unit tests.
- **Budget:** Worked through the actual arithmetic. Documented why log-prob evaluation is a hard constraint.
- **A3/W5:** Provided concrete proposals (layer skip, spectral editing) while preserving agent latitude to propose alternatives.

---

## Purpose

Discover novel, surprising structural modifications to small language models that measurably change performance on a hard reasoning benchmark. The goal is exploration of underexplored intervention space, not replication of known results. When something interesting is found, it becomes a candidate for deeper investigation (mechanistic interpretability, ablation studies, etc.) in a later phase.

---

## Hardware Constraint

- NVIDIA RTX 3060 12GB VRAM
- 64GB system RAM
- All inference and manipulation must fit in this envelope
- No multi-GPU, no cloud offloading

---

## Three-Stage Design

This project proceeds through three locked stages. Each stage has defined entry criteria, exit criteria, and outputs. A stage's outputs are frozen before the next stage begins. No backtracking to revise earlier decisions based on later results.

---

## Stage 1: Calibration

### Objective

Select a model-benchmark pair and characterize its properties.

### Benchmark Selection

#### Primary benchmark: BBH multiple-choice tasks (log-probability evaluation)

The primary benchmark uses selected BIG-Bench Hard (BBH) multiple-choice tasks evaluated via log-probability comparison. This is a hard constraint driven by the compute budget: generation-based evaluation (e.g., GSM8K chain-of-thought) costs ~10 seconds per item on this hardware, yielding only ~27 total intervention trials in 100 GPU-hours. Log-probability evaluation costs ~0.3 seconds per item, yielding ~4,800 trials — sufficient for meaningful search.

**Task selection criteria (apply in order):**

1. **Reasoning relevance.** The task must test multi-step inference, logical deduction, or structured reasoning — not recall, commonsense, or pattern matching.
2. **Unambiguous evaluation.** Multiple-choice format with a single correct answer. No subjective judgment in scoring.
3. **Sufficient item count.** The combined benchmark must contain at least 700 items to support the statistical framework (see Power Analysis below). At N=500, the 3pp Tier 2 threshold is knife-edge after BH correction over ~515 tests; N=700 provides adequate margin. Individual tasks may have fewer items if bundled.
4. **Difficulty calibration.** The chosen model should score 25-45% on the combined benchmark. Tasks where the model scores near chance (1/N for N options) or near ceiling contribute noise, not signal.

**Recommended tasks to evaluate (in priority order):**

| Task | Items | Options | Reasoning type |
|------|-------|---------|----------------|
| Tracking shuffled objects (3, 5, 7) | 750 | 3-5 | State tracking, multi-step |
| Logical deduction (3, 5, 7) | 750 | 3-5 | Constraint satisfaction |
| Navigate | 250 | 2 | Sequential instruction following |
| Date understanding | 250 | 6 | Temporal reasoning |
| Web of lies | 250 | 2 | Boolean logic chains |

Select 2-4 tasks whose combined item count is at least 700 and whose combined baseline accuracy falls in the 25-45% window. Document the selection rule (e.g., "all items from tracking_shuffled_objects_five_objects and logical_deduction_five_objects") and apply it mechanically.

**Evaluation protocol (log-probability):**

For each benchmark item with answer options {A, B, C, ...}:

1. Construct the full prompt: `[few-shot examples if any] + [question text] + [answer prefix]`
2. Compute the log-probability of each answer option token (or token sequence) conditioned on the prompt
3. Select the option with the highest log-probability as the model's answer
4. Score: 1 if the selected option matches the gold answer, 0 otherwise

Implementation details:
- Use 0-shot evaluation (no few-shot examples) unless the model scores below 15% at 0-shot, in which case use the standard 3-shot BBH prompts from the original paper. Document which protocol is used and freeze it.
- For multi-token answer options (e.g., full sentences), use the arithmetic mean of per-token log-probabilities to avoid length bias: `(1/n) * sum(log p(token_i | prefix))`, where the prefix includes all preceding tokens of the answer option. This is equivalent to the geometric mean of token probabilities.
- The prompt template is a Stage 1 artifact. It is frozen before Stage 2 begins. The exact template (including any whitespace, formatting, instruction text) must be recorded.

#### Secondary benchmark (optional): GSM8K

If budget permits after Stage 3, the top 10 interventions (by tier ranking) may be validated on GSM8K with generation-based evaluation. This is exploratory and not subject to the precommitted statistical framework. Results are reported descriptively, not as hypothesis tests.

If GSM8K is used:
- Answer extraction: parse for `#### <number>` pattern. Fallback: extract last numerical value from output. If neither produces a number, score as incorrect (not excluded).
- Generation parameters: greedy decoding, max 512 new tokens, no repetition penalty.
- Budget: ~4 hours per full pass. Limit to 10 interventions × 1 pass each = ~40 hours. This is outside the 100-hour Stage 3 budget and should only be attempted if Stage 3 completes under budget.

### Model Selection

After the benchmark is chosen, find a model that:

1. Scores in the approximate range of 25-45% on the chosen benchmark (this is a soft target — the priority is that baseline performance leaves room for movement in both directions)
2. Fits in VRAM at fp16 with at least 4GB headroom (for KV cache, CUDA context, and one modified layer copy simultaneously)
3. Uses a standard dense transformer architecture with separate Q, K, V projections (not fused QKV) and standard multi-head attention (not grouped-query attention). This simplifies intervention implementation and avoids architecture-specific coupling bugs.
4. Has publicly available weights on HuggingFace

**Candidates (ranked by hardware fit):**

| Model | Params | fp16 VRAM | Headroom | Architecture notes |
|-------|--------|-----------|----------|--------------------|
| Qwen2-1.5B | 1.54B | ~3.0 GB | ~9.0 GB | Separate Q/K/V. GQA with num_kv_heads < num_heads — verify intervention compatibility |
| StableLM-2-1.6B | 1.64B | ~3.1 GB | ~8.9 GB | Standard MHA. Clean architecture for interventions |
| Gemma-2-2B | 2.61B | ~5.0 GB | ~7.0 GB | GQA + sliding window attention. Complicates W4/A2. Use only if smaller models miss the accuracy window |
| OLMo-1B | 1.18B | ~2.2 GB | ~9.8 GB | Standard MHA. May score too low on BBH tasks |

**Phi-3-mini (3.8B) is excluded.** At ~7.3GB fp16, it leaves insufficient headroom for KV cache growth, weight copies, and CUDA overhead. It also uses fused QKV projections, which breaks the assumption of independent Q/K/V matrices in W4 interventions.

**Architecture verification gate.** Before finalizing the model, the implementing agent must verify:
- Q, K, V projection matrices are separately accessible (not fused into a single `qkv_proj`)
- Attention head dimensions are uniform across all layers
- MLP structure matches the expected pattern (gate_proj, up_proj, down_proj for gated architectures, or fc1/fc2 for standard)
- Layer indexing is consistent and all layers are architecturally identical (same dimensions, same structure)

If the chosen model uses grouped-query attention (num_kv_heads < num_attention_heads), document the head grouping structure and specify how W4 and A2 interventions handle the asymmetry (e.g., does "ablate head 5" mean ablating one query head or one KV head group?).

### Power Analysis (precommitted)

Before finalizing the benchmark, compute the minimum detectable effect size:

1. Assume a benchmark of N items and a baseline accuracy of p (estimated from a single calibration run).
2. McNemar's test detects an asymmetry in discordant pairs. The number of discordant pairs (items that flip between baseline and intervention) is the effective sample size.
3. For a given total number of planned interventions K, the BH correction at FDR=0.05 requires individual p-values of approximately 0.05 × rank/K. For the most significant result, this is ~0.05/K.
4. Compute: given N items, baseline accuracy p, and K planned interventions, what is the minimum number of net flips (correct→incorrect minus incorrect→correct, or vice versa) needed for McNemar's test to reach significance after BH correction? This determines the minimum detectable accuracy delta.
5. If the minimum detectable delta exceeds 3 percentage points, either increase N (add more BBH tasks) or reduce K (cut intervention categories).

Document this computation. It is a Stage 1 artifact.

### Calibration Outputs (frozen before Stage 2)

1. **Benchmark specification:** exact task list, item count, evaluation protocol (prompt template verbatim, 0-shot vs. few-shot, log-probability scoring method, answer extraction for multi-token options)
2. **Model specification:** HuggingFace repo ID, revision hash, precision (fp16), loading configuration (device_map, torch_dtype, attn_implementation), architecture verification results
3. **Baseline accuracy:** single-run accuracy on the full benchmark with deterministic inference
4. **VRAM profile:** model loaded (bytes), peak during benchmark inference (bytes), available headroom (bytes)
5. **Wall-clock timing:** time per benchmark item (mean, min, max over all items), time for one full benchmark pass
6. **Power analysis results:** minimum detectable effect size, planned number of interventions K, effective significance threshold after BH correction
7. **Inference configuration:** all generation/evaluation parameters (temperature=0 for greedy, any torch determinism flags, CUDA configuration)

---

## Stage 2: Determinism Verification and Baseline Characterization

### Objective

Confirm that the evaluation pipeline is fully deterministic and characterize per-item baseline performance for use in Stage 3 statistical tests.

### Why deterministic mode

This spec commits to deterministic greedy decoding (temperature=0, argmax token selection) with deterministic CUDA operations. Rationale:

- **Simplicity.** Each intervention trial requires exactly one benchmark pass, not N passes averaged together. This maximizes the number of distinct interventions explorable within the compute budget.
- **Statistical clarity.** Any change in item-level outcomes between baseline and intervention is a real signal, not sampling noise. McNemar's test applies directly to binary outcomes with no need for pass-rate estimation.
- **Reproducibility.** Any researcher with the same hardware, software, and random seeds can reproduce results exactly.

The cost is that we cannot observe "soft" effects — interventions that shift the probability of a correct answer without crossing the argmax boundary. This is an accepted limitation for the exploration phase.

### Determinism Configuration

Apply all of the following before any inference:

```python
import torch
import os

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

If `torch.use_deterministic_algorithms(True)` raises an error for the chosen model's attention implementation, fall back to `torch.use_deterministic_algorithms(True, warn_only=True)` and document which operations are non-deterministic. Then proceed to the empirical verification below with heightened scrutiny.

### Verification Protocol

1. Run the full benchmark twice with identical configuration.
2. Compare per-item outputs (selected answer option and its log-probability) across both runs.
3. **If all items match:** Determinism is confirmed. Proceed with a single baseline run as the authoritative baseline.
4. **If items differ:** Count the number of differing items.
   - If ≤1% of items differ: likely floating-point non-determinism. Document the differing items, set the determinism flag to "near-deterministic," and run 5 additional verification passes. Use majority-vote across all 7 runs as the authoritative baseline for each item. Flag items with any disagreement as "unstable" and exclude them from McNemar's test in Stage 3 (they are neither stable-correct nor stable-incorrect).
   - If >1% of items differ: the inference pipeline has a nondeterminism bug. Debug before proceeding. Common causes: flash attention, non-deterministic CUDA kernels, batch-order effects.

### Baseline Characterization

From the authoritative baseline run, classify each item:

- **Correct:** model's selected answer matches gold answer
- **Incorrect:** model's selected answer does not match gold answer

Record the full per-item result table: item ID, gold answer, model's selected answer, log-probability of each option.

In deterministic mode, there are no "swing items." Every item is either correct or incorrect, and this classification is stable.

### Stage 2 Outputs (frozen before Stage 3)

1. **Determinism status:** "deterministic" or "near-deterministic" with documentation
2. **Authoritative baseline:** per-item results table
3. **Aggregate baseline accuracy:** exact percentage (not a mean ± std, since there is no variance)
4. **Wall-clock budget for Stage 3:** (time per full benchmark pass) × (planned number of trials). Confirm this fits within 100 GPU-hours. If not, reduce scope per the budget feasibility rules in Stage 3.
5. **Unstable items list** (if near-deterministic mode): items excluded from statistical tests

---

## Stage 3: Intervention Search

### Objective

Apply structural interventions to the model and measure their effect against the locked evaluation protocol.

### Weight Management Protocol

**This protocol is mandatory for all weight-space interventions (Class W). Violations will produce cascading corruption across trials.**

1. **Load original model weights into GPU memory** at the start of Stage 3. These weights are the "golden copy."
2. **Before each trial:** deep-copy the relevant weight tensors to CPU memory, apply the intervention on the CPU copy, then transfer the modified tensors to GPU (overwriting the corresponding parameters in the loaded model).
3. **After each trial:** restore the original weights by copying from a CPU-resident pristine copy back to GPU. Never rely on "reversing" an intervention — always restore from the clean copy.
4. **Verification:** after restoration, compute a checksum (e.g., hash of the first and last layer's Q projection weights) and compare against the pre-trial checksum. If they differ, abort and debug.

Alternative (if the model is small enough): hold two full copies of the model — one pristine on CPU, one working copy on GPU. Before each trial, copy the full model from CPU to GPU. This is simpler but costs ~3GB of system RAM for a 1.5B model, which is easily affordable with 64GB.

### Intervention Serialization Schema

Every intervention must be fully specified by a JSON object that is sufficient to reproduce the exact modification. The schema is defined per intervention type and recorded in the experiment log.

Example schema for W2 (cross-layer transplant):

```json
{
  "trial_id": "W2-0042",
  "category": "W2",
  "description": "Transplant Q projection from layer 3 to layer 18",
  "params": {
    "source_layer": 3,
    "target_layer": 18,
    "component": "q_proj",
    "method": "replace"
  },
  "baseline_checksum": "a3f8c2...",
  "timestamp_utc": "2026-03-28T14:22:01Z"
}
```

The implementing agent must define the schema for all intervention types before running any trials. Schemas are a Stage 3 setup artifact.

### Intervention Taxonomy

Interventions are organized into two classes based on when they act.

#### Class W: Weight-Space Interventions

These modify the model's parameters before inference. The modified model is a static artifact that could be saved and reloaded.

---

**W1: Intra-layer weight permutation.**

Rearrange rows, columns, or blocks of weight matrices within a single layer.

**Critical: coupled vs. uncoupled permutations.**

- **Coupled permutation:** permute the output dimension (columns) of matrix A and apply the inverse permutation to the input dimension (rows) of the downstream matrix B that consumes A's output. This preserves the input-output mapping of the layer while rearranging internal neuron ordering. It tests whether the model is sensitive to neuron identity vs. neuron function.
- **Uncoupled permutation:** permute one matrix without adjusting its connected matrices. This deliberately breaks the correspondence and tests fault tolerance — how much structural damage can the layer absorb before the model degrades?

The spec requires both types to be tested, clearly labeled, and analyzed separately. They test different hypotheses and should not be conflated.

**Coupling graph (must be verified for the chosen model):**

For a standard transformer layer with separate Q, K, V, O projections and gated MLP:

| Matrix | Output dimension connects to |
|--------|------------------------------|
| `q_proj` | Attention score computation (paired with `k_proj` transposed) |
| `k_proj` | Attention score computation (paired with `q_proj`) |
| `v_proj` | Attention output (multiplied by attention weights, then fed to `o_proj`) |
| `o_proj` | Residual stream (output goes to residual add) |
| `gate_proj` | Element-wise product with `up_proj` output |
| `up_proj` | Element-wise product with `gate_proj` output |
| `down_proj` | Residual stream (output goes to residual add) |

**Coupling rules (explicit):**
- A coupled permutation of `q_proj` output columns requires the same permutation of `k_proj` output columns (they interact via QK^T). `v_proj` output columns are coupled with `o_proj` input rows (V feeds into O).
- A coupled permutation of `up_proj` columns requires the same permutation of `gate_proj` columns AND the inverse permutation of `down_proj` rows.

**Permutation strategies:** random, sorted by L2 norm (ascending/descending), sorted by mean activation magnitude on a 50-item calibration set, reverse index order, interleave even/odd indices.

**One permutation applied to one matrix (uncoupled) or one coupled group in one layer per trial.**

---

**W2: Cross-layer weight transplant (same model).**

Copy a weight matrix from layer A to layer B within the same model, replacing B's original matrix.

**Scope per trial:** one component type (e.g., `q_proj`, or `gate_proj`) transplanted from one source layer to one target layer. Do not transplant multiple components simultaneously — that conflates effects.

**Sampling strategy for the coarse sweep:** rather than enumerating all L×(L-1) pairs, sample along structured axes:
- **Adjacent pairs:** (layer i → layer i+1) and (layer i → layer i-1) for all i. Tests local redundancy.
- **Distant pairs:** (layer 0 → layer L/2), (layer L/2 → layer L-1), (layer 0 → layer L-1), and 10 random distant pairs. Tests global structure.
- **Symmetric pairs:** (layer i → layer L-1-i) for all i. Tests whether early and late layers are interchangeable.

This gives approximately 3L + 13 pairs for the coarse sweep. For a 24-layer model, that is ~85 pairs.

**Components to sweep:** prioritize `q_proj` and `gate_proj` in the coarse sweep (one attention component, one MLP component). If signal is found, expand to other components.

---

**W3: Partial reinitialization.**

Reset a targeted subset of parameters to a specified initialization distribution while leaving the rest of the trained model intact.

**Initialization distributions (try each):**
- **Kaiming normal** (fan_in mode): `N(0, sqrt(2/fan_in))`. This is a reasonable default for ReLU/SiLU networks.
- **Xavier uniform:** `U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))`. Standard for linear layers.
- **Zero:** all values set to 0.0. This is destructive by design — tests whether the layer is needed at all.
- **Scaled noise:** `N(0, σ)` where σ matches the empirical standard deviation of the original weight matrix. Tests whether the learned structure matters or only the scale.

**Granularity options (one per trial):**
- Full attention block (Q, K, V, O projections)
- Full MLP block (gate, up, down projections)
- Single projection matrix (e.g., just `q_proj`)
- LayerNorm parameters (weight and bias) for one layer
- All bias terms in one layer (if the model uses biases)

---

**W4: Attention head surgery.**

Modify individual attention head weight matrices.

**Operations (one head, one operation per trial):**

- **Ablation:** zero out head i's rows in `o_proj` (output projection). The head still computes attention, but its output contributes nothing to the residual stream.
- **Duplication:** copy head i's Q, K, V, O weight slices over head j's slices within the same layer. Layer now has two copies of head i and no head j.
- **Averaging:** replace head i's and head j's weights with the element-wise average of both. Both heads become identical.
- **Negation:** multiply head i's rows in `o_proj` by -1. The head now subtracts from the residual stream instead of adding.

**For models with grouped-query attention (GQA):** if the chosen model uses GQA (num_kv_heads < num_attention_heads), document how head indices map to KV groups. "Ablate head i" means zeroing head i's rows in `o_proj` and `q_proj` but leaving the shared KV projections intact (since other heads in the same KV group depend on them). This must be verified in the unit tests.

---

**W5: Rank-selective spectral editing.**

Compute the SVD of a weight matrix: W = UΣV^T. Modify the singular value spectrum and reconstruct:

**Operations (one matrix, one layer per trial):**

- **Top-k truncation:** zero all singular values below the k-th largest. Reconstructed matrix is rank-k. Sweep k ∈ {1, 5, 10, 25, 50, rank/4, rank/2, rank-10}. Here `rank = min(rows, cols)` of the weight matrix. Skip any k value that is ≤ 0 or ≥ rank.
- **Bottom-k removal:** zero the k smallest singular values. Similar to truncation but removes different components. Sweep same k values.
- **Spectral inversion:** swap the largest and smallest singular values (reverse the ordering of Σ while keeping U and V fixed). Tests whether the dominant directions or the minor directions carry task-relevant information.
- **Uniform spectrum:** replace all singular values with their mean. Preserves the subspaces (U, V) but removes magnitude differentiation. Tests whether the model relies on spectral structure or just subspace orientation.

**Justification:** This is a more principled version of W3 (partial reinitialization). Instead of destroying all structure in a weight matrix, spectral editing selectively removes or modifies specific rank-components. This connects to the low-rank adaptation literature (LoRA operates in a low-rank subspace) and tests whether task-relevant information lives in the high-rank or low-rank components of weight matrices — a question with direct implications for model compression and fine-tuning.

---

#### Class A: Activation-Time Interventions

These modify the model's forward pass without changing stored weights. They act on activations during inference via forward hooks.

---

**A1: Residual stream injection.**

Add a fixed vector to the residual stream at a specific layer boundary (after the layer's output is added to the residual stream, before the next layer processes it).

**Vector sources (one per trial):**
- **Random unit vector** scaled to magnitude α: `v = α * randn(hidden_dim) / ||randn(hidden_dim)||`. Fix the random seed per trial for reproducibility.
- **Mean activation vector:** compute the mean residual stream activation at this layer over a 50-item calibration set. Inject the mean (or negative mean) at magnitude α.
- **Difference vector:** compute mean residual stream activations for correct vs. incorrect items from the baseline. Inject the difference vector (correct minus incorrect) at magnitude α. This is a steering vector approach.

**Magnitude calibration:** before the coarse sweep, compute the mean L2 norm of the residual stream at each layer over 20 benchmark items. Use these norms to set α values: try α ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 5.0} × (mean norm at that layer). This ensures injection magnitudes are scaled appropriately to the model's activation space.

**A1 calibration vectors (mean activation, difference vector) are computed once from the unmodified baseline model during Stage 3 setup and frozen for all A1 trials.** They are Stage 3 setup artifacts, not recomputed per trial.

**One injection point (one layer, one vector, one magnitude) per trial.**

---

**A2: Attention head output scaling.**

Multiply a specific attention head's output by a scalar before it enters the residual stream. Specifically: scale the output of head i's slice of `o_proj`, before the residual addition and before any post-attention LayerNorm. Implemented via a forward hook on the attention output projection.

**Scalars to try:** 0.0 (ablation), -1.0 (negation), 0.5 (attenuation), 2.0 (amplification), 5.0 (strong amplification).

**One head, one scalar per trial.**

Note: A2 with scalar=0.0 is functionally equivalent to W4 ablation (zeroing o_proj rows). If both are tested, they should produce identical results — this serves as a cross-validation check between Class W and Class A implementations.

---

**A3: Layer skip (bypass).**

During the forward pass, skip layer N entirely: the residual stream passes through without being modified by layer N's attention or MLP computation. Equivalently, replace layer N's output with a zero vector (so the residual stream addition is a no-op).

**Variants:**
- **Full layer skip:** bypass both attention and MLP sublayers of layer N.
- **Attention-only skip:** bypass only the attention sublayer; MLP still executes.
- **MLP-only skip:** bypass only the MLP sublayer; attention still executes.

**One layer, one variant per trial. Sweep all layers.**

**Justification:** Layer skip is mechanistically clean — it tests whether a layer is *necessary* for task performance. Unlike ablation (which targets individual heads), layer skip tests the entire computational unit. Results directly inform model compression (which layers can be pruned?) and the hypothesis that transformers have redundant layers. While "layer dropping" has been studied during training, systematic evaluation of layer skip at inference time on reasoning benchmarks is underexplored for small models.

---

### Pre-Registered Predictions

Before Stage 3 begins, the implementing agent must write and freeze a list of 10-15 explicit, falsifiable predictions about intervention outcomes. These predictions serve two purposes: (1) they make Tier 3 classification objective, and (2) they document the experimenter's priors, which are scientifically valuable regardless of whether the predictions are confirmed.

**Example predictions (to be replaced with actual predictions based on the chosen model and benchmark):**

1. "Ablating (A2 scalar=0) any single attention head in the last 3 layers will decrease accuracy by at most 2 percentage points."
2. "Cross-layer transplanting Q projections between adjacent layers (|source - target| ≤ 2) will decrease accuracy less than transplanting between distant layers (|source - target| ≥ 10)."
3. "Reinitializing (W3) any single MLP block in layers 2 through L-3 with Kaiming normal will decrease accuracy by at least 5 percentage points."
4. "Spectral truncation (W5) to rank-50% of Q projection matrices will preserve at least 90% of baseline accuracy."
5. "Full layer skip (A3) of any single layer in the middle third of the model will decrease accuracy by at least 3 percentage points."

Predictions must be specific enough that they can be evaluated as true/false after Stage 3. A Tier 3 result is one that violates a pre-registered prediction AND meets all Tier 2 criteria.

### Statistical Decision Rule (precommitted)

This rule is fixed before any intervention trial runs. It is not adjusted based on results.

#### Three-Tier Result Classification

Every intervention result is classified into exactly one tier.

**Tier 1 — Statistically detectable.**

The intervention produced an effect distinguishable from the null hypothesis (no change) after multiple-comparisons correction. Specifically, McNemar's test detects *directional asymmetry* in discordant pairs: it tests whether the number of items flipped correct→incorrect differs from the number flipped incorrect→correct. An intervention that causes massive symmetric disruption (many flips in both directions, roughly balanced) will NOT reach Tier 1 — this is desirable, as symmetric disruption indicates generic damage rather than a targeted effect.

**Tier 2 — Practically meaningful.**

The intervention is Tier 1 AND satisfies BOTH of the following:
- (a) The absolute accuracy delta is at least 3 percentage points (in either direction — improvement or degradation).
- (b) The effect is predominantly one-directional: at least 70% of discordant pairs (flipped items) go in the same direction.

**Tier 2 sub-classification:**
- **Tier 2-constructive:** accuracy improved (more items flipped incorrect→correct than correct→incorrect).
- **Tier 2-destructive:** accuracy degraded (more items flipped correct→incorrect than incorrect→correct).

Both sub-types are scientifically valuable. Constructive results suggest the intervention enhanced a capability; destructive results reveal which components are critical. The anomaly report covers both.

These thresholds are locked before Stage 3 begins. If the power analysis from Stage 1 shows these thresholds are unreachable given the benchmark size, they must be adjusted during Stage 1 (not Stage 3) with written justification.

**Tier 3 — Scientifically interesting.**

The intervention is Tier 2 AND it violates at least one pre-registered prediction from the list above. The implementing agent must cite the specific prediction violated and explain how the result contradicts it.

Additionally, any Tier 2 result that exhibits sharply localized effects qualifies for Tier 3 consideration: the intervention depends on a specific layer, head, or component, and applying the same operation to adjacent layers/heads (±2 indices) produces no Tier 1 effect. This must be verified by running the adjacent variants, not assumed.

Any Tier 2 result where the set of flipped items is semantically coherent also qualifies. To test this: precommit to a taxonomy of item properties (e.g., number of reasoning steps, number of entities, deduction depth) and test whether flipped items are non-randomly distributed across these properties using Fisher's exact test at α=0.05. This converts a subjective judgment into a statistical one. **The item-property taxonomy is a Stage 3 setup artifact, defined before any trials run, based on properties observable from the benchmark items alone (not from model outputs).**

#### Significance Testing

1. **McNemar's exact test** on the paired 2×2 table of item outcomes (baseline correct/incorrect × intervention correct/incorrect) for each trial. In deterministic mode, this is applied directly to the binary outcomes.

2. **Benjamini-Hochberg correction** at FDR = 0.05 across ALL interventions in ALL categories. This is a single global correction, not a hierarchical procedure. Rationale: a single BH pass controls FDR at the stated level regardless of category structure, is easy to implement and explain, and avoids the ambiguity of the two-stage approach.

3. Interventions that survive BH correction are Tier 1. Tier 2 and Tier 3 are additional filters applied on top of Tier 1, not separate hypothesis tests.

#### No Post-Hoc Adjustment

If nothing survives correction at any tier, that is a valid result. Do not relax thresholds, change the FDR level, switch to a different test, or subset the benchmark to rescue marginal findings.

### Sanity Check Intervention

Before running any novel interventions, run exactly two sanity checks:

1. **Positive control (should degrade):** zero out all attention heads in the final layer (A2 scalar=0 for every head in layer L-1). Expected result: substantial accuracy degradation (≥10pp). If this does not produce a large effect, the evaluation pipeline may have a bug (e.g., the forward hooks are not being applied).

2. **Negative control (should be no-op):** apply a coupled identity permutation to one layer (permute and inverse-permute with the same permutation = no actual change). Expected result: zero accuracy change and bitwise-identical outputs. If this changes any item outcome, the weight management or permutation code has a bug.

If either sanity check fails, debug before proceeding. These two trials are outside the 100-hour budget and do not count toward intervention totals.

### Search Strategy

**Budget allocation:**

- Total budget: 100 GPU-hours (wall-clock, including overhead for weight manipulation, logging, and checkpointing).
- Deduct 10% for overhead: 90 GPU-hours available for actual benchmark passes.
- Compute the cost of one full benchmark pass from Stage 2 timing. Let T = time per pass in hours.
- Total available trials: floor(90 / T).
- Allocate trials across categories proportional to the category's search space size, with a minimum of 15 trials per category.

**Proposed allocation (to be adjusted based on actual T):**

| Category | Estimated trials (coarse sweep) | Search space |
|----------|--------------------------------|--------------|
| W1 (permutation) | ~80 (40 coupled + 40 uncoupled) | Layers × matrices × strategies |
| W2 (transplant) | ~85 (3L + 13 pairs × 2 components) | Layer pairs × components |
| W3 (reinitialization) | ~60 (layers × 4 distributions × 2 granularities) | Layers × distributions × granularities |
| W4 (head surgery) | ~60 (sample of heads × 4 operations) | Heads × operations |
| W5 (spectral editing) | ~60 (layers × k-values × 2 components) | Layers × truncation levels × components |
| A1 (residual injection) | ~50 (layers × magnitudes × 2 vector types) | Layers × magnitudes × vector sources |
| A2 (head scaling) | ~50 (sample of heads × 5 scalars) | Heads × scalars |
| A3 (layer skip) | ~70 (all layers × 3 variants) | Layers × variants |

Total: ~515 trials for the coarse sweep. At T ≈ 3.5 minutes per pass (BBH 700 items, log-prob at ~0.3s/item), this is ~30 hours. The remaining ~60 hours are reserved for:
- Narrowing searches in categories that show Tier 1 signal
- Running adjacent-layer variants to test Tier 3 localization
- Running item-property tests for Tier 3 semantic coherence

**Adaptive reallocation rule (precommitted):** after the coarse sweep for all categories is complete, run a *provisional* BH correction to identify candidate Tier 1 results. This provisional pass is used solely for budget reallocation decisions — it is NOT the authoritative tier classification. Rank categories by the number of provisionally-significant results. Allocate the remaining budget proportionally to categories with signal. Categories with zero provisionally-significant results receive no additional budget. If no category has any signal, distribute remaining budget equally across all categories for a second coarse sweep with different random samples. Document the reallocation decision.

**The final BH correction across ALL trials (coarse + narrowing) is the authoritative classification.** Tier classifications from the provisional coarse-sweep pass may change. Results that lose significance in the final pass are demoted regardless of whether they triggered follow-up work. The `bh_significant` column in the database reflects only the final BH correction.

**Budget feasibility is a hard gate.** If the coarse sweep alone (515 trials × T) exceeds 90 GPU-hours, reduce scope by:
1. Cut the category with the smallest expected search space (likely A1 or A2)
2. Reduce per-category coarse sweep to 10 trials minimum
3. Document the tradeoff

### Degenerate Output Detection

Since the primary benchmark uses log-probability evaluation (not generation), degeneracy manifests differently than in generation-based evaluation:

- **Log-prob degeneracy:** the model assigns identical (or near-identical, within 1e-6) log-probabilities to all answer options for ≥50% of items. This indicates the intervention has destroyed the model's ability to discriminate between options.
- **Numerical degeneracy:** log-probabilities are NaN or ±inf for any item. This indicates numerical instability from the intervention (e.g., zeroing LayerNorm parameters).

**Early abort:** evaluate the first 20% of benchmark items. If degeneracy is detected, abort the trial. Log the intervention as "degenerate" with the failure mode. Do not run the remaining 80%.

**Degenerate trials are excluded from the BH correction pool.** They are logged for documentation (with `is_degenerate = TRUE` and `accuracy = NULL`) but do not contribute to significance testing. Including them would introduce incomparable p-values from incomplete evaluations.

**For optional GSM8K validation:** use the generation-based degeneracy checks from v0.2 (empty output, repetition loops, mode collapse).

### Per-Intervention Unit Tests

Before Stage 3 begins, implement and pass unit tests for each intervention type. Tests run on a minimal model (2 layers, 64-dim hidden, 4 heads) and verify:

1. **Correctness:** the intervention modifies exactly the intended parameters/activations and nothing else. Compare the modified model's state dict against the original, element by element.
2. **Isolation:** after the intervention is applied and the trial is complete, restoring from the golden copy produces a model that is bitwise-identical to the original. Verify via full state dict comparison.
3. **Determinism:** applying the same intervention twice (same parameters, same seed) produces bitwise-identical modified weights.
4. **Coupling (W1 only):** for coupled permutations, verify that the layer's input-output mapping is preserved by running a forward pass with random input before and after the coupled permutation and confirming identical output (within fp16 tolerance).
5. **Cross-validation (A2 vs. W4):** verify that A2 with scalar=0 and W4 ablation produce identical model outputs on 10 random inputs.

---

## Output Artifacts

### Experiment Log (SQLite database)

Use SQLite, not JSON files. Each trial result is written as a committed transaction immediately upon completion. This provides crash recovery (incomplete trials are not committed) and queryable results.

**Schema:**

```sql
CREATE TABLE trials (
    trial_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,           -- W1, W2, ..., A1, A2, A3
    intervention_spec TEXT NOT NULL,  -- Full JSON intervention specification
    is_degenerate BOOLEAN NOT NULL,
    accuracy REAL,                    -- NULL if degenerate
    accuracy_delta REAL,              -- NULL if degenerate
    items_flipped_to_correct INTEGER,
    items_flipped_to_incorrect INTEGER,
    mcnemar_p_value REAL,
    bh_significant BOOLEAN,
    tier INTEGER,                     -- 1, 2, 3, or NULL if not significant
    tier_justification TEXT,          -- For Tier 3: which prediction violated
    wall_clock_seconds REAL NOT NULL,
    vram_peak_bytes INTEGER,
    timestamp_utc TEXT NOT NULL,
    random_seed INTEGER
);

CREATE TABLE item_results (
    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
    item_id TEXT NOT NULL,
    baseline_correct BOOLEAN NOT NULL,
    intervention_correct BOOLEAN,     -- NULL if degenerate/aborted
    baseline_logprobs TEXT,           -- JSON: {"A": -1.2, "B": -3.4, ...}
    intervention_logprobs TEXT,       -- JSON: same format
    PRIMARY KEY (trial_id, item_id)
);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Metadata includes: model_id, model_revision, benchmark_tasks,
-- baseline_accuracy, power_analysis, inference_config, library_versions,
-- cuda_version, gpu_name, determinism_status
```

### Leaderboard (human-readable)

Generated from the SQLite database. Interventions organized by tier (Tier 3 first, then Tier 2, then notable Tier 1). Each entry includes:

- Tier classification with justification
- Intervention description (human-readable summary + trial_id for lookup)
- Accuracy delta with exact item counts (flipped correct→incorrect, incorrect→correct)
- McNemar's test p-value (BH-adjusted)
- Category

### Anomaly Report

For each Tier 3 result:

1. **What was done:** full intervention specification in plain language
2. **What changed:** aggregate accuracy delta, list of flipped items with their content
3. **What's surprising:** which pre-registered prediction was violated and why the result was unexpected
4. **Item analysis:** if applicable, Fisher's exact test results for semantic coherence of flipped items
5. **Localization evidence:** results of adjacent-layer/head variants showing the effect is localized
6. **Suggested follow-ups:** 2-3 specific investigations for a future interpretability phase

### Null Results Log

For each intervention category: number of trials run, range of interventions explored (e.g., "W2: transplanted q_proj and gate_proj between 85 layer pairs spanning all 24 layers"), maximum observed accuracy delta, and the upper bound on effect size that can be ruled out given the sample size (i.e., "no W2 intervention produced more than X flipped items, so we can rule out effects larger than Y percentage points at 95% confidence").

---

## Design Principles

- **Stages are locked.** No revisiting benchmark or model selection based on intervention results. This prevents the search from biasing its own evaluation.
- **Reproducibility over speed.** Every experiment must be exactly reproducible from the log. Pin random seeds, record library versions, store exact intervention parameters. Use SQLite with atomic writes for crash safety.
- **Deterministic evaluation.** One pass per trial. Any change is a real change. No averaging over stochastic runs.
- **Cheap trials, expensive confirmation.** Most trials run once. Only BH-surviving results get the full confirmation write-up and Tier 3 analysis.
- **Interventions are stateless.** Each trial starts from the original model. No cumulative modifications across trials. Verified via checksum.
- **Fail fast.** Degenerate output detection aborts bad trials at 20% completion. Sanity checks catch pipeline bugs before the main experiment.
- **Novelty over magnitude.** A small but surprising result (Tier 3) is more valuable than a large but expected one (Tier 2). The anomaly report prioritizes Tier 3 results.
- **Null results are real results.** "Category X produces no significant effects under these conditions" is first-class output, documented with effect-size bounds.
- **Test your tools.** Per-intervention unit tests and sanity check interventions validate the pipeline before it generates data.

---

## Out of Scope (for now)

- Fine-tuning / gradient-based optimization of any kind
- Multi-model merging (mergekit-style cross-model weight combination)
- Prompt engineering / in-context learning variations
- Distillation or training
- Composing multiple interventions (phase 2)
- Mechanistic interpretability of found results (phase 2)
- Searching for "why" (phase 2 — this phase is strictly "what")
- Stochastic evaluation mode (deliberately excluded in favor of deterministic for budget reasons; revisit if determinism cannot be achieved)

---

## Implementation Checklist

This checklist must be completed in order. Each item is a gate — do not proceed to the next until the current item is verified.

### Stage 1 Setup
- [ ] Select BBH tasks and verify item count ≥ 500
- [ ] Download and verify model weights (check HuggingFace revision hash)
- [ ] Verify model architecture: separate Q/K/V projections, uniform head dimensions, consistent layer structure
- [ ] Implement log-probability evaluation pipeline
- [ ] Run baseline accuracy; confirm 25-45% window
- [ ] Profile VRAM: model loaded, peak during inference, headroom
- [ ] Time one full benchmark pass
- [ ] Run power analysis; confirm 3pp threshold is detectable given benchmark size and planned trial count
- [ ] Freeze and record all Stage 1 artifacts

### Stage 2 Setup
- [ ] Configure deterministic inference (all torch flags, CUBLAS config)
- [ ] Run two identical benchmark passes; confirm bitwise-identical outputs
- [ ] If not identical: debug or fall back to near-deterministic protocol
- [ ] Record authoritative baseline (per-item results)
- [ ] Compute wall-clock budget for Stage 3; confirm feasibility
- [ ] Freeze and record all Stage 2 artifacts

### Stage 3 Setup (before any trials)
- [ ] Implement weight management protocol (golden copy, per-trial restore, checksum verification)
- [ ] Define intervention serialization schemas for all categories
- [ ] Implement all intervention types (W1-W5, A1-A3)
- [ ] Write and pass unit tests for all intervention types
- [ ] Run sanity check interventions (positive and negative control)
- [ ] Write pre-registered predictions (10-15 items)
- [ ] Create SQLite database with schema
- [ ] Implement degenerate output detection
- [ ] Implement early-abort logic
- [ ] Implement BH correction and tier classification
- [ ] Implement crash recovery (resume from last committed trial)

### Stage 3 Execution
- [ ] Run coarse sweep for all categories
- [ ] Apply BH correction to coarse sweep results
- [ ] Determine adaptive reallocation (which categories get additional budget)
- [ ] Run narrowing sweeps for signal-bearing categories
- [ ] For Tier 2 results: run adjacent-layer/head variants to test localization
- [ ] For Tier 2 results with ≥10 flipped items: run Fisher's exact test on item properties
- [ ] Apply final BH correction to all trials
- [ ] Classify all significant results into tiers
- [ ] Generate leaderboard, anomaly reports, null results log

---

## Open Questions for Implementing Agent

1. Which 2-4 BBH tasks produce a combined baseline accuracy of 25-45% for the chosen model? (Requires empirical calibration runs.)
2. Does the chosen model achieve deterministic inference with the specified torch configuration? If not, what is the source of nondeterminism and can it be resolved?
3. What are your 10-15 pre-registered predictions? (These should be informed by the baseline characterization and your understanding of transformer internals.)
4. For W1 coupled permutations: what is the exact coupling graph for the chosen model's architecture? (Verify by inspecting the model's module structure.)
5. For A1 residual injection: what are the empirical residual stream norms at each layer? (Needed to calibrate injection magnitudes.)
6. Does the total coarse sweep (~515 trials) fit within the 90-hour adjusted budget? If not, which categories are cut?
