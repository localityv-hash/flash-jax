# flash-jax: design specification (v1)

**Status:** final spec for the v1 implementation. **This document is the single source of truth.** Several
engineers implement it in parallel without talking to each other, so:

- Every shared file is pasted **verbatim** in Appendix A and is owned by work package WP1 ("core").
  Nobody else edits those files. If a shared contract seems wrong, the fix goes through WP1, never a local
  workaround.
- Everything that crosses a module boundary has an exact signature and contract in §4.
- Appendix B holds **normative reference implementations** of the trickiest algorithms. They were run and
  validated while writing this spec (§0.3). Implementers may restructure them but must keep the behavior.
- §11 records the review of this spec: every issue raised, how it was verified, and what changed.

Environment: Python 3.11, `jax==0.10.2` (CPU in development, TPU v5e/v6e as the target), numpy, safetensors.
Hugging Face is unreachable, so every test uses tiny random models and synthetic checkpoints.

---

## 0. Proposal review and verdict

### 0.1 Scores

Each axis is scored from 1 to 10, higher is better. For "correctness risk", 10 means the lowest risk.

| Proposal | Correctness risk | TPU performance | Simplicity | Testability | Total |
|---|---|---|---|---|---|
| **P1: simplicity-first** | 8 | 6 | 9 | 8 | **31** |
| P2: TPU-performance-first | 5 | 8 | 4 | 7 | 24 |
| P3: correctness/testability-first | 8 | 5 | 6 | 10 | 29 |

**P1**
- Correctness risk 8: one scheduling concept; a kernel with no mutable cross-step state and a host-built work
  plan; every kernel line runs in interpret mode. It loses points for picking a preemption victim that inverts
  priority (never itself), for having no invariant checkers, and for an oracle that is another JAX model.
- TPU performance 6: head-major pages give contiguous, TPU-proven K/V reads, and there is per-pair causal
  kv-tile skipping. Against that: dummy DMAs of page 0 for invalid pages, an in-kernel q-head fold that needs
  an f32 upcast when G is small, a KV write done by XLA scatter with unknown TPU cost, and a synchronous host
  loop.
- Simplicity 9 and testability 8.

**P2**
- TPU performance 8, the best: KV write fused into the kernel, valid-page-only fetches, a kv-head-major q
  layout, one packed host-to-device (H2D) transfer, async lookahead, and tuned tables.
- Correctness risk 5 and simplicity 4. The one-step lookahead state machine, uint32 bf16 packing, lane-packed
  D=64 caches and dynamic-length DMAs are all hard to get right. Several were only checked at the MLIR-lowering
  level. The token-major interleaved layout needs strided and bit-trick reads.

**P3**
- Testability 10, the best: invariants written as code, a float64 NumPy oracle, NaN canaries, a
  teacher-forced tie-aware comparator, and a FakeRunner.
- TPU performance 5:
  - the fused write issues one DMA per token, which is thousands of tiny DMAs per layer for a prefill;
  - there is no causal kv-tile skipping;
  - the 4-D `qkv_w [H, Hkv, G+2, Dp]` weight pads the TPU (8/16, 128) tiles several-fold;
  - head-dim padding is baked into the weights.
- Simplicity 6: two jits, and a bundled-kernel fallback backend that needs a second kernel.

### 0.2 Verdict

**Base: P1.** From the other proposals we graft:

- **From P2:**
  - the kv-head-major q layout `[Hkv, T*G, D]`, prepared by the wrapper;
  - `pl.when`-guarded **valid-page-only** DMAs, with the same predicate for start and wait;
  - the CPU→TPU Mosaic lowering smoke test;
  - per-request RNG keys `fold_in(seed, gen_idx)`, with `lax.top_k` candidates for top-k/top-p;
  - the `max_prefill_chunk` knob.
- **From P3:**
  - the independent float64 NumPy oracle over raw HF weights, and the teacher-forced tie-aware comparator;
  - sharp random init with a diversity guard;
  - invariants as code (I1–I7, S1–S4) behind `debug_checks`;
  - a FakeRunner, extended here with a **simulated paged KV store**;
  - `EngineDeadlock`;
  - touch-before-pop in allocation;
  - the rule "never use negative indices with `mode="drop"`";
  - strict loader key checks and `o_bias`;
  - `abort()`;
  - the env-var override of `auto`.
- **Rejected for v1** (see the decision log, §10): async lookahead scheduling, u32/lane-packed caches, the
  in-kernel per-token KV write, two-jit sampling, the AOT executable dict, and the bundled-kernel backend.

### 0.3 Facts verified while writing this spec (jax 0.10.2, CPU)

The prototypes live in the scratchpad, not in the repo.

1. **The chosen attention kernel matches the references in interpret mode.** The kernel is Appendix B.3 (as
   revised in the review, §11):
   head-major cache, page-major VMEM double buffer, host work plan, valid-page-only guarded DMAs, and
   kv-head-major q. It ran under `interpret=pltpu.InterpretParams()`, whose defaults are
   `dma_execution_mode="on_wait"`, `uninitialized_memory="nan"` and `out_of_bounds_reads="raise"`. Unused
   cache pages were filled with **NaN**. Across 8 random ragged batches (G ∈ {1,2,4,8}, ps ∈ {4,8,16},
   bq ∈ {8,16,32}, pages per kv tile ∈ {1,2,4}, D ∈ {16,64,128}, f32 and bf16):
   - kernel vs `ops.reference.paged_attention`: at most 1.4e-6 in f32 and 1.6e-2 in bf16;
   - reference vs JAX's bundled `ref_ragged_paged_attention`: at most 9.5e-7;
   - padding rows exactly 0 and all outputs finite;
   - the same 8 batches also pass with `dma_execution_mode="eager"`, and with `detect_races=True`
     (`races_found` stays False);
   - it also passes at the exact TPU default configuration (bf16, bs=16, ppb=8 so bkv=128, bq=32, Dc=128, G=4,
     Hkv=8), with bkv=Dc=256 (which exercises the `jnp.tile` widening), and with the small decode tile bq=4.
2. **The kernel and the fused norm/SwiGLU kernels lower to Mosaic** for v5e and v6e from a CPU host (the
   private `tpu_info.registry["cpu"]` hook, §4.2), at Llama-3-8B shapes in bf16, with G=7 (Hkv=2), and in f32.
   - **What CPU lowering checks:** unsupported primitives, the dot precision (only DEFAULT and HIGHEST lower),
     and BlockSpec tiling: `_check_block_mappings` raises `ValueError` unless the last two block dims are
     divisible by (8, 128) or equal to the array dims, and it checks the rank-1 block rules (verified).
   - **What it does NOT check** (libtpu does, at compile time): `pl.ANY` operands, scratch shapes, DMA slab
     shapes, in-kernel value reshapes (e.g. `vector.shape_cast 1x8x1x1x16x128xbf16 -> 128x128`), bf16 16-row
     packing, vector-layout legality, and the VMEM/SMEM budgets. These are asserted by `ops.tpu_rules` (WP1,
     §5.4.6) in the runner and in the wrappers.
   - **Mosaic MLIR audit** of B.3 at Llama-3-8B bf16 shapes (captured with `pallas_call(debug=True)`): no i1
     lane-broadcasts, no bf16 `arith.select`, no width-1 lane slices, and no `contract_precision<fp32>` matmul,
     even under a global `jax_default_matmul_precision="highest"`. The only bf16 value reshape left is the
     `[ppb, bs, Dc] → [bkv, Dc]` K/V reshape (R1 fallback).
3. **The full reference path matches the float64 NumPy oracle token-for-token.** The path is: loader
   conversion, then the model scanned over layers with a donated head-major cache, then the XLA-scatter KV
   write and reference attention, then the on-device sampler, driven by the Appendix B scheduler and block
   manager.
   - The exact §8.2 matrix: 6 configs × prefix caching off/on, 8 greedy requests each including the second
     wave. Every request was **exact** with 0 near-ties; rows 2 and 5 preempted (1–6 times); every caching-on
     run had prefix hits; outputs were identical with caching off and on.
   - Seeded top-k/top-p sampling gave identical outputs with and without preemption.
   - Re-run after the review with the revised Appendix A/B code (`prepare_step_input` in `batch.py`,
     `KernelConfig.attn_tile_q`, explicit device placement, the architecture-only static key): same results;
     10 buckets compiled = 10 executables = 10 traces; three EOS-only `ModelConfig` variants of an
     already-compiled config added **0** executables.
4. **The same holds with all three Pallas kernels in interpret mode**, at about 20 s per §8.2 slow config
   (8/8 exact, 4 preemptions in config (a)). It also holds with the cache head dim padded (D=32 → 64/128), and
   in bf16 (8/8 tokens agree with the f64 oracle). Re-run with the revised B.3/B.6: (a) 8/8 exact, 4
   preemptions, 21 s; (b) 8/8 exact, 17 s.
5. **Donation works:** `memory_analysis()` shows alias = the whole cache and the donated buffer
   `is_deleted()`. `jitted._cache_size()` exists and counts executables. **Compiles are not traces:** a jit
   called with a committed param and an uncommitted `jnp.zeros` cache compiled twice (the donated output
   cache comes back committed) while the Python trace counter stayed at 1; with every input on one explicit
   device it compiled once. `jit.lower(...).compile()` and a later call with the same shapes share one
   executable (0 extra backend compiles, counted with `jax.monitoring`).
6. **The scheduler and block manager survived 2000 randomized workloads** driven by the FakeRunner with a
   simulated KV store: staggered arrivals, duplicate prompts, shared prefixes, tiny pools, budgets 1–40,
   chunk caps, and 1385 preemptions. Every output was exact and I1–I7 held after every step. Mutation checks:
   the harness **catches** sealing uncomputed blocks, a hash chain without its parent, a missing
   eviction-unmap, and a missing last-token cap. It does **not** catch LRU-policy changes, which is why a
   targeted unit test is required for those. Preemption is **not strict priority**: in 600 workloads, 19 steps
   preempted a sequence while a newer one was scheduled (§5.2).
7. **The sampler is correct:** total variation (TV) distance to the exact distribution is ≤ 0.013 at 20k
   samples for temperature, top-k and top-p, with V=50, `max_candidates=16` and logits `3·N(0,1)` (the 0.8
   nucleus then has ≤ 12 tokens, so the K-candidate truncation never bites); greedy and top_k=1 give argmax;
   output is invariant to batch permutation. With unit-scale logits the 0.8 nucleus has a median of 23 tokens,
   and the documented truncation to K=16 gives TV 0.21 against the exact nucleus. A deliberately wide nucleus
   (logits `0.3·N(0,1)`, top_p 0.9, 43 tokens) matches the K-truncated distribution at TV 0.0095 (§5.6).
8. **Interpret-mode cost:** 0.12–0.23 s per attention call (steady state), about 1–2 s for the first call,
   and about 20 ms per norm or SwiGLU call. The reference model step costs about 1 ms, and compiles in about
   1 s per bucket.

---

## 1. Overview

flash-jax is a JAX/TPU-native mini vLLM with these pieces:

- **Paged KV cache.** One head-major array for all layers.
- **One Pallas ragged paged attention kernel.** It serves a flat, mixed batch of prefill chunks and decode
  tokens in one call per layer.
- **Hash-based prefix caching.** Chained sha256 over full blocks, with ref counts and one LRU of free blocks.
- **Chunked prefill.** vLLM-v1 unified "tokens to compute" scheduling with a per-step token budget and
  decodes first.
- **Kernel fusion.** Fused QKV and gate_up weights, a Pallas residual-add+RMSNorm, a Pallas matmul with a
  SiLU·mul epilogue, the KV write in the same layer body as attention, and the whole step in one jit with the
  cache donated.

The engine is pure JAX: params are dict pytrees, and model code is pure functions.

### 1.1 Request lifecycle

```
 LLM.generate(prompts, SamplingParams)
   └─ add_request ─► Sequence(WAITING) ─► Scheduler.waiting (FIFO by seq_id)
 LLM.step():  (repeated until all done; synchronous)
   1. Scheduler.schedule()                                  [host, pure Python]
        pass 1: RUNNING decodes (num_remaining==1), by seq_id           ┐ may preempt latest-arrived
        pass 2: RUNNING partial prefills, n=min(remaining,budget,cap)   ┘ unscheduled seqs (recompute)
        pass 3: admit WAITING FIFO (skipped if anything was preempted):
                prefix-cache lookup ─► allocate hits + fresh blocks ─► RUNNING
        ─► batch = [ScheduledSeq(seq, start_pos, n, samples)]  (decodes, prefills, admissions)
   2. ModelRunner.run(batch)
        batch.prepare_step_input(): numpy StepInput (T_pad bucket, S_pad, slots, cu_q_lens, page table,
                                    work plan with the bucket's q-tile, sampling)
        jax.device_put(StepInput, runner.device)   (one pytree transfer)
        step_fn(params, kv_cache(donated), inp) ─ one XLA program:
            embed ─► lax.scan over layers:
                 add+RMSNorm ─► fused QKV matmul ─► (q/k-norm) ─► RoPE
                 ─► write_kv (scatter into donated cache) ─► ragged paged attention ─► o-proj
                 ─► add+RMSNorm ─► gate_up matmul + SiLU*mul ─► down-proj
            ─► gather last token of each seq ─► final add+RMSNorm ─► lm_head (f32) ─► sampler
        ◄─ int32[S_pad] tokens (the only device→host transfer)
   3. Scheduler.update(batch, tokens)
        num_computed += n; seal newly full computed blocks into the prefix cache;
        if samples: append token, check stop (EOS / stop ids / max_tokens / max_model_len)
        finished ─► free blocks (sealed ones stay cached) ─► RequestOutput
```

---

## 2. Package layout

Each path lists its owner (WP) and its responsibility. "verbatim" means the full content is in Appendix A.

```
pyproject.toml                          WP1 verbatim  deps: jax>=0.10.2,<0.11, numpy, safetensors; pytest markers,
                                                      pythonpath=["."]
README.md                               WP6           usage, testing, benchmark instructions (edit existing)
docs/DESIGN.md                          —             this document
flash_jax/__init__.py                   WP1 verbatim  exports; lazy LLM/RequestOutput
flash_jax/config.py                     WP1 verbatim  ModelConfig(.from_hf), EngineConfig, KernelConfig (+attn_tile_q),
                                                      buckets, backend resolution, kv_head_dim
flash_jax/sampling_params.py            WP1 verbatim  SamplingParams
flash_jax/batch.py                      WP1 verbatim  ScheduledSeq, AttentionMetadata, StepInput,
                                                      build_work_plan, build_attention_metadata, prepare_step_input
flash_jax/engine/__init__.py            WP1 verbatim  (docstring only)
flash_jax/engine/sequence.py            WP1 verbatim  Sequence, SeqStatus
flash_jax/engine/block_manager.py       WP4           BlockManager, hash_block, ROOT_HASH
flash_jax/engine/scheduler.py           WP4           Scheduler, EngineDeadlock
flash_jax/engine/model_runner.py        WP5           step_fn (jit), ModelRunner (KV alloc, prepare, run, warmup)
flash_jax/engine/llm.py                 WP5           LLM, RequestOutput
flash_jax/model/__init__.py             WP1 verbatim  (docstring only)
flash_jax/model/rope.py                 WP3           rope_inv_freq (llama3 scaling), rope_cos_sin, apply_rope
flash_jax/model/llama.py                WP3           Params, param_shapes, init_params, forward, compute_logits
flash_jax/model/loader.py               WP3           expected_hf_names, convert_hf_weights, load_hf_checkpoint,
                                                      load_tokenizer
flash_jax/ops/__init__.py               WP1 verbatim  backend-dispatching ops: paged_attention, add_rms_norm,
                                                      rms_norm, swiglu
flash_jax/ops/common.py                 WP1 verbatim  MASK_VALUE, interpret_params(**overrides)
flash_jax/ops/reference.py              WP1 verbatim  pure-JAX semantics: write_kv, paged_attention, add_rms_norm,
                                                      rms_norm, swiglu
flash_jax/ops/tpu_rules.py              WP1 verbatim  TPU shape rules, VMEM/SMEM budgets and estimates,
                                                      check_attention_config (runner + wrappers)
flash_jax/ops/sampler.py                WP5           on-device sample()
flash_jax/ops/pallas/__init__.py        WP1 verbatim  (docstring only)
flash_jax/ops/pallas/ragged_paged_attention.py  WP2   Pallas RPA kernel
flash_jax/ops/pallas/rms_norm.py        WP2           Pallas fused residual-add + RMSNorm
flash_jax/ops/pallas/swiglu.py          WP2           Pallas gate_up matmul + SiLU*mul epilogue
flash_jax/testing/__init__.py           WP1 verbatim  (docstring only)
flash_jax/testing/tiny_models.py        WP1 verbatim  TINY_HF_CONFIGS, random_hf_weights, tiny_model,
                                                      write_hf_checkpoint
flash_jax/testing/numpy_reference.py    WP1 verbatim  float64 oracle: dense_logits, greedy_generate, check_greedy
flash_jax/testing/fake_runner.py        WP4           FakeRunner (simulated paged KV store, deterministic "model")
flash_jax/testing/tpu_lowering.py       WP2           lower_for_tpu(), tpu_lowering_available()
benchmarks/__init__.py                  WP6           empty; benchmarks run as `python -m benchmarks.<name>`
benchmarks/throughput.py                WP6           end-to-end throughput benchmark
benchmarks/kernels.py                   WP6           per-kernel microbenchmarks (Pallas vs reference)
tests/conftest.py                       WP1 verbatim  markers (tpu skip), fixtures (rng, tiny_name)
tests/test_core.py                      WP1
tests/test_pallas_attention.py          WP2
tests/test_pallas_norm_mlp.py           WP2
tests/test_tpu_lowering.py              WP2
tests/test_rope.py                      WP3
tests/test_model.py                     WP3
tests/test_loader.py                    WP3
tests/test_block_manager.py             WP4
tests/test_scheduler.py                 WP4
tests/test_sampler.py                   WP5
tests/test_model_runner.py              WP5
tests/test_llm.py                       WP5
tests/test_e2e.py                       WP6
```

`tests/` has **no** `__init__.py`. Tests import helpers only from `flash_jax.testing.*`, never from each other or
from `conftest.py`.

**Running from a checkout needs no install.** `pyproject.toml` sets `pythonpath = ["."]`, so plain `pytest` from
the repo root imports `flash_jax` (without it, loading `tests/conftest.py` fails with
`ModuleNotFoundError: No module named 'flash_jax'`, verified). Benchmarks run from the repo root as
`python -m benchmarks.throughput …`, which puts the root on `sys.path`; `python benchmarks/throughput.py` does
not. `uv pip install -e .` (the venv has no pip) is optional and only needed to import `flash_jax` from
elsewhere. Every WP and the README assume this.

---

## 3. Data structures and array layouts

### 3.1 Notation

| Symbol | Meaning |
|---|---|
| L | layers |
| H | hidden size |
| Hq, Hkv | query heads and KV heads |
| G = Hq/Hkv | query heads per KV head |
| D | model head dim |
| Dc | KV-cache head dim, ≥ D (zero-padded lanes) |
| I | intermediate size |
| V | vocab size |
| bs = ps | `block_size`, tokens per KV page |
| P | `num_kv_blocks`, pages per layer |
| MP | `pages_per_seq = cdiv(max_model_len, bs)` |
| T = T_pad | token bucket |
| S = S_pad | `min(T_pad, max_num_seqs)` |
| bq | `KernelConfig.attn_tile_q(T_pad)`: `min(attn_block_q, T_pad)`, or the small decode tile for `T_pad <= next_pow2(max_num_seqs)` (§5.4.6) |
| NB | `T_pad // bq`, number of q-tiles |
| ppb | `attn_block_kv // bs`, pages per kv tile |
| bkv | `ppb * bs` = `attn_block_kv`, tokens per kv tile |
| N | real (unpadded) token count |

**Units.** A **block** is a **page**: `bs` tokens, the unit of the block manager (`num_kv_blocks`, `block_size`,
`block_table`). A **kv tile** is `ppb` pages = `bkv` tokens, the unit the attention kernel streams
(`attn_block_kv`, `pair_num_kv_tiles`, `pair_kv_offset`). A **q-tile** is `bq` token rows.

### 3.2 KV cache: head-major pages, one array for all layers

```
kv_cache: [L, P, Hkv, 2, bs, Dc]   dtype = EngineConfig.dtype   (index 0 = K, 1 = V)
          allocated once with jnp.zeros; donated to every step and updated in place
```

- **Pages.** Block id `b` from the block manager is page `b` in every layer. `kv_cache[l, b]` is one contiguous
  `[Hkv, 2, bs, Dc]` slab, so one DMA fetches K and V for all heads.
- **Tiling.** The TPU tile covers the `(bs, Dc)` minor dims. For every Hkv there is no HBM padding, provided
  `bs % 16 == 0` (bf16) or `bs % 8 == 0` (f32), and `Dc % 128 == 0`.
- **Head-dim padding.** `Dc = kv_head_dim(model, engine, kernels) = round_up(D, align)`, where align is
  `EngineConfig.kv_head_dim_align`. When that is None, align is 128 if the attention backend resolves to
  `"pallas"`, else 1. The attention op zero-pads q/k/v to Dc and slices the output back to D. Zero lanes do not
  change q·k, and `sm_scale` always uses the real D.
- **Slots.** A slot is `page * bs + pos % bs`. Position `pos` of a sequence lives at
  `(page = block_table[pos // bs], offset = pos % bs)`.

### 3.3 Params pytree

Plain dicts of `jax.Array` in the engine dtype. Linear weights are `[in, out]`, and layers are stacked on
axis 0 for `lax.scan`.

```
embed        [V, H]
layers:
  attn_norm  [L, H]
  qkv        [L, H, Hkv*(G+2)*D]   fused; column order (kv_head j, slot c, d): c<G -> q head j*G+c, c=G -> k_j, c=G+1 -> v_j
  qkv_bias   [L, Hkv*(G+2)*D]      only if cfg.qkv_bias (same column order)
  o          [L, Hq*D, H]
  o_bias     [L, H]                only if cfg.o_bias
  q_norm     [L, D]                only if cfg.qk_norm
  k_norm     [L, D]                only if cfg.qk_norm
  mlp_norm   [L, H]
  gate_up    [L, H, 2*I]           fused, columns [gate | up]
  down       [L, I, H]
final_norm   [H]
lm_head      [H, V]                absent iff cfg.tie_word_embeddings (then logits use embed)
```

- **2-D weights.** Weights stay 2-D so small minor dims never pad TPU tiles. For example, a `[H, 2, I]` or
  `[H, Hkv, G+2, D]` weight would pad its tiles.
- **Grouped q heads.** Q head `h = j*G + c` uses KV head `j`, which is HF `repeat_kv` order. The grouped QKV
  order makes a contiguous output shard of `qkv` a whole GQA group, which matters for future TP.

### 3.4 Per-step batch (the jitted step's input)

`ScheduledSeq`, `AttentionMetadata`, `StepInput` and the builders, including **`prepare_step_input`**, the one
StepInput builder used by the runner, the model tests and the benchmarks, are defined verbatim in
`flash_jax/batch.py` (Appendix A). Every array is int32 unless stated. Shapes depend only on T_pad. Tokens are
laid out back to back in batch order: decodes, then partial prefills, then admissions.

| field | shape | real entries | padding / sentinel |
|---|---|---|---|
| `input_ids` | i32[T] | `token_ids[start:start+n]` | 0 |
| `positions` | i32[T] | `start + j` | 0 |
| `logits_indices` | i32[S] | `cu[i+1]-1` | 0 |
| `temperature` | f32[S] | `params.temperature` (≤0 = greedy) | 0.0 |
| `top_k` | i32[S] | `params.top_k` (0 = off) | 0 |
| `top_p` | f32[S] | `params.top_p` | 1.0 |
| `seeds` | i32[S] | `seq.seed` | 0 |
| `gen_idx` | i32[S] | `seq.num_output_tokens` (index of the token being sampled) | 0 |
| `attn.slot_mapping` | i32[T] | `page*bs + pos%bs` | **-1** (never written) |
| `attn.cu_q_lens` | i32[S+1] | `cu[0]=0`, prefix sums of q_len | repeat N |
| `attn.kv_lens` | i32[S] | `start + n` (length AFTER this step) | 0 |
| `attn.page_table` | i32[S*MP] | row i = block table of seq i, flattened | 0 (a valid page id) |
| `attn.num_seqs` | i32[1] | number of real seqs | — |
| `attn.pair_seq` | i32[NB+S] | work plan (§5.4.3) | 0 |
| `attn.pair_num_kv_tiles` | i32[NB+S] | work plan: kv tiles (not pages) to stream | 0 |
| `attn.pair_kv_offset` | i32[NB+S] | work plan | 0 |
| `attn.tile_pair_start` | i32[NB+1] | CSR row pointer over q-tiles | — |

**Invariant B1.** `positions[t] == kv_lens[i] - q_len_i + (t - cu[i])`: host RoPE and the kernel's causal mask
use the same position. **B2.** `1 <= q_len_i <= kv_lens[i] <= MP*bs`. **B3.** Real slots are distinct, and each
lies in its own sequence's pages.

All scalar-prefetch arrays are 1-D because 2-D SMEM arrays get padded.

**Worked example.** Settings: bs=4, T=16, S=4, MP=4, bq=8, bkv=8. Seq A is a decode (ctx 9, table [7,3,5]).
Seq B is a chunk (ctx 4, q 6, table [2,9,4]). Seq C is a new prompt (q 5, table [8,1]). The builder produces
this, verified with the verbatim code:

```
slot_mapping  [21, 36,37,38,39,16,17, 32,33,34,35,4, -1,-1,-1,-1]
cu_q_lens     [0, 1, 7, 12, 12]      kv_lens [10, 10, 5, 0]      num_seqs [3]
page_table    [7,3,5,0, 2,9,4,0, 8,1,0,0, 0,0,0,0]
pair_seq      [0, 1, 2, 2, 0, 0]     pair_num_kv_tiles  [2, 2, 1, 1, 0, 0]
pair_kv_offset[0, 2, 4, 5, 0, 0]     tile_pair_start    [0, 3, 4]
```

`prepare_step_input` reproduces these arrays from Sequences with the tables above (test config:
`block_size=4, attn_block_q=8, attn_block_kv=8, max_model_len=16, max_num_seqs=4, max_num_batched_tokens=16`;
batch: A decode `ScheduledSeq(A, 9, 1)`, B chunk `(B, 4, 6)`, C admission `(C, 0, 5)`), plus `positions`
`[9, 4..9, 0..4, 0×4]` and `logits_indices [0, 6, 11, 0]` (verified).

### 3.5 Host objects

- `Sequence` (verbatim, Appendix A) has these fields:
  - `seq_id`, which is also the arrival order and the priority;
  - `token_ids` (prompt + outputs), `num_prompt_tokens`, `params`, `seed`, `max_tokens` (clamped);
  - `status`;
  - `block_table`;
  - `block_hashes` (memo of chained full-block hashes);
  - `num_sealed_blocks`, `num_computed_tokens`, `num_cached_tokens`, `num_preemptions`;
  - `finish_reason`.

  Properties: `num_tokens`, `num_remaining`, `output_token_ids`, `num_output_tokens`, `is_finished`.
- `ScheduledSeq(seq, start_pos, num_new_tokens, samples)` is frozen. `samples` is
  `start_pos + n == seq.num_tokens`, evaluated at schedule time.

---

## 4. Public signatures and contracts

Everything in WP1 files is fixed by Appendix A. This section lists the cross-module API of the other packages.
Type aliases: `Array = jax.Array`, `Params = dict[str, Any]`.

### 4.1 Core (WP1, Appendix A). Summary of what the others rely on

```python
# flash_jax/config.py
def cdiv(a: int, b: int) -> int; def round_up(a: int, b: int) -> int; def next_pow2(n: int) -> int
class ModelConfig:  # frozen; fields in Appendix A
    q_per_kv: int; qkv_width: int                                    # properties
    @classmethod from_hf(hf: dict, generation_config: dict | None = None) -> ModelConfig
    @classmethod from_pretrained(path: str) -> ModelConfig
class EngineConfig:  # frozen, validated in __post_init__
    pages_per_seq: int; attn_pages_per_kv_tile: int                  # properties
    def token_buckets(self) -> tuple[int, ...]; def bucket_for(self, num_tokens: int) -> int
    def num_seq_slots(self, t_pad: int) -> int
class KernelConfig:  # frozen: attention/norm/mlp in {"pallas","interpret","reference"}, attn_block_q,
                     # attn_block_q_small, attn_small_bucket, attn_pages_per_kv_tile
    def attn_tile_q(self, t_pad: int) -> int   # THE q-tile of a bucket: work plan (prepare_step_input) and kernel
def resolve_backend(requested: str, platform: str | None = None) -> str
def resolve_kernel_config(cfg: EngineConfig, platform: str | None = None, *, q_per_kv: int = 1) -> KernelConfig
def kv_head_dim(model: ModelConfig, cfg: EngineConfig, kc: KernelConfig) -> int

# flash_jax/batch.py
class ScheduledSeq; class AttentionMetadata(NamedTuple); class StepInput(NamedTuple)
def build_work_plan(cu_q_lens, kv_lens, num_seqs, t_pad, block_q, block_kv) -> tuple[np.ndarray x4]
def build_attention_metadata(q_lens, kv_lens, block_tables, *, t_pad, s_pad, block_size, pages_per_seq,
                             block_q, block_kv) -> AttentionMetadata   # numpy arrays
def prepare_step_input(batch: Sequence[ScheduledSeq], cfg: EngineConfig, kc: KernelConfig, *,
                       t_pad: int | None = None) -> StepInput         # numpy; t_pad forces a bucket (dummy input)

# flash_jax/ops/__init__.py   (backend: "pallas" | "interpret" | "reference")
def paged_attention(q, k, v, kv_cache, layer, md, *, sm_scale, backend, block_q, pages_per_kv_tile) -> (o, kv_cache)
def add_rms_norm(x, residual, weight, eps, *, backend) -> (y, new_residual)
def rms_norm(x, weight, eps) -> y
def swiglu(h, w_gate_up, *, backend) -> Array
# flash_jax/ops/common.py:  MASK_VALUE: float; def interpret_params(**overrides) -> pltpu.InterpretParams
# flash_jax/ops/reference.py: write_kv, paged_attention(..., chunk=32, max_chunk_bytes=256 MiB), add_rms_norm,
#                             rms_norm, swiglu (semantics = contracts)
# flash_jax/ops/tpu_rules.py: LANES, SMEM_BUDGET_BYTES, VMEM_BUDGET_BYTES, itemsize, sublanes,
#                             attention_smem_bytes, attention_vmem_bytes, attention_problems, check_attention_config
# flash_jax/testing/tiny_models.py: TINY_HF_CONFIGS, random_hf_weights, tiny_model, write_hf_checkpoint
# flash_jax/testing/numpy_reference.py: dense_logits, greedy_generate, check_greedy, GreedyCheck
```

### 4.2 WP2: Pallas kernels

```python
# All three kernels take `interpret: bool | pltpu.InterpretParams`: False = compiled Mosaic (TPU rules checked),
# True = ops.common.interpret_params(), an InterpretParams instance is used as is (tests: eager DMAs, races).

# flash_jax/ops/pallas/ragged_paged_attention.py
def ragged_paged_attention(q: Array, kv_cache: Array, layer: Array, md: AttentionMetadata, *, sm_scale: float,
                           block_q: int, pages_per_kv_tile: int,
                           interpret: bool | pltpu.InterpretParams = False) -> Array:
    """Causal ragged paged attention over an ALREADY-WRITTEN cache (read-only on the cache).
    q [T, Hq, Dc] (T % block_q == 0), kv_cache [L, P, Hkv, 2, bs, Dc], layer: i32 scalar (traced OK).
    md's work plan MUST have been built with (block_q, pages_per_kv_tile * bs).
    Returns o [T, Hq, Dc] in q.dtype, equal to ops.reference.paged_attention within f32 1e-4 / bf16 3e-2;
    rows t >= N are exactly 0. When interpret is False, raises ValueError listing every
    ops.tpu_rules.attention_problems(...) violation (§5.4.6). Every matmul pins precision=DEFAULT."""

# flash_jax/ops/pallas/rms_norm.py
def add_rms_norm(x: Array, residual: Array, weight: Array, eps: float, *,
                 interpret: bool | pltpu.InterpretParams = False) -> tuple[Array, Array]:
    """Same contract as ops.reference.add_rms_norm (bit-exact in f32 on CPU interpret). x, residual [T, H]
    (any T >= 1), weight [H]. Residual buffer aliased to output 1 (input_output_aliases={1: 1}). Row tile from
    the VMEM budget (§5.5.1); if no tile fits, warns and returns ops.reference.add_rms_norm."""

# flash_jax/ops/pallas/swiglu.py
def swiglu(h: Array, w_gate_up: Array, *, interpret: bool | pltpu.InterpretParams = False) -> Array:
    """Same contract as ops.reference.swiglu for ANY h [T, H] (every row written). w_gate_up [H, 2I] -> [T, I].
    Never materializes [T, 2I]. Every matmul pins precision=DEFAULT. If no tiling fits the VMEM budget, or
    (compiled only) the tiles violate the TPU rules of §5.5.2, falls back to ops.reference.swiglu with a
    warnings.warn (no global flag: Python's default filter shows it once per call site and message)."""

# flash_jax/testing/tpu_lowering.py
def tpu_lowering_available() -> bool
    """True iff the private hook exists: jax._src.pallas.mosaic.tpu_info.registry is a dict and
    tpu_info.get_tpu_info has cache_clear (tests skip otherwise)."""
def lower_for_tpu(fn: Callable, *avals: jax.ShapeDtypeStruct, chip: str = "v5e") -> jax.stages.Lowered
    """chip in {"v5e", "v6e"}. Inside try/finally:
         ti.registry["cpu"] = lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion(chip), 1)   # enum, not str
         ti.get_tpu_info.cache_clear()          # get_tpu_info is memoized: without this "v6e" silently lowers as v5e
         assert pltpu.get_tpu_info().chip_version.value == chip
         return jax.jit(fn).trace(*avals).lower(lowering_platforms=("tpu",))
       finally: restore the previous registry entry (or delete it) and cache_clear() again."""
```

### 4.3 WP3: model and weights

```python
# flash_jax/model/rope.py
def rope_inv_freq(cfg: ModelConfig) -> np.ndarray         # f32[D/2]; llama3 scaling (HF _compute_llama3_parameters)
def rope_cos_sin(positions: Array, inv_freq: np.ndarray) -> tuple[Array, Array]   # f32 [T, D/2] each
def apply_rope(x: Array, cos: Array, sin: Array) -> Array  # x [T, h, D]; HF rotate_half; f32 math, returns x.dtype

# flash_jax/model/llama.py
Params = dict[str, Any]
def param_shapes(cfg: ModelConfig, dtype: str | jnp.dtype) -> Params
    """Pytree of jax.ShapeDtypeStruct exactly as §3.3 (optional keys present iff their flag is set)."""
def init_params(cfg: ModelConfig, key: jax.Array, dtype: str | jnp.dtype) -> Params
    """Random params for benchmarks: normal(0, 0.02) matrices/biases, ones for norms. Shapes = param_shapes."""
def forward(params: Params, kv_cache: Array, inp: StepInput, cfg: ModelConfig, kc: KernelConfig
            ) -> tuple[Array, Array]:
    """Jittable. Returns (hidden [S, H] final-normed hidden state at logits_indices, updated kv_cache).
    Writes K/V of every real token (slot >= 0) of every layer; nothing else in the cache changes."""
def compute_logits(params: Params, hidden: Array, cfg: ModelConfig) -> Array
    """f32 [S, V] = hidden @ lm_head (or embed.T when tied), preferred_element_type=f32."""

# flash_jax/model/loader.py
def expected_hf_names(cfg: ModelConfig) -> set[str]
    """Exact set of HF tensor names the checkpoint must provide (lm_head.weight iff not tied)."""
def convert_hf_weights(get: Mapping[str, np.ndarray] | Callable[[str], np.ndarray], cfg: ModelConfig,
                       dtype: str | jnp.dtype) -> Params
    """HF-named [out, in] tensors -> §3.3 params (fused, stacked, transposed), device_put one stacked
    group at a time. `get` is a Mapping (e.g. the dict from tiny_model; wrapped as get.__getitem__) or a
    callable name -> array (the lazy safetensors reader). Arrays may be float32/bfloat16 numpy (ml_dtypes);
    the cast happens on host."""
def load_hf_checkpoint(path: str, dtype: str = "bfloat16") -> tuple[ModelConfig, Params]
    """config.json (+generation_config.json) + model.safetensors[.index.json] or *.safetensors.
    Missing expected names -> KeyError. Unexpected names -> ValueError, except the ignorable ones
    (*.rotary_emb.inv_freq, and lm_head.weight when tied)."""
def load_tokenizer(path: str) -> Tokenizer | None
    """<path>/tokenizer.json via the optional `tokenizers` package, wrapped to the Tokenizer protocol; None if
    unavailable."""
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
```

### 4.4 WP4: block manager and scheduler

The normative code is Appendix B.1–B.2.

```python
# flash_jax/engine/block_manager.py
ROOT_HASH: bytes
def hash_block(parent: bytes, tokens: Sequence[int]) -> bytes
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True) -> None
    num_blocks: int; block_size: int; ref_counts: list[int]; block_hash: list[bytes | None]
    hash_to_block: dict[bytes, int]; free_blocks: OrderedDict[int, None]
    num_query_tokens: int; num_hit_tokens: int; num_evictions: int
    @property num_free_blocks -> int
    def find_cached_prefix(self, seq: Sequence) -> list[int]
    def can_allocate(self, seq: Sequence, cached: list[int], num_new_tokens: int) -> bool
    def allocate(self, seq: Sequence, cached: list[int], num_new_tokens: int) -> None
    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool
    def append_slots(self, seq: Sequence, num_new_tokens: int) -> None
    def free(self, seq: Sequence) -> None
    def seal_computed_blocks(self, seq: Sequence) -> None
    def check_invariants(self, live: Iterable[Sequence]) -> None      # raises AssertionError

# flash_jax/engine/scheduler.py
class EngineDeadlock(RuntimeError)
class Scheduler:
    def __init__(self, config: EngineConfig, num_kv_blocks: int, eos_token_ids: Sequence[int] = ()) -> None
    config: EngineConfig; block_manager: BlockManager; eos_token_ids: frozenset[int]
    waiting: deque[Sequence]; running: list[Sequence]; num_preemptions: int
    def add(self, seq: Sequence) -> None                  # seq_ids must be added in increasing order
    def has_unfinished(self) -> bool
    def schedule(self) -> list[ScheduledSeq]             # non-empty whenever has_unfinished(); else raises EngineDeadlock
    def update(self, batch: list[ScheduledSeq], sampled_tokens: Sequence[int]) -> list[Sequence]  # finished seqs
    def abort(self, seq_id: int) -> Sequence | None
    def check_batch(self, batch: list[ScheduledSeq]) -> None   # S1-S4 + I7

# flash_jax/testing/fake_runner.py
class FakeRunner:
    def __init__(self, num_kv_blocks: int, block_size: int, vocab_size: int = 1000) -> None
    def run(self, batch: list[ScheduledSeq]) -> np.ndarray   # same contract as ModelRunner.run
    @staticmethod def next_token(context: Sequence[int], vocab_size: int) -> int
    def reference_output(self, prompt: Sequence[int], max_tokens: int, stop_token_ids: Sequence[int] = ()) -> list[int]
```

### 4.5 WP5: sampler, runner and LLM API

```python
# flash_jax/ops/sampler.py
def sample(logits: Array, temperature: Array, top_k: Array, top_p: Array, seeds: Array, gen_idx: Array, *,
           max_candidates: int) -> Array
    """logits f32[S, V] -> i32[S]. temperature<=0: exact argmax (lowest index on ties). Otherwise sampling from
    softmax(logits/temperature) restricted by top_k (0=off) then top_p (1=off). Draws depend only on
    (logits row, seeds[i], gen_idx[i]), never on the row position or batch composition."""

# flash_jax/engine/model_runner.py   (imports flash_jax.model lazily, inside _step / ModelRunner.__init__)
TRACE_COUNTER: collections.Counter[str]    # step_fn increments TRACE_COUNTER["step_fn"] at TRACE time
step_fn = jax.jit(_step, static_argnames=("mcfg", "kc", "max_candidates"), donate_argnames=("kv_cache",))
    # step_fn._cache_size() = number of compiled executables (compiles, not traces; §5.3)
def _step(params: Params, kv_cache: Array, inp: StepInput, *, mcfg: ModelConfig, kc: KernelConfig,
          max_candidates: int) -> tuple[Array, Array]           # (tokens i32[S], kv_cache)
class ModelRunner:
    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, params: Params) -> None
    model_config; engine_config; kernel_config: KernelConfig; params: Params
    arch: ModelConfig          # static key of step_fn: model_config with eos_token_ids=() and max_position_embeddings=0
    device: jax.Device         # jax.local_devices()[0]; params, kv_cache and every StepInput are committed to it
    kv_head_dim: int; num_kv_blocks: int; kv_cache: Array
    def prepare(self, batch: list[ScheduledSeq]) -> StepInput   # = batch.prepare_step_input(batch, engine_config, kernel_config)
    def dummy_input(self, t_pad: int) -> StepInput              # = prepare_step_input([], ..., t_pad=t_pad): num_seqs=0
    def lower(self, t_pad: int, num_blocks: int | None = None) -> jax.stages.Lowered
        """step_fn.lower(params, ShapeDtypeStruct cache of num_blocks (default num_kv_blocks) pages committed to
        device, device_put(dummy_input(t_pad))). Never touches self.kv_cache. .compile() shares the executable
        with later step_fn calls of the same shapes (verified), so memory_analysis() costs no extra compile."""
    def run(self, batch: list[ScheduledSeq]) -> np.ndarray               # i32[len(batch)]
    def warmup(self) -> float                                            # compiles every bucket; returns seconds
    @property buckets_used -> set[int]                                   # T_pad values dispatched so far

# flash_jax/engine/llm.py
@dataclass
class RequestOutput:
    request_id: int; prompt_token_ids: list[int]; output_token_ids: list[int]; text: str | None
    finish_reason: str          # "stop" | "length" | "abort"
    num_cached_tokens: int; num_preemptions: int
class LLM:
    def __init__(self, model: str | None = None, *, model_config: ModelConfig | None = None,
                 params: Params | None = None, engine_config: EngineConfig | None = None,
                 tokenizer: Tokenizer | None = None) -> None
    model_config; engine_config; runner: ModelRunner; scheduler: Scheduler; tokenizer
    def add_request(self, prompt: str | Sequence[int], sampling_params: SamplingParams | None = None) -> int
    def step(self) -> list[RequestOutput]    # queued abort outputs + requests that finished in this step
    def has_unfinished(self) -> bool
    def abort(self, request_id: int) -> bool
    def generate(self, prompts: str | Sequence[str] | Sequence[Sequence[int]],
                 sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
                 *, progress: bool = False) -> list[RequestOutput]           # in input order
    def warmup(self) -> float
    last_step: dict[str, float | int]   # set by every step() that ran the model:
        # num_seqs, num_tokens (N), num_decodes, t_pad, seconds (wall time of runner.run, incl. the token sync);
        # a step is "decode-only" iff num_decodes == num_seqs
    @property stats -> dict[str, int]
        # keys: steps (model steps run), num_preemptions, prefix_query_tokens, prefix_hit_tokens,
        #       prefix_evictions, num_kv_blocks,
        #       prompt_tokens    = sum of prompt lengths of all requests added,
        #       generated_tokens = sum of output tokens appended (a recompute after preemption appends nothing),
        #       num_traces   = TRACE_COUNTER["step_fn"]      (process-wide),
        #       num_compiles = step_fn._cache_size()          (process-wide; executables, the real compile count)
```

**LLM contracts**

- **Construction.** Exactly one of `model` or `model_config` must be given; otherwise raise `ValueError`.
  - `model` is a checkpoint directory: `load_hf_checkpoint(model, engine_config.dtype)`, and the tokenizer
    defaults to `load_tokenizer(model)`.
  - With `model_config`, `params` must be a §3.3 Params pytree, for example from `convert_hf_weights`; it is
    cast to the engine dtype and `device_put` to `runner.device`. If `params` is None, `init_params` is used
    with `jax.random.key(engine_config.seed)`.
  - `engine_config=None` means `EngineConfig()`. `sampling_params=None` means `SamplingParams()`, which has
    temperature 1.0 as in vLLM; tests pass `temperature=0`.
- **`add_request`** validates, before touching the scheduler, and raises `ValueError` on:
  - `not 1 <= len(prompt) < max_model_len`;
  - a `str` prompt without a tokenizer;
  - any token id outside `[0, model_config.vocab_size)` (the embedding gather would silently clamp it:
    `e[jnp.array([12])]` on a 10-row table returns the last row, verified).

  It clamps `max_tokens = min(params.max_tokens, max_model_len - len(prompt))`. The seed is
  `params.seed % 2**31` (StepInput seeds are int32; a raw `2**40` raised `OverflowError` inside
  `prepare_step_input`, mid-step, verified), or else `int(self._rng.integers(0, 2**31 - 1))` from
  `np.random.default_rng(engine_config.seed)`. `request_id` is the `seq_id` (0, 1, 2, …).
- **`step`** does, in order:
  1. `batch = scheduler.schedule()`;
  2. if `batch` is empty (nothing pending), skip to the return: **the runner is never called with an empty
     batch** (it would run a full dummy step in the smallest bucket);
  3. if `debug_checks`, `scheduler.check_batch(batch)`;
  4. `tokens = runner.run(batch)`, timed into `last_step`;
  5. `finished = scheduler.update(batch, tokens)`;
  6. if `debug_checks`, `block_manager.check_invariants([*scheduler.running, *scheduler.waiting])`
     (`running` is a list and `waiting` a deque, so `running + waiting` is a `TypeError`).

  It returns the queued abort outputs (below) followed by a `RequestOutput` for every sequence in `finished`.
  `text` is `tokenizer.decode(output_token_ids)` when a tokenizer exists.
- **`abort(request_id)`** calls `scheduler.abort`; if a sequence was found it queues
  `RequestOutput(finish_reason="abort", output_token_ids=<generated so far>, …)`, which the **next** `step()`
  returns (even when no model step runs), and returns True. Otherwise it returns False.
- **`generate`** adds every prompt, steps until those requests finish, and returns outputs in input order.
  - A `str`, or a flat sequence of ints, counts as **one** prompt; the result is still a list.
  - A single `SamplingParams` applies to all prompts. A sequence of them must match the prompts in length.
  - An aborted request counts as finished; its output has `finish_reason == "abort"`.
  - `progress=True` prints a one-line progress counter to stderr; there is no tqdm dependency.

---

## 5. Algorithms

### 5.1 Block manager and prefix cache (WP4, normative code in Appendix B.1)

**State**

- `ref_counts[b]`.
- `block_hash[b]`: set once the block is full and its KV is computed ("sealed").
- `hash_to_block`.
- `free_blocks`: **one** `OrderedDict` LRU holding every ref-0 block, sealed or not. The front is evicted
  first. There is no separate "cached pool". A sealed free block remains a cache hit until it is popped for
  reuse; popping it is the eviction.

**Hash.**
`hash_block(parent, tokens) = sha256(parent || np.int32(tokens).tobytes()).digest()`, with
`ROOT_HASH = sha256(b"flash-jax/prefix-cache/v1")` for block 0. The hash is chained, so identical tokens after
a different prefix never alias. It is deterministic across processes and collision-safe. The hashes of the
full blocks of `token_ids` are memoized on `seq.block_hashes`. Since `token_ids` only grows, the memo stays
valid across preemptions.

**Operations**

- **Lookup, `find_cached_prefix(seq)`.** Returns `[]` when caching is off. Otherwise it walks the chain over
  the first `(num_tokens - 1) // bs` full blocks and stops at the first miss. It does not mutate anything.
  - The `-1` cap guarantees at least one token is computed, so there are logits to sample. A fully cached
    prompt of `k*bs` tokens therefore recomputes its last block privately.
  - Consequence: **a step never writes into a shared or sealed block**, so no copy-on-write is needed.
- **Admission check, `can_allocate(seq, cached, n)`.**
  - `need = cdiv(len(cached)*bs + n, bs) - len(cached)`.
  - `revived` = number of hits with ref 0 (they sit in `free_blocks`).
  - Admissible iff `len(free) - revived >= need`.
- **Admission, `allocate(seq, cached, n)`.**
  1. **Touch hits first**: `ref += 1`, and remove the hit from `free_blocks` if it was there.
  2. Set `num_computed_tokens = num_cached_tokens = len(cached)*bs` and `num_sealed_blocks = len(cached)`.
  3. Call `append_slots(seq, n)`.

  If fresh blocks were popped before the hits were touched, a pop could evict and hand out the very block
  being reused.
- **Growth, `can_append` / `append_slots(seq, n)`.** Grows `block_table` to `cdiv(num_computed + n, bs)`
  entries. Pages are allocated on demand, one chunk at a time.
- **Pop (`_pop_free`).** Takes the LRU front. If that block is sealed, unregister its hash; this is the
  eviction. Then set ref to 1.
- **Free, `free(seq)`.** Walk `reversed(block_table)` and decrement refs. A block that reaches 0 goes to the
  MRU end. Tail blocks therefore become older than head blocks and are evicted first; head blocks are the
  more shareable ones.
  - **Unhashed** blocks are moved to the **front**, because they have no reuse value.
  - Sealed blocks keep their hash.
  - Afterwards `block_table = []` and `num_sealed_blocks = 0`.
- **Sealing, `seal_computed_blocks(seq)`.** Runs in `update()` after every step, before a finished sequence is
  freed. For each `i` in `[num_sealed_blocks, num_computed_tokens // bs)`:
  - if hash `h_i` is unknown, register it;
  - if an identical block was already published (for example, duplicate prompts admitted in the same step),
    leave this block private and unhashed.

  **A block is published only after its KV exists, and a published block is immutable.**

**Invariants.** `check_invariants(live)`, where live = every WAITING and RUNNING sequence.

| # | invariant |
|---|---|
| I1 | `ref_counts[b]` equals the number of occurrences of b across all live block tables |
| I2 | b is in `free_blocks` iff `ref_counts[b] == 0` |
| I3 | `hash_to_block` is a bijection with `{b : block_hash[b] is not None}` |
| I4 | no block appears twice in a table; WAITING sequences hold no blocks |
| I5 | for a RUNNING sequence, `len(block_table) == cdiv(num_computed_tokens, bs)` at step boundaries |
| I6 | each sealed block of a running sequence has the hash recomputed from its tokens |
| I7 | (in `Scheduler.check_batch`) every block written this step has ref 1 and no hash |

### 5.2 Scheduler (WP4, normative code in Appendix B.2)

Every sequence has `token_ids` and `num_computed_tokens`. Its work in a step is `n` new tokens starting at
`start_pos = num_computed_tokens`:

- a decode is `num_remaining == 1`;
- a prefill chunk is anything else;
- recompute after preemption is just a prefill of prompt plus outputs.

`schedule()` works as follows. It starts with `budget = max_num_batched_tokens`.

1. **Pass 1, decodes.** RUNNING sequences with `num_remaining == 1`, in `seq_id` order.
2. **Pass 2, partial prefills.** The remaining RUNNING sequences, in `seq_id` order, with
   `n = min(num_remaining, budget, max_prefill_chunk or budget)`.

   In passes 1 and 2, while `!can_append(seq, n)`:
   - The **victim** is the RUNNING sequence with the largest `seq_id` that is not yet scheduled this step.
     This **can be `seq` itself**.
   - Preempt it (recompute mode):
     - free its blocks (sealed ones stay cached);
     - `num_computed_tokens = 0`;
     - WAITING;
     - insert into `waiting` in `seq_id` order.
   - If the victim was `seq`, skip `seq`. Otherwise retry.

   On success, `append_slots(seq, n)` and emit `ScheduledSeq(seq, start, n, samples = n == num_remaining)`.
3. **Pass 3, admission.** Skipped **if anything was preempted this step** (anti-thrash). While `waiting`,
   `budget > 0` and `len(running) < max_num_seqs`:
   - `seq = waiting[0]`;
   - `cached = find_cached_prefix(seq)`;
   - `n = min(num_tokens - len(cached)*bs, budget, cap)`;
   - if `!can_allocate`, **stop** (head-of-line blocking gives FIFO fairness);
   - otherwise allocate, mark RUNNING, append to `running`, and emit.
4. If nothing was emitted but requests are pending, raise `EngineDeadlock`.

   This cannot happen when `num_kv_blocks >= pages_per_seq`, which the runner validates. **Progress is
   guaranteed:** if nothing else was scheduled, every running sequence is unscheduled, so the victim loop
   preempts all newer ones and the lowest-`seq_id` running sequence then has the whole pool (cached free blocks
   are evictable), which holds `pages_per_seq` blocks; with nothing running, `waiting[0]` is admissible for the
   same reason.

   **The policy is not strict priority.** Victims are drawn only from the sequences not yet scheduled in this
   step, and pass 1 schedules newer decodes before pass 2 reaches an older partial prefill, which may therefore
   preempt itself while newer sequences run (19 such steps in 600 FakeRunner workloads, §0.3). It cannot
   starve: it re-enters `waiting` at its `seq_id` position, admission is FIFO with head-of-line blocking, and
   nothing is admitted in a preempting step, so no newer sequence is admitted before it and it is re-admitted
   at the latest when the running set drains. Un-scheduling newer decodes to enforce strict priority is
   deliberately not done: it would un-append blocks within a step and was not validated.

**`update(batch, tokens)`** walks the batch in order with `zip(batch, tokens, strict=True)`: a runner that
returns the wrong number of tokens raises `ValueError` at once instead of leaving blocks appended and
`num_computed_tokens` stale. For each `(item, tok)`:

1. `num_computed += n`.
2. `seal_computed_blocks`.
3. If `item.samples`:
   - append `tok`;
   - finish with `"stop"` if `tok ∈ eos and !ignore_eos` or `tok ∈ stop_token_ids`;
   - otherwise finish with `"length"` if `num_output_tokens >= max_tokens` or `num_tokens >= max_model_len`;
   - on finish: FINISHED, `free`, remove from `running`.

   EOS and stop tokens are included in `output_token_ids`, matching HF and the oracle. Partial chunks discard
   the sampled token.

**Properties**

- **Chunked prefill is budget truncation.** No extra code is needed.
- **Mixed batches** of decodes and chunks are the norm.
- **Preemption plus prefix caching makes recompute cheap.**
- **Batch token order** is decodes, then partial prefills, then admissions. Decode tokens pack densely into
  attention q-tiles.

**Scheduler-output checks, `check_batch`**

| # | check |
|---|---|
| S1 | Σn ≤ budget, and #seqs ≤ `max_num_seqs` |
| S2 | no sequence is scheduled twice |
| S3 | `start_pos == num_computed_tokens`, `1 ≤ n ≤ num_tokens - start`, and `len(block_table) == cdiv(start+n, bs)` |
| S4 | `samples` is true iff the chunk reaches the end |
| I7 | as in §5.1 |

### 5.3 Model runner (WP5)

**Buckets and compilation**

- The only free shape dimension is `T_pad ∈ EngineConfig.token_buckets()`: powers of two from
  `min_token_bucket` up to `next_pow2(max_num_batched_tokens)`.
- Everything else derives from T_pad: `S_pad = min(T_pad, max_num_seqs)`, `bq = kc.attn_tile_q(T_pad)`,
  `NB = T_pad // bq`. MP is static.
- `step_fn` is a **module-level** `jax.jit`:
  - `static_argnames=("mcfg", "kc", "max_candidates")`, all hashable;
  - `mcfg` is the runner's **architecture-only key** `arch = dataclasses.replace(model_config,
    eos_token_ids=(), max_position_embeddings=0)`: neither field affects the step (EOS handling is host-side),
    so engines that differ only in EOS share executables (verified: 3 EOS variants added 0 executables);
  - `donate_argnames=("kv_cache",)`;
  - params passed as arguments, never closed over.

  Hence at most `len(token_buckets())` compilations per distinct `(arch, KernelConfig, max_candidates)`, and
  engines with identical configs share executables (tests benefit).
- **Every input is committed to one device.** `self.device = jax.local_devices()[0]`; params are
  `jax.device_put(params, self.device)`, the cache is `jnp.zeros(shape, dtype, device=self.device)`, and each
  StepInput is `jax.device_put(prepare(batch), self.device)`. JAX recompiles **without retracing** when
  committedness or sharding changes: a committed param with an uncommitted cache compiled twice per bucket
  (the donated output cache comes back committed) while the trace counter stayed at 1 (§0.3).
- **Compiles are counted, not traces.** `step_fn._cache_size()` (private, exists in 0.10.2) is the number of
  executables and is what the compile-bound test asserts; `TRACE_COUNTER["step_fn"] += 1` runs inside `_step`
  at trace time and is kept as a diagnostic. `stats` reports both (`num_compiles`, `num_traces`).

```python
def _step(params, kv_cache, inp, *, mcfg, kc, max_candidates):
    TRACE_COUNTER["step_fn"] += 1
    hidden, kv_cache = llama.forward(params, kv_cache, inp, mcfg, kc)          # [S, H]
    logits = llama.compute_logits(params, hidden, mcfg)                        # f32 [S, V]
    tokens = sample(logits, inp.temperature, inp.top_k, inp.top_p, inp.seeds, inp.gen_idx,
                    max_candidates=max_candidates)
    return tokens, kv_cache
```

**`__init__`**

1. `self.device = jax.local_devices()[0]`; `self.arch` as above.
2. `kc = resolve_kernel_config(ecfg, q_per_kv=mcfg.q_per_kv)`.
3. `Dc = kv_head_dim(mcfg, ecfg, kc)`.
4. Cast `params` to `ecfg.dtype` if needed and `device_put` them to `self.device`.
5. Validate, **before** anything large is allocated:
   - if `kc.attention == "pallas"`: `ops.tpu_rules.check_attention_config(mcfg, ecfg, kc, Dc)`. It checks every
     bucket (with that bucket's `attn_tile_q`): `Dc % 128 == 0` (explicitly: `kv_head_dim_align=64` makes it
     64), `bs % sublanes == 0`, `attn_block_kv % 128 == 0`, `(bq*G) % sublanes == 0`, the SMEM budget and
     the VMEM estimate (§5.4.6). It raises one `ValueError` listing every violation with the knob to change;
   - if `kc.attention == "reference"` on a TPU platform: warn that the reference attention is a correctness
     fallback only (it gathers the whole `max_model_len` context per token, §9.1 R1), including the per-token
     row size `MP*bs*Hkv*2*Dc*4` bytes, and that it cannot bound temporaries below one row;
   - warn if `max_model_len > max_position_embeddings`.
6. Size the pool. `num_kv_blocks = ecfg.num_kv_blocks` if set. Otherwise:
   - `bytes_per_block = L * Hkv * 2 * bs * Dc * itemsize`;
   - `stats = self.device.memory_stats()`;
   - if `stats` is None or lacks `"bytes_limit"` (CPU): `num_kv_blocks = cpu_kv_cache_bytes // bytes_per_block`;
   - otherwise (TPU): `free = stats["bytes_limit"] - stats.get("bytes_in_use", 0)` (params already resident);
     `P = int(free * kv_memory_fraction) // bytes_per_block`; then measure the step's own temporaries (the
     `[S, V]` f32 logits and full-vocab Gumbel noise, top-k temps, MLP activations, and with the reference
     attention the gather temps): `temp = self.lower(max_bucket, num_blocks=P).compile().memory_analysis()
     .temp_size_in_bytes`. If `free - P * bytes_per_block < temp + 256 MiB`, shrink
     `P = (free - temp - 256 MiB) // bytes_per_block`. When P is unchanged the executable is reused by
     warmup/the first step (no extra compile); when it shrinks, that one bucket compiles once more.
   - `num_kv_blocks >= pages_per_seq`, otherwise raise `ValueError` asking for more blocks or a smaller
     `max_model_len`. This guarantees that any single sequence can finish.
7. `kv_cache = jnp.zeros((L, P, Hkv, 2, bs, Dc), dtype, device=self.device)`.

**`prepare(batch)` = `batch.prepare_step_input(batch, engine_config, kernel_config)`** (WP1, verbatim,
Appendix A.6). Numpy only; one Python iteration per sequence, never per token.

1. `N = Σ n_i`, `T = bucket_for(N)` (or the forced `t_pad`), `S = num_seq_slots(T)`, `bq = kc.attn_tile_q(T)`.
2. `md = build_attention_metadata([n_i], [start_i + n_i], [seq_i.block_table], t_pad=T, s_pad=S, block_size=bs,
   pages_per_seq=MP, block_q=bq, block_kv=attn_block_kv)`.
3. Fill `input_ids`, `positions`, `logits_indices = cu[1:n+1]-1`, and the sampling arrays per the §3.4 table,
   with its padding values.

**`run(batch)`** (never called with an empty batch; `LLM.step` short-circuits)

1. `inp = jax.device_put(self.prepare(batch), self.device)`, one pytree transfer.
2. `tokens, self.kv_cache = step_fn(self.params, self.kv_cache, inp, mcfg=self.arch, kc=..., max_candidates=...)`.
3. `return np.asarray(tokens)[:len(batch)]`. This is the single device→host sync per step.

**`warmup()`** runs `step_fn` on `dummy_input(t)` for every bucket t (`num_seqs = 0`: all slots -1, empty plan,
temperature 0). It writes nothing to the cache and returns the elapsed seconds. It is used by the benchmark;
tests do not call it.

**`lower(t_pad, num_blocks=None)`** lowers `step_fn` for one bucket against an abstract cache
(`jax.ShapeDtypeStruct` with `SingleDeviceSharding(self.device)`), so pool sizing (step 6) and the benchmark's
`memory_analysis()` report never touch or donate the real cache.

**Host/device overlap.** None in v1 beyond JAX async dispatch. The loop is synchronous: schedule → prepare →
dispatch → wait for tokens → update. Lookahead scheduling is future work (§9).

**In-place cache.** The cache is a `lax.scan` carry. The KV write is an `.at[].set` scatter on the carry, and
the Pallas kernel reads the whole cache through `pl.ANY` with the layer index as a scalar. No per-layer slice
is materialized, and the compiled step aliases the donated input (verified on CPU).

### 5.4 Ragged paged attention (ops dispatch = WP1; kernel = WP2, normative code in Appendix B.3)

#### 5.4.1 Op contract (`flash_jax.ops.paged_attention`, verbatim in Appendix A)

1. Zero-pad q/k/v from D to Dc lanes.
2. `kv_cache = reference.write_kv(kv_cache, layer, k, v, md.slot_mapping)`. **Both backends share this write.**
3. `o = reference.paged_attention(...)` for `"reference"`, or
   `pallas.ragged_paged_attention(..., interpret=backend == "interpret")` otherwise.
4. Return `(o[..., :D], kv_cache)`.

Semantics: token t of sequence i sits at `pos = kv_lens[i] - q_len_i + (t - cu[i])` and attends causally to
cache positions `[0, pos]` of its own pages, including the tokens written in this same step. Rows `t >= N` are
exactly 0.

#### 5.4.2 How new KV gets written

The write is an XLA scatter placed right next to attention in the same jitted layer body:

```python
page = where(slot >= 0, slot // bs, P + arange(T))   # padding -> DISTINCT out-of-bounds pages -> dropped
off  = where(slot >= 0, slot % bs, 0)
cache = cache.at[layer, page, :, :, off, :].set(stack([k, v], 2), mode="drop", unique_indices=True)
```

- Never pass a negative index: jnp scatter wraps it even with `mode="drop"` (verified: `x.at[-1]` writes the
  last element).
- The out-of-bounds pad indices are distinct, so `unique_indices=True` is truthful. It is a TPU performance
  hint.
- Fusing the write into the kernel is future work (§9). The head-major layout makes per-token DMAs sub-tile.

#### 5.4.3 Work plan (host, `batch.build_work_plan`, verbatim)

- q-tile `b` covers flat token rows `[b*bq, (b+1)*bq)`, so decode tokens of many sequences share a tile.
- A **pair** is one (tile b, sequence s) overlap. It carries `nkv = pair_num_kv_tiles[p] = cdiv(last_pos + 1,
  bkv)` **kv tiles** (not pages), where `last_pos` is seq s's largest position inside tile b. That gives causal
  kv-tile skipping.
- Pairs are emitted in (s, b) order, which equals (b, s) order.
- `pair_kv_offset` is the exclusive prefix sum of `nkv`: the global kv-tile index of the pair's first tile. It
  gives the double-buffer **parity** `slot = (pair_kv_offset[p] + j) % 2` with no mutable cross-step state.
- `tile_pair_start` is the CSR row pointer.
- Capacity is `NB + S`.

#### 5.4.4 Kernel structure (`pallas_call`)

- **Wrapper layout.** `q_hm = q.reshape(T, Hkv, G, Dc).transpose(1, 0, 2, 3).reshape(Hkv, T*G, Dc)`, so row
  `r` of head block j is token `r // G`, q head `j*G + r % G`. The output uses the inverse transform. XLA fuses
  the transposes into the RoPE producer and the o-proj consumer.
- **Grid.** `grid = (NB,)`, `dimension_semantics=("arbitrary",)`. It is sequential because the prefetched KV
  stream crosses grid steps; v5e/v6e have one TensorCore per chip.
- **Scalar prefetch (SMEM), in order:** `layer[1]`, `cu_q_lens`, `kv_lens`, `page_table`, `pair_seq`,
  `pair_num_kv_tiles`, `pair_kv_offset`, `tile_pair_start`.
- **in_specs:**
  - q_hm: `BlockSpec((Hkv, bq*G, Dc), lambda b, *_: (0, b, 0))`, auto-pipelined into VMEM;
  - kv_cache: `BlockSpec(memory_space=pl.ANY)`, raw HBM ref of the whole `[L, P, Hkv, 2, bs, Dc]`.
- **out_specs:** same spec as q_hm; out_shape `(Hkv, T*G, Dc)`, q dtype.
- **Scratch:**

  | Name | Shape and space | Notes |
  |---|---|---|
  | `kv_buf` | `VMEM[2, ppb, Hkv, 2, bs, Dc]` | cache dtype; page-major double buffer |
  | `sems` | `DMA((2,))` | one per slot |
  | `m`, `l` | `VMEM f32[Hkv, bq*G, 128]` | lane-replicated |
  | `acc` | `VMEM f32[Hkv, bq*G, Dc]` | |

- **DMA.** `make_async_copy(cache_hbm.at[layer, page], kv_buf.at[slot, i], sems.at[slot])`. Both source and
  destination are contiguous `[Hkv, 2, bs, Dc]` slabs; this is the TPU-proven pattern of JAX's
  `paged_attention` v1.
  - Page i of kv tile j of pair p is `page_table[s*MP + j*ppb + i]`.
  - It is DMA'd **only if** `j*ppb + i < cdiv(kv_lens[s], bs)`.
  - `start` and `wait` are wrapped in `pl.when` with the **same predicate**, so there are no dummy fetches and
    the semaphore counts stay balanced.

#### 5.4.5 Kernel algorithm (per grid step b)

```
prologue (b == 0 and total_pairs > 0): start_fetch(pair 0, kv tile 0, slot 0)
m = MASK_VALUE (finite, -0.7*f32max); l = 0; acc = 0           # a q-tile finishes inside its grid step
for p in [tile_pair_start[b], tile_pair_start[b+1]):           # lax.fori_loop, dynamic bounds
    s, nkv, base = pair_seq[p], pair_num_kv_tiles[p], pair_kv_offset[p]
    # full-shape [rows, bkv] int grids (rows = bq*G); never an [N, 1] bool broadcast
    row_tok = b*bq + iota2d((rows, bkv), 0)//G;  row_ok = cu[s] <= row_tok < cu[s+1]
    row_pos = kv_lens[s] - q_len_s + (row_tok - cu[s])
    for j in [0, nkv):                                          # lax.fori_loop
        slot = (base + j) % 2
        next = (p, j+1) if j+1 < nkv else (p+1, 0)
        if next.p < total_pairs: start_fetch(next, 1-slot)      # may prefetch the NEXT grid step's first tile
        wait_fetch(p, j, slot)
        col = j*bkv + iota2d((rows, bkv), 1);  mask = row_ok & (col <= row_pos)      # mask implies col < kv_len
        v_ok = j*bkv + iota2d((bkv, Dc), 0) < kv_lens[s]
        for h in range(Hkv):                                    # static unroll
            k = kv_buf[slot, :, h, 0].reshape(bkv, Dc)          # value reshape; K is NOT masked (see Numerics)
            v = where(v_ok, f32(kv_buf[slot, :, h, 1].reshape(bkv, Dc)), 0).astype(cache dtype)  # f32 select
            s_ = where(mask, dot(q_hm[h], k.T, DEFAULT, f32 acc) * sm_scale, MASK)             # [rows, bkv]
            m_next = maximum(m_prev, rowmax(s_))                # [rows, 128] lane-replicated; [rows,1] f32 broadcast
            p_ = where(mask, exp(s_ - tile(m_next, bkv/128)), 0)                               # [rows, bkv]
            alpha = exp(m_prev - m_next)                                                        # [rows, 128]
            l = alpha*l + rowsum(p_); m = m_next
            acc = tile(alpha, Dc/128)*acc + dot(p_.astype(cache dtype), v, DEFAULT, f32 acc)
o[h] = acc[h] / where(L == 0, 1, L) with L = tile(l[h], Dc/128)   # padding rows (never valid) -> exactly 0
```

**Numerics**

- Kv tile 0 of every pair contains position 0, and `0 <= row_pos` for every valid row. So a valid row sees a
  real key in its first tile, `m` becomes finite, and `exp(MASK - m) == 0`.
- Rows belonging to another sequence (and valid rows whose causal window ends before this tile) are fully
  masked: `rowmax = MASK`, so `m_next = max(m_prev, MASK) = m_prev` (m starts at MASK), `alpha = 1` and
  `p_ = 0`. No separate `row_valid` select on m is needed.
- Unfetched or stale VMEM (NaN under `uninitialized_memory="nan"`): column c of `q @ k.T` depends only on K row
  c, so a NaN K row poisons only its own score column, and `where(mask, ·, MASK)` replaces that column because
  `mask ⊆ (col < kv_len)`. V must be zeroed (`0 * NaN = NaN` in `p_ @ v`); the select runs in f32 (v5e has no
  bf16 VPU), exactly as JAX's bundled `ragged_paged_attention` does.

**Mosaic idioms.** These follow JAX's bundled TPU kernels, not just any pattern that lowers:
- masks from full-shape 2-D `broadcasted_iota`s (the bundled RPA builds `row_ids`/`col_ids` this way, including
  the `// G` for any G);
- no bf16 `where`: V is masked in f32 and cast back;
- m/l kept at `[rows, 128]` and widened with `jnp.tile(x, (1, n // 128))`, as `flash_attention.py` does (it
  raises `NotImplementedError` unless `block_k % 128 == 0`); hence the `bkv % 128 == 0` rule. Widths that are
  not multiples of 128 only occur in interpret mode (`tile` then slice);
- `precision=lax.Precision.DEFAULT` on both matmuls, so a global `jax_default_matmul_precision` (conftest sets
  "highest" on TPU) cannot turn them into `contract_precision<fp32>` or, for "high", a lowering error.

The resulting MLIR was audited (§0.3 item 2).

#### 5.4.6 TPU rules (`ops.tpu_rules`, WP1 verbatim)

The rules and memory estimates live in one pure-Python module, `flash_jax/ops/tpu_rules.py` (Appendix A.14).
`ModelRunner.__init__` runs `check_attention_config(model, cfg, kc, Dc)` over **every bucket** when attention
resolves to `"pallas"`, before the KV pool is allocated; the wrapper re-checks its own call with
`attention_problems(...)` when `interpret is False`. Neither copy can drift: both call the same function.

**Alignment**

- `Dc % 128 == 0`. Checked explicitly: `kv_head_dim_align=64` with the pallas backend gives Dc=64.
- `bs % sublanes(cache) == 0`, i.e. 16 for bf16 and 8 for f32. This keeps the `[ppb, bs, Dc] → [bkv, Dc]`
  value reshape tile-aligned. If Mosaic rejects the bf16 reshape, the fallback is the v1 kernel's pattern:
  upcast to f32, then reshape.
- `bkv % 128 == 0`: bkv is the lane dim of the scores and the contraction dim of `p_ @ v`, and m/l are widened
  with `jnp.tile(·, bkv // 128)`.
- `(bq * G) % sublanes(q) == 0` for the q-tile of **each** bucket (`kc.attn_tile_q(T_pad)`).

**SMEM** at each bucket must be ≤ `SMEM_BUDGET_BYTES` = 512 KiB (half of the 1 MiB on v5e/v6e):

`attention_smem_bytes = 4 * (1 + (S+1) + S + S*MP + 3*(NB+S) + NB + 1)` bytes.

Example: 64 seqs × MP 128 comes to about 34 KiB. With 256 seqs and 8k context at bs=16, raise `bs` to 32 or
more. The error message says so.

**VMEM** must be ≤ `VMEM_BUDGET_BYTES` = 12 MiB:

`attention_vmem_bytes = 2*ppb*Hkv*2*bs*Dc*isz(cache) + 4*Hkv*bq*G*Dc*isz(q)` (q/o blocks, double-buffered)
`+ 2*Hkv*bq*G*128*4` (m, l) `+ Hkv*bq*G*Dc*4` (acc) `+ 1 MiB` (values: scores, probabilities, masks, K/V tiles).

For Llama-3-8B bf16 with bq=32, G=4, ps=16, ppb=8 this is 4.5 MiB. The budget keeps every kernel under the
**default** scoped-VMEM limit (16 MiB on v5e, 32 MiB on v6e) with 4 MiB for Mosaic's internal scratch. v1
never raises the limit: `CompilerParams(vmem_limit_bytes=...)` only works together with
`LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=N` (N KiB > the limit; jax 0.10.2 docstring), so the error
message tells the user to shrink `attn_block_kv` / `attn_block_q` instead.

**Defaults and the decode q-tile.** `attn_block_q=32` (bq*G = 128 MXU rows for G=4) and `attn_block_kv=128`
(ppb = 8 at bs=16). Decode tokens of up to bq different sequences share one q-tile, and each (tile, seq) pair
runs the full `[bq*G, bkv]` QK, exp and PV for every KV head while only G rows are valid: 32× wasted work on a
decode-only step. For Llama-3-8B on v5e with 64 decodes at 1k context that is ~1.1 TFLOP (~5.6 ms) of MXU work
and ~2G exps per step against ~10 ms of KV reads, so decode loses almost all compute headroom, and G=7–8 models
become compute-bound. Therefore the q-tile is **bucket-dependent**: `kc.attn_tile_q(T_pad)` uses
`attn_block_q_small` for buckets `T_pad <= next_pow2(max_num_seqs)`, which is where every decode-only step lands
(`bucket_for(n) = next_pow2(n)` for n ≥ `min_token_bucket`). Its auto
value on `"pallas"` is the smallest power of two with `bq*G` a multiple of the sublane tile: 4 for G=4, 2 for
G=8, 16 for G=1 or G=7 in bf16. The cost is paid by a prefill chunk that lands in a small bucket: its context
is re-streamed once per q-tile (`q_len/bq` times instead of `q_len/32`); this only happens for short
remainders, because a long prefill fills the budget and lands in a large bucket. Set
`attn_block_q_small = attn_block_q` to disable it; `benchmarks/kernels.py` measures both. A dedicated
decode-only kernel flavor stays future work (§9.2 #5).

### 5.5 Fused kernels (WP2) and the complete fusion list

#### 5.5.1 Fused residual-add + RMSNorm, `pallas.rms_norm.add_rms_norm`

- Grid `(T // bt,)`. The row tile comes from the VMEM budget, not "the largest of 256…8": `bt` is the first of
  `T` (if `T <= 1024`; a full dim is always legal) and then the powers of two 1024…8 that divide T and are
  multiples of the sublane tile (8 f32, 16 bf16), whose estimate
  `vmem_bytes(bt, H, isz) = 8*bt*H*isz + 3*bt*H*4` (four `(bt, H)` blocks double-buffered + ~3 f32 temporaries)
  fits `tpu_rules.VMEM_BUDGET_BYTES`. The op is memory-bound, so a small tile costs nothing.
  - Llama-3-8B (H=4096): bt=64 for T ≥ 128 in bf16 and f32 (bt=32 at H=8192); T=16 → 16.
  - The old rule picked bt=256 at T ∈ {256, 512}: 4 operands × 2 buffers × 2 MiB = 16 MiB of blocks alone,
    over v5e's 16 MiB default scoped limit. CPU lowering accepts it (it does not check VMEM), so only this rule
    prevents a first-compile failure on hardware. `input_output_aliases` aliases HBM, not VMEM.
  - If nothing fits, warn and use `ops.reference.add_rms_norm`.
- Specs: `(bt, H)` for x, residual, y and new_residual; `(1, H)` for the weight (the wrapper reshapes it).
- `input_output_aliases={1: 1}`: the residual is updated in place.
- `dimension_semantics=("parallel",)`.
- Body, HF numerics, bit-exact with the reference:

```
r = (f32(x) + f32(res)).astype(dtype); r_out = r
y = (f32(r) * rsqrt(mean(f32(r)^2) + eps)).astype(dtype) * w.astype(dtype)
```

It is used 2L+1 times per step. The first layer passes `residual = zeros`, so the scan body is uniform. The
final norm runs on the S gathered rows only. `rms_norm` for qk-norm stays jnp, and XLA fuses it into the RoPE
chain.

#### 5.5.2 Fused gate_up matmul + SiLU·mul epilogue, `pallas.swiglu.swiglu`

- Grid `(T // bt, I // bi, H // bh)`, `("parallel", "parallel", "arbitrary")`.
- The weight `[H, 2I]` is passed **twice**, with index maps `(k, j)` for gate and `(k, j + I // bi)` for up.
- Two f32 VMEM accumulators, zeroed at `k == 0`. At the last k step: `o = (silu(acc_g) * acc_u).astype(dtype)`.
- Both matmuls pin `precision=lax.Precision.DEFAULT` (§5.4.5).
- The `[T, 2I]` intermediate never touches HBM.

**Tiles** (`swiglu.tiles(T, H, I, dtype)`)

- `bi = I` if `I <= 512`, else the first of 512/256/128 dividing I.
- Then the **largest row tile** first: `bt` runs over `T` (if `T <= 1024`) and the powers of two 1024…8 that
  divide T and are multiples of the sublane tile; for each bt, `bh` runs over `H` (if `H <= 512`) and 512/256/128
  dividing H. The first pair whose `vmem_bytes = 2*(bt*bh + 2*bh*bi + bt*bi)*isz + 4*bt*bi*4` fits
  `VMEM_BUDGET_BYTES` wins. `T % bt == 0` always holds, so every row is written (the old `min(T, 256)` rule left
  rows 256+ of T=384 unwritten: 16384 NaNs, verified).
- Why the largest bt: the grid's T axis is outermost, so the whole `[H, 2I]` weight is streamed `T / bt` times.
  At Llama-3-8B bf16 (H=4096, I=14336) the rule gives `(bt, bh, bi) = (T, 512, 512)` up to T=512 and
  `(1024, 256, 512)` at T=1024 (12 MiB): one weight pass per step up to T=1024, instead of 4 passes (~0.57 ms
  of re-reads on v6e against ~0.26 ms of MXU work) with the old bt=256. The h tile is still re-read `I / bi` = 28
  times (~224 MB at T=1024); `benchmarks/kernels.py` reports both re-read factors.
- In interpret mode any tiling works. Compiled, `bi % 128 == 0` and (`bh % 128 == 0` or `bh == H`) are also
  required. Otherwise, or when no tiling fits, fall back to the reference with a `warnings.warn` (no global
  "warned" flag: the default filter shows it once per call site and message, and `pytest.warns` always sees it).

#### 5.5.3 Complete fusion list

1. **QKV weight fusion.** One matmul per layer, GQA-grouped column order; the q/k/v split is a reshape.
2. **gate_up weight fusion plus the Pallas SiLU·mul epilogue kernel.** No `[T, 2I]` HBM round trip.
3. **Residual-add + RMSNorm in one Pallas pass,** 2L+1 per step, with the residual aliased in place.
4. **KV-cache write adjacent to attention.** It sits in the same layer body as the kernel and is an in-place
   scatter into the donated scan carry. The kernel DMAs pages straight from the whole-cache HBM ref, so no
   per-layer slices are made.
5. **Elementwise chains left to XLA fusion:** RoPE (cos/sin computed once per step, outside the scan), qk-norm,
   bias, head-dim padding and the q kv-head-major transposes.
6. **Logits only for the last token of each sequence.** The gather happens before the final norm and the LM
   head.
7. **The whole step is one jit,** sampling included, with the KV cache donated; only `int32[S_pad]` leaves the
   device.

### 5.6 Sampling (WP5, normative code in Appendix B.4)

- `greedy = argmax(logits)`. If every row has `temperature <= 0`, `lax.cond` returns greedy and skips all
  random work.
- **Keys.** Each row gets `key_i = fold_in(fold_in(key(0), seeds[i]), gen_idx[i])`. A draw depends only on
  (logits, seed, gen_idx), so it is independent of batch position and padding, and a recompute after
  preemption replays it exactly.
- `x = logits / temperature`.
- **Plain temperature rows** use full-vocab Gumbel-max: `argmax(x + gumbel(fold_in(key_i, 0), [V]))`.
- **top_k > 0 or top_p < 1 rows:**
  - `vals, idx = lax.top_k(x, K)` with `K = min(max_candidates, V)`;
  - `keep_k = rank < min(top_k, K)`, or all K when top_k is 0;
  - probabilities: `exp(vals - lse)`, where `lse` is the logsumexp over the kept top-k (renormalised, HF
    order), or over the **full** vocab when top_k is 0, so top-p mass is exact;
  - `keep = keep_k & (cumsum(probs) - probs < top_p)` (rank 0 always kept);
  - choose `idx[argmax(where(keep, vals + gumbel(fold_in(key_i, 1), [K]), -inf))]`.
- **Documented approximation:** a nucleus larger than `sampler_max_candidates` (default 128) is truncated to it:
  the draw is from the top-K candidates by logit, then the nucleus inside them. Tests therefore pin the logits
  scale so the nucleus fits in K, and test the truncation separately (§7 WP5).
- Output: `where(temperature > 0, where(filtered, tok_c, tok_full), greedy)`.

### 5.7 Model forward (WP3, normative code in Appendix B.5)

```
x = embed[input_ids];  cos, sin = rope_cos_sin(positions, rope_inv_freq(cfg))      # once per step
scan over layers, carry (x, res, kv_cache, layer_idx), xs = params["layers"]; init res = zeros_like(x):
    h, res = add_rms_norm(x, res, attn_norm)
    qkv = h @ qkv (+ qkv_bias) -> reshape [T, Hkv, G+2, D];  q = [:, :, :G] -> [T, Hq, D];  k = [:, :, G];  v = [:, :, G+1]
    if qk_norm: q, k = rms_norm(q, q_norm), rms_norm(k, k_norm)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    attn_out, kv_cache = ops.paged_attention(q, k, v, kv_cache, layer_idx, inp.attn, sm_scale=D**-0.5,
                                             block_q=kc.attn_tile_q(T), ...)
    x = attn_out.reshape(T, Hq*D) @ o (+ o_bias)        # attn_out = the kernel output, o = the o-proj weight
    h, res = add_rms_norm(x, res, mlp_norm);  x = swiglu(h, gate_up) @ down
after scan: h, _ = add_rms_norm(x[logits_indices], res[logits_indices], final_norm)  -> (h [S, H], kv_cache)
```

- RoPE uses the HF rotate_half convention in f32. `inv_freq` is computed in float64 numpy, then cast to f32
  (a trace-time constant).
- **llama3 scaling** follows HF `_compute_llama3_parameters` exactly, with
  `(factor, low, high, old) = cfg.rope_scaling`:

  ```
  wl = 2*pi / inv
  scaled = where(wl > old/low, inv/factor, inv)
  smooth = (old/wl - low) / (high - low)
  smoothed = (1 - smooth) * scaled/factor + smooth * scaled
  medium = !(wl < old/high) & !(wl > old/low)
  inv = where(medium, smoothed, scaled)
  ```

### 5.8 Weight loading (WP3)

**Name mapping (HF → params).** Torch Linear weights are `[out, in]`, and the table shows the transforms.

| HF tensor | param | transform |
|---|---|---|
| `model.embed_tokens.weight` [V,H] | `embed` | as is |
| `model.layers.{i}.input_layernorm.weight` | `layers.attn_norm[i]` | as is |
| `…self_attn.q_proj.weight` [Hq·D,H], `k_proj` [Hkv·D,H], `v_proj` [Hkv·D,H] | `layers.qkv[i]` [H, Hkv(G+2)D] | `concat([q.T.reshape(H,Hkv,G,D), k.T.reshape(H,Hkv,1,D), v.T.reshape(H,Hkv,1,D)], axis=2).reshape(H,-1)` |
| `…self_attn.{q,k,v}_proj.bias` | `layers.qkv_bias[i]` | `concat([q.reshape(Hkv,G,D), k.reshape(Hkv,1,D), v.reshape(Hkv,1,D)], 1).reshape(-1)` |
| `…self_attn.o_proj.weight` [H, Hq·D] | `layers.o[i]` | `.T` |
| `…self_attn.o_proj.bias` | `layers.o_bias[i]` | as is (only when `o_bias`) |
| `…self_attn.q_norm.weight`, `k_norm.weight` [D] | `layers.q_norm[i]`, `k_norm[i]` | as is (Qwen3) |
| `…post_attention_layernorm.weight` | `layers.mlp_norm[i]` | as is |
| `…mlp.gate_proj.weight` [I,H], `up_proj` [I,H] | `layers.gate_up[i]` [H, 2I] | `concat([gate.T, up.T], axis=1)` |
| `…mlp.down_proj.weight` [H,I] | `layers.down[i]` | `.T` |
| `model.norm.weight` | `final_norm` | as is |
| `lm_head.weight` [V,H] | `lm_head` [H,V] | `.T`; only when not tied |

- **Ignored names:** `*.rotary_emb.inv_freq`, and `lm_head.weight` when `tie_word_embeddings`.

**`ModelConfig.from_hf` (verbatim)**

| Model | QKV bias | o_bias | qk-norm |
|---|---|---|---|
| Llama | `attention_bias` | `attention_bias` | no |
| Qwen2 | always | never | no |
| Qwen3 | `attention_bias` | `attention_bias` | yes |

- `head_dim` defaults to `H // Hq`.
- `rope_theta` is read from the top level or from `rope_parameters`.
- `rope_scaling` supports `llama3` and `default`; anything else raises `NotImplementedError`.
- Also rejected: `use_sliding_window=True`, `mlp_bias`, and non-silu activations.
- EOS ids are the union of the `config.json` and `generation_config.json` `eos_token_id` values (int or list).

**Reading and memory**

1. The file map comes from `model.safetensors.index.json` (`weight_map`) or a glob of `*.safetensors`.
2. Tensors are read lazily with `safe_open(f, framework="numpy")` (bf16 comes back as `ml_dtypes.bfloat16`),
   keeping one open handle per file.
3. `convert_hf_weights` fills params **one group at a time, one layer at a time**:
   - for each stacked group it allocates the device array `jnp.zeros((L, ...), dtype)`;
   - for each layer it builds the fused, transposed layer slice on the host, casts it with `.astype(np_dtype)`,
     and writes it with a jitted `lax.dynamic_update_index_in_dim(stacked, layer, i, 0)`. That call donates
     `stacked` (in place, verified) and takes `i` as a traced int32, so there is a single compile.

   Peak host RAM is about one layer of one group; for example, one 8B `gate_up` layer at bf16 is about 235 MB.
   Unstacked tensors (`embed`, `final_norm`, `lm_head`) are cast and `device_put` directly.
4. Strict checking: `set(checkpoint names)` is compared with `expected_hf_names(cfg)`.

**Tokenizer (optional).** `load_tokenizer(path)` imports `tokenizers` lazily. If the package or
`tokenizer.json` is missing it returns None; the engine always accepts token ids.

---

## 6. Backend selection and configuration

### 6.1 Kernel backends

Kernel ops are named `"attention"`, `"norm"` and `"mlp"`. Each one resolves **once**, in `ModelRunner.__init__`,
through `resolve_kernel_config(engine_config, q_per_kv=G)`, which also fixes the per-bucket attention q-tiles.
The resolved strings travel as the static `KernelConfig` into the jitted step. There is no global mutable state
and nothing is consulted at trace time.

| requested | resolved |
|---|---|
| `"auto"` | `$FLASH_JAX_KERNEL_BACKEND` if set; else `"pallas"` if `jax.default_backend() == "tpu"`; else `"reference"` |
| `"pallas"` | compiled Pallas TPU kernel (Mosaic); TPU rules asserted |
| `"interpret"` | the same Pallas kernel with `interpret=ops.common.interpret_params()`, which runs on CPU |
| `"reference"` | pure JAX (`flash_jax.ops.reference`) |

- **Precedence.** The per-op value comes from `EngineConfig.kernel_overrides`, for example
  `(("mlp", "reference"),)`; otherwise `kernel_backend` applies.
- **Interpret parameters.** `interpret_params()` is `pltpu.InterpretParams()` with the jax 0.10.2 defaults:
  - `dma_execution_mode="on_wait"`, so a missing `.wait()` shows up as wrong data;
  - `uninitialized_memory="nan"`, so NaN canaries are free;
  - `out_of_bounds_reads="raise"`.

  `FLASH_JAX_DETECT_RACES=1` adds `detect_races=True`. `interpret_params(**overrides)` replaces any field;
  kernel tests pass the result as `interpret=` (for example `dma_execution_mode="eager"`, which executes each
  DMA when it is started and so exposes a write-after-read hazard on the double buffer that `"on_wait"`
  hides, or `detect_races=True`).
- **Forcing interpret mode in tests.** Tests force Pallas-interpret on CPU in one of two ways:
  - explicitly: `EngineConfig(kernel_backend="interpret")`, or calling `ops.pallas.*` with `interpret=True`
    (or an `InterpretParams`);
  - globally, for any config left at `"auto"`: `FLASH_JAX_KERNEL_BACKEND=interpret pytest`. Only `"auto"`
    responds to it, so every test that means "reference" passes `kernel_backend="reference"` explicitly
    (the §8.2 matrix, test_model, test_llm), and backend-resolution tests `monkeypatch.delenv(
    "FLASH_JAX_KERNEL_BACKEND", raising=False)` first.
- **KV head dim.** `kv_head_dim_align=None` means 128 when attention resolves to `"pallas"`, else 1.
- **Attention q-tiles.** `attn_block_q_small=None` means the auto decode tile (§5.4.6) when attention resolves
  to `"pallas"`, else `attn_block_q`; `KernelConfig.attn_tile_q(T_pad)` is the only place the tile is chosen.

### 6.2 EngineConfig flags

All flags are in Appendix A.

| flag | default | meaning |
|---|---|---|
| `max_model_len` | 2048 | max prompt + output tokens per sequence; `MP = cdiv(max_model_len, block_size)` |
| `max_num_seqs` | 64 | max RUNNING sequences; `S_pad = min(T_pad, max_num_seqs)` |
| `max_num_batched_tokens` | 512 | per-step token budget. Use 512 on v5e and 1024 on v6e, from the roofline ridge points (~240 and ~560 FLOP/B) |
| `max_prefill_chunk` | None | optional per-sequence per-step cap (a latency knob) |
| `block_size` | 16 | tokens per page; ≥16 and a multiple of 16 for bf16 on TPU; 32–64 for long contexts (SMEM) |
| `num_kv_blocks` | None | pages per layer; None means sized from device memory (§5.3) |
| `kv_memory_fraction` | 0.85 | fraction of free HBM, after params, given to the KV cache |
| `cpu_kv_cache_bytes` | 256 MiB | KV budget when `memory_stats()` is None (CPU) |
| `enable_prefix_caching` | True | |
| `dtype` | "bfloat16" | params, activations and KV (`"float32"` in tests) |
| `kernel_backend`, `kernel_overrides` | "auto", () | §6.1 |
| `attn_block_q` | 32 | attention q-tile (power of two) for buckets `T_pad > max_num_seqs` |
| `attn_block_q_small` | None | q-tile for buckets `T_pad <= next_pow2(max_num_seqs)`; None = auto on pallas (§5.4.6), else `attn_block_q` |
| `attn_block_kv` | 128 | attention kv-tile in tokens (a multiple of `block_size`; of 128 on TPU) |
| `kv_head_dim_align` | None | §3.2 |
| `min_token_bucket` | 16 | smallest T_pad |
| `sampler_max_candidates` | 128 | top-k/top-p candidate set |
| `seed` | 0 | engine RNG: per-request seeds and random init |
| `debug_checks` | False | run I1–I7 and S1–S4 every step (tests: True) |

Environment variables: `FLASH_JAX_KERNEL_BACKEND` and `FLASH_JAX_DETECT_RACES` (§6.1).

---

## 7. Work packages

There are six packages with **disjoint file ownership**. They are built concurrently from this document.

- **WP1 lands first.** It is mechanical: paste Appendix A, then write `test_core.py`. Every other package
  codes against WP1's files and the signatures in §4.
- **Dependency graph** (by interface only): WP2 → WP1; WP3 → WP1; WP4 → WP1; WP5 → WP1, WP3, WP4;
  WP6 → everything.
- **Standalone tests.** WP1–WP4 tests run standalone once WP1 exists. The *(integration)* tests of WP5 and WP6
  need WP3 and/or WP4 merged. They carry **no marker** (A.1 registers only `slow` and `tpu`); instead each such
  test module, or test, starts with `pytest.importorskip("flash_jax.model.llama")` and/or
  `pytest.importorskip("flash_jax.engine.scheduler")`, so it skips cleanly until its dependency lands.
- **Running:** from the repo root, with no install (§2): `pytest`, `python -m benchmarks.throughput …`.
- **Code style:**
  - small modules, type hints, docstrings where they add value;
  - no new runtime dependencies (`tokenizers` stays optional and lazily imported);
  - no global mutable state, except `model_runner.TRACE_COUNTER`.

### WP1: Core contracts

- **Creates:** `pyproject.toml`, `flash_jax/__init__.py`, `flash_jax/config.py`, `flash_jax/sampling_params.py`,
  `flash_jax/batch.py`, `flash_jax/engine/__init__.py`, `flash_jax/engine/sequence.py`,
  `flash_jax/model/__init__.py`, `flash_jax/ops/__init__.py`, `flash_jax/ops/common.py`,
  `flash_jax/ops/reference.py`, `flash_jax/ops/tpu_rules.py`, `flash_jax/ops/pallas/__init__.py`,
  `flash_jax/testing/__init__.py`, `flash_jax/testing/tiny_models.py`, `flash_jax/testing/numpy_reference.py`,
  `tests/conftest.py`: **all verbatim from Appendix A**. Plus `tests/test_core.py`.
- **Check:** `pytest tests/test_core.py` from the repo root works without an install (`pythonpath` in A.1).
- **Depends on:** nothing.
- **Acceptance** (`tests/test_core.py`, under 60 s):
  1. **`ModelConfig.from_hf`:**
     - every `TINY_HF_CONFIGS` entry gives the expected fields: llama3 scaling tuple `(8.0, 1.0, 4.0, 64)`;
       qwen2 `qkv_bias`, tied, `o_bias=False`; qwen3 `qk_norm` with `head_dim=32`; EOS union with a
       generation config;
     - the `rope_parameters` form works, and `head_dim: None` falls back to H/Hq;
     - `NotImplementedError` for `model_type="mistral"`, `rope_type="yarn"`, `use_sliding_window=True` and
       `hidden_act="gelu"`.
  2. **`EngineConfig`:**
     - validation errors for each rule (including a non-power-of-two `attn_block_q_small`);
     - `token_buckets()` is `(16, …, 512)` by default and `(16,)` for budget 7;
     - `bucket_for` edges, including a `ValueError` above the maximum;
     - `num_seq_slots` and `pages_per_seq`.
  3. **Backend resolution** (every test first calls `monkeypatch.delenv("FLASH_JAX_KERNEL_BACKEND",
     raising=False)`, so a suite run under the env override still passes):
     - `resolve_backend("auto", "cpu") == "reference"` and `resolve_backend("auto", "tpu") == "pallas"`;
     - the env var (via `monkeypatch.setenv`) replaces `"auto"`, and a bad value raises;
     - overrides apply per op;
     - `kv_head_dim` gives 128 for pallas, D for reference, and follows an explicit align;
     - `KernelConfig.attn_tile_q`: the auto small tile on `"tpu"` is 4/2/16/16 for G = 4/8/1/7 in bf16 (2 for
       G=4 in f32), used for `T_pad <= next_pow2(max_num_seqs)` only; on `"cpu"` it equals `attn_block_q`; an explicit
       `attn_block_q_small` wins; the tile never exceeds T_pad.
  4. **`build_work_plan` / `prepare_step_input`:**
     - the §3.4 worked example exactly, through `build_attention_metadata` and through `prepare_step_input` on
       hand-built Sequences with preset block tables (config and batch as in §3.4), including every padding
       value of every StepInput field and `prepare_step_input([], cfg, kc, t_pad=16)` (num_seqs 0, all slots -1);
     - 200 random ragged batches against a brute-force enumeration: the set of (tile, seq) overlaps, `nkv`,
       offsets, CSR, capacity never exceeded, trailing zeros.
  5. **`build_attention_metadata`:** the §3.4 example exactly; `AssertionError` for `q_len = 0`, `kv_len < q_len`,
     or a block table too short.
  6. **`reference.write_kv`:**
     - pads are dropped and the last page/offset of the last layer is unchanged;
     - real rows are written bit-exactly and everything else is unchanged;
     - the result equals a numpy loop.
  7. **`reference.paged_attention`:**
     - matches JAX's `ref_ragged_paged_attention` (cache converted with
       `jnp.moveaxis(cache[l], 3, 1).reshape(P, bs, 2*Hkv, Dc)` and the page table reshaped to `[S, MP]`) on
       random ragged batches, f32 atol 1e-5, including T=48 (C = gcd(48, 32) = 16);
     - `max_chunk_bytes=1` (forces C=1) agrees with the default within 1e-6;
     - pad rows are exactly 0;
     - NaN-filled unused pages give finite outputs.
  8. **`ops.paged_attention`:** `Dc > D` (zero-padded cache) gives the same `o` as `Dc == D`, and the cache pad
     lanes stay 0.
  9. **Norm and MLP references:** `ops.reference.add_rms_norm`, `rms_norm` and `swiglu` match numpy formulas
     (f32 1e-6).
  9b. **`tpu_rules`:** `attention_vmem_bytes` is 4.5 MiB and `attention_problems` is empty at Llama-3-8B bf16
     defaults (Hkv=8, G=4, Dc=128, bs=16, ppb=8, bq=32, S=64, MP=128); each rule fires alone (Dc=64, bs=8 bf16,
     bkv=64, bq*G=8 bf16, SMEM at S=256/MP=512/bs=16, VMEM at ppb=64); `check_attention_config` lists every
     violation over all buckets in one `ValueError`. `interpret_params(dma_execution_mode="eager")` overrides.
  10. **`numpy_reference`:**
      - causality: logits of a prefix are unchanged by appending tokens;
      - `greedy_generate` stops at EOS;
      - `check_greedy` raises on a flipped token with a large gap and counts a near-tie (craft it by
        monkeypatching `dense_logits`).
  11. **`tiny_models`:**
      - `write_hf_checkpoint` round-trips 1 shard f32 and 3 shards bf16 (+index, +generation_config), and
        `ModelConfig.from_pretrained` agrees;
      - **diversity guard:** for every tiny config and seed 0, greedy continuations (20 tokens) of 4 random
        prompts average ≥ 5 distinct tokens.

### WP2: Pallas kernels

- **Creates:** `flash_jax/ops/pallas/ragged_paged_attention.py`, `flash_jax/ops/pallas/rms_norm.py`,
  `flash_jax/ops/pallas/swiglu.py`, `flash_jax/testing/tpu_lowering.py`, `tests/test_pallas_attention.py`,
  `tests/test_pallas_norm_mlp.py`, `tests/test_tpu_lowering.py`.
- **Depends on** (WP1 only): `batch.AttentionMetadata`, `batch.build_attention_metadata`, `ops.reference.*`,
  `ops.common.{MASK_VALUE, interpret_params}`, `ops.tpu_rules.{attention_problems, VMEM_BUDGET_BYTES,
  sublanes, itemsize}`.
- **Implement:** §4.2, §5.4.3–5.4.6 and §5.5, starting from Appendix B.3 and B.6. Keep the Mosaic idioms of
  §5.4.5 (full-shape iota masks, f32 selects, `[rows, 128]` m/l widened with `jnp.tile`, `precision=DEFAULT`).
- **Test inputs.** Any T works for the reference (C = gcd(T, 32)); kernel cases need `T % bq == 0`.
- **Acceptance** (under 3.5 min in total):
  - **`test_pallas_attention.py`**, all in interpret mode against `ops.reference.paged_attention` on a cache
    already written by `reference.write_kv`:
    - **Random ragged batches**, about 8 cases (G, Hkv, D, bs, bq, ppb, T, S, dtype), **each run with
      `interpret=interpret_params(dma_execution_mode=m)` for m ∈ {"on_wait", "eager"}**. Together they cover:
      - G ∈ {1,2,4,8}, bs ∈ {4,8,16}, bq ∈ {4,8,16,32}, ppb ∈ {1,2,4} and D ∈ {16,64,128};
      - f32 at atol 1e-4, and bf16 at atol 3e-2 against the reference on the same bf16 inputs.

      The generator mixes decodes, chunks with context and fresh prefills, with random page permutations. It
      fills **unused pages with NaN**.
    - **The exact TPU default configuration** in interpret mode: bf16, bs=16, ppb=8 (bkv=128), bq=32, Dc=128,
      G=4, Hkv=8, contexts up to ~300 tokens (several kv tiles); plus bkv=Dc=256 (exercises the `jnp.tile`
      widening) in f32 and bf16.
    - **Race detection:** one random case with `interpret_params(detect_races=True)`, then assert
      `jax._src.pallas.mosaic.interpret.interpret_pallas_call.races.races_found is False` (skip if the private
      attribute is missing).
    - **Edge cases** (on_wait and eager): a single decode with `kv_len=1`; `kv_len == MP*bs`; one sequence
      spanning ≥3 q-tiles; ≥3 sequences in one q-tile; a sequence ending exactly on a tile boundary;
      `num_seqs == S`; `num_seqs == 0` (all-zero output); fully padded tiles.
    - **Invariance:**
      - pad output rows are exactly 0, and all outputs are finite;
      - changing q rows `t >= N` (including to NaN) leaves real rows bitwise unchanged;
      - two runs are bitwise identical.
    - **Inside `lax.scan`:** over 2 layers with a traced layer index, the result equals per-layer calls.
    - **Wrapper rules:** with `interpret=False`, bs=4 bf16, Dc=64, bkv=64 and bq*G=8 bf16 each raise
      `ValueError` naming the rule (no Mosaic needed: the check runs first).
    - **`@pytest.mark.tpu`:** compiled Mosaic against the reference, bf16, D=128, bs=16, G=4, Hkv=8,
      T ∈ {16, 256}, with bq ∈ {32, 4} (the default and the decode tile), tolerance 3e-2.
  - **`test_pallas_norm_mlp.py`:**
    - `add_rms_norm` interpret vs reference **bit-exact** in f32 for T ∈ {5,16,64,512}, H ∈ {64,256}; bf16
      within 1 ulp-level tolerance (2e-2); still correct under `jax.jit(..., donate_argnums=1)`;
    - `row_tile(T=512, H=4096)` is 64 for bf16 and f32, and `vmem_bytes` of the chosen tile is within
      `VMEM_BUDGET_BYTES` for T ∈ {16, 256, 512, 1024}, H ∈ {4096, 8192};
    - `swiglu` for (T,H,I) ∈ {(16,64,128), (64,64,96), (5,64,128), (512,256,1024), **(384,64,256)**} at f32
      atol 1e-5, plus bf16, all outputs finite (T=384 must write every row);
    - `tiles(1024, 4096, 14336, bf16) == (1024, 256, 512)` and `tiles(512, 4096, 14336, bf16) == (512, 512, 512)`;
    - the fallback path: `interpret=False` on CPU with I=96 warns (`pytest.warns`; no global flag, so the test
      does not depend on test order) and returns the reference result;
    - `tpu`-marked compiled variants: `add_rms_norm` at (T=512, H=4096) in bf16 and f32, `swiglu` at
      (T ∈ {16, 512, 1024}, H=4096, I=14336) in bf16 (tolerance 3e-2: the kernel pins `precision=DEFAULT`).
  - **`test_tpu_lowering.py`** is skipped unless `tpu_lowering_available()`. For chip ∈ {v5e, v6e} it lowers
    (asserting `pltpu.get_tpu_info().chip_version.value == chip` inside `lower_for_tpu`):
    - attention at bf16 Llama-3-8B shapes (T=512, Hq=32, Hkv=8, D=128, bs=16, ppb=8, bq=32, S=64, MP=128),
      with the decode tile (T=64, bq=4), at G=7 (Hq=14, Hkv=2), and at f32 G=1;
    - attention again under `with jax.default_matmul_precision("highest")`, asserting that the Mosaic text
      contains no `contract_precision<fp32>` (capture it with `pallas_call(debug=True)`, or skip that assert);
    - `add_rms_norm` at (T ∈ {16, 512}, H=4096, bf16 and f32);
    - `swiglu` at (T ∈ {16, 512, 1024}, H=4096, I=14336).

    It asserts `"tpu_custom_call"` in `lowered.as_text()`. It also lowers one hand-written `pallas_call` with a
    misaligned BlockSpec ((8, 64) blocks of a (16, 256) f32 array) and expects `ValueError` **from lowering**,
    next to the wrapper-level cases above; everything else that is misaligned (scratch, DMA slabs, value
    reshapes, VMEM) is not visible to lowering (§0.3). Runtime is about 10 s.

### WP3: Model and weights

- **Creates:** `flash_jax/model/rope.py`, `flash_jax/model/llama.py`, `flash_jax/model/loader.py`,
  `tests/test_rope.py`, `tests/test_model.py`, `tests/test_loader.py`.
- **Depends on** (WP1 only): `config.{ModelConfig, EngineConfig, KernelConfig, resolve_kernel_config}`,
  `batch.{StepInput, ScheduledSeq, prepare_step_input}`, `engine.sequence.Sequence`,
  `ops.{paged_attention, add_rms_norm, rms_norm, swiglu}` (the reference backend is enough), and
  `testing.{tiny_models, numpy_reference}`.
- **Implement:** §3.3, §4.3, §5.7 and §5.8, starting from Appendix B.5. `forward` passes
  `block_q=kc.attn_tile_q(T)` to `ops.paged_attention`.
- **Test harness.** Tests never hand-roll token layouts: they build `Sequence`s with preset contiguous block
  tables and `num_computed_tokens`, wrap them in `ScheduledSeq(seq, start, n, samples)`, and call
  `prepare_step_input(batch, ecfg, kc)` (WP1, the same builder the runner uses). So T_pad is always
  `ecfg.bucket_for(N)` (a power of two), the work plan uses `kc.attn_tile_q(T_pad)`, and the sampling fields are
  filled. `ecfg = EngineConfig(dtype="float32", kernel_backend="reference", …)` and
  `kc = resolve_kernel_config(ecfg, q_per_kv=G)` (or a `KernelConfig(attention="interpret", …)` for the smoke
  test). The cache is `jnp.zeros((L, P, Hkv, 2, bs, Dc))`.
- **Acceptance** (under 90 s):
  - **`test_rope.py`:**
    - position 0 is the identity;
    - `apply_rope(q, m)·apply_rope(k, n)` depends only on m−n;
    - `rope_inv_freq` for the `llama3` tiny config equals an independent float64 transcription of the HF
      formula, and the test checks that all three frequency bands occur;
    - `apply_rope` matches the numpy rotate_half formula.
  - **`test_model.py`**, parametrized over `tiny_name` (f32, reference backends):
    - one full-prompt step gives `compute_logits` equal to `dense_logits(...)[-1]` (atol 1e-4);
    - a ragged step with 3 prompts matches each prompt's last-row oracle logits;
    - a prompt split into 2 chunks over 2 `forward` calls sharing the cache gives chunk-2 logits equal to the
      oracle;
    - prefill followed by 2 decode steps matches the oracle at each position;
    - `kv_head_dim_align`-style padding (Dc=32 for D=16) gives identical logits;
    - `param_shapes` equals the tree and shapes of `convert_hf_weights`, and `init_params` matches
      `param_shapes`;
    - a `KernelConfig(attention="interpret", norm="interpret", mlp="interpret")` smoke test on the `llama`
      tiny config, one step, is marked `slow` and becomes runnable once WP2 lands (skip if
      `flash_jax.ops.pallas.ragged_paged_attention` is not importable).
  - **`test_loader.py`:**
    - `write_hf_checkpoint` with 1 shard f32 and with 3 shards bf16 + `generation_config`, then
      `load_hf_checkpoint`: the config equals `from_hf`, and the params equal `convert_hf_weights(w, cfg, dtype)`
      with the weights dict `w` (bf16: equal after the same cast); `convert_hf_weights(w.__getitem__, …)` gives
      the same tree;
    - forward logits vs the oracle (f32 1e-4; bf16 loose 5e-2 relative on logits);
    - a missing tensor raises `KeyError`, and an unexpected tensor raises `ValueError`;
    - `*.rotary_emb.inv_freq` is ignored, and a tied checkpoint carrying a stray `lm_head.weight` loads;
    - **fused layouts:** `h @ qkv` reshaped `[T, Hkv, G+2, D]` equals the separate q/k/v projections, head by
      head, and `h @ gate_up` equals `[h@gate.T | h@up.T]`;
    - `load_tokenizer(tmpdir)` is None without `tokenizer.json`.

### WP4: Block manager and scheduler

- **Creates:** `flash_jax/engine/block_manager.py`, `flash_jax/engine/scheduler.py`,
  `flash_jax/testing/fake_runner.py`, `tests/test_block_manager.py`, `tests/test_scheduler.py`.
- **Depends on** (WP1 only): `engine.sequence.{Sequence, SeqStatus}`, `batch.ScheduledSeq`,
  `config.{EngineConfig, cdiv}`. No JAX anywhere in this package.
- **Implement:** §4.4, §5.1 and §5.2. Appendix B.1 and B.2 are normative, and B.7 is the FakeRunner.
- **Acceptance** (under 60 s):
  - **`test_block_manager.py`**, targeted cases:
    - ref counts under shared prefixes;
    - **LRU order:** freeing a seq whose table `[a, b, c]` is fully sealed pops c, then b, then a; unhashed
      blocks pop before any hashed block;
    - revival of a cached free block (removed from the free list, ref 1);
    - **touch-before-pop:** the hit block sits at the LRU front and the pool is exactly full; `allocate`
      succeeds and never hands the hit out as a fresh block;
    - eviction unmaps the hash, and `num_evictions` counts it;
    - chaining: same tokens after different parents give different hashes;
    - the last-token cap for prompt lengths kB−1, kB and kB+1;
    - duplicate sealing leaves the second block unhashed;
    - `can_allocate` counts revived blocks;
    - caching off gives no hits and no hashes.

    **Property test:** 10k random operations (admit, grow, seal, free, preempt) on 16 blocks, with
    `check_invariants` after each.
  - **`test_scheduler.py`**, with the FakeRunner and `check_batch` + `check_invariants` after every step:
    - budget respected;
    - **decode priority:** a prompt of 3× the budget arriving while 3 decodes run; every step schedules all 3
      decodes;
    - chunk continuity (`start_pos` progression);
    - `max_num_seqs`, and `max_prefill_chunk`;
    - **preemption:** the victim is the latest-arrived unscheduled seq (possibly the requester, even while a
      newer decode is scheduled in the same step: not strict priority, §5.2); `waiting` stays sorted; nothing
      is admitted in a preempting step;
    - `update` with fewer tokens than `len(batch)` raises `ValueError` (`zip(..., strict=True)`);
    - stop conditions: EOS, `ignore_eos`, `stop_token_ids`, `max_tokens`, `max_model_len`; EOS included in the
      output;
    - `abort` of a waiting and of a running seq;
    - `EngineDeadlock` for a pool smaller than `pages_per_seq`;
    - **stress:** at least 500 randomized workloads, as in the design prototype:
      - staggered arrivals, exact duplicate prompts, shared prefixes;
      - bs ∈ {1,2,4,8}, budgets 1–40, `max_num_seqs` 1–5, chunk caps {None, 3, 7};
      - pools from `pages_per_seq` to 3× that.

      Every output equals `FakeRunner.reference_output`, and the pool is fully free at the end. Summed over the
      stress set, preemptions > 0 and prefix-hit tokens > 0.

### WP5: Sampler, model runner and LLM API

- **Creates:** `flash_jax/ops/sampler.py`, `flash_jax/engine/model_runner.py`, `flash_jax/engine/llm.py`,
  `tests/test_sampler.py`, `tests/test_model_runner.py`, `tests/test_llm.py`.
- **Depends on:**
  - WP1: config (incl. `resolve_kernel_config(..., q_per_kv=)`), batch (incl. `prepare_step_input`), Sequence,
    `ops`, `ops.tpu_rules.check_attention_config`;
  - WP3: `llama.forward`, `llama.compute_logits`, `llama.init_params`, `llama.param_shapes`,
    `loader.load_hf_checkpoint`, `loader.load_tokenizer`, `Tokenizer`;
  - WP4: `Scheduler`, `EngineDeadlock`.
- **Implement:** §4.5, §5.3 and §5.6, starting from Appendix B.4.
- **Acceptance** (under 90 s):
  - **`test_sampler.py`** (standalone). Inputs are pinned: `logits = 3.0 * rng.standard_normal(50)`
    (`np.random.default_rng(1)`), V=50, `max_candidates=16`; at this scale the 0.8 nucleus has ≤ 12 tokens, so
    K-truncation never bites (with unit-scale logits it exceeds 16 in 99.5% of draws and TV reaches 0.2,
    §0.3 item 7). The expected distribution applies top-k (renormalised), then top-p with the full-vocab mass
    when top_k is 0.
    - greedy equals argmax, and exact ties resolve to the lowest index;
    - `top_k=1` equals argmax;
    - TV distance < 0.03 at 20k samples for (T=1), (0.7, k=5), (1.0, p=0.8) and (1.3, k=10, p=0.9);
    - **documented truncation:** logits `0.3 * rng.standard_normal(50)`, (1.0, p=0.9), whose nucleus has ~43
      tokens: TV < 0.03 against the K-truncated expectation (top-16 by logit, then the nucleus inside it) and
      no sample outside the top 16;
    - samples never fall outside the top-k set or the nucleus;
    - mixed per-row params in one batch;
    - batch-permutation invariance;
    - same (seed, gen_idx) gives the same token; a different `gen_idx` changes the draw distribution.
  - **`test_model_runner.py`** *(integration: `pytest.importorskip("flash_jax.model.llama")`)*, on the tiny
    llama with `kernel_backend="reference"` (the §3.4 `prepare_step_input` example is WP1's test now):
    - `run()` returns `len(batch)` tokens; `params`, `kv_cache` and the device_put StepInput are all committed
      to `runner.device`;
    - the donated old cache `is_deleted()`;
    - **compile bound:** a varied workload touching every bucket increases `step_fn._cache_size()` (compiles,
      not traces) by at most `len(token_buckets())`, and `TRACE_COUNTER["step_fn"]` by the same amount (use a
      unique `sampler_max_candidates` so no other test shares the executable);
    - **architecture-only key:** a second runner whose ModelConfig differs only in `eos_token_ids` and
      `max_position_embeddings` adds 0 executables;
    - `dummy_input(t)` has `num_seqs == 0`; `lower(max_bucket).compile().memory_analysis()` works and does not
      touch `kv_cache`;
    - `warmup()` leaves the cache all zeros;
    - the CPU `num_kv_blocks` fallback equals `cpu_kv_cache_bytes // bytes_per_block`;
    - `ValueError` when `num_kv_blocks < pages_per_seq`; with `kernel_backend="pallas"` on CPU and
      `kv_head_dim_align=64`, `__init__` raises the `tpu_rules` `ValueError` before allocating the cache.
  - **`test_llm.py`** *(integration: importorskip both)*, with `debug_checks=True` and
    `kernel_backend="reference"`:
    - `generate` returns outputs in input order with all fields; `stats` keys are present, `prompt_tokens` and
      `generated_tokens` match their §4.5 definitions, `last_step` is filled;
    - a `str` prompt without a tokenizer raises `ValueError`; so do a token id ≥ `vocab_size` or < 0, before
      the scheduler changes;
    - `SamplingParams(seed=2**40)` is accepted (reduced mod 2**31) and equals `seed=2**40 % 2**31`;
    - a fake tokenizer object (encode/decode) round-trips text;
    - an overlong prompt raises `ValueError`;
    - `abort` of a waiting and of a running request: `abort()` returns True, the next `step()` returns its
      `RequestOutput` with `finish_reason == "abort"`, and `generate` of other requests is unaffected;
    - a streaming `add_request` / `step` loop; `step()` with nothing pending returns `[]` without running the
      model (`stats["steps"]` unchanged);
    - `LLM(model=<synthetic checkpoint dir>)` loads and generates;
    - greedy output of 2 prompts equals `greedy_generate` of the oracle.

### WP6: End-to-end tests, benchmarks and README

- **Creates:** `tests/test_e2e.py`, `benchmarks/__init__.py` (empty), `benchmarks/throughput.py`,
  `benchmarks/kernels.py`, and edits `README.md`.
- **Depends on:** the public API only: `LLM` (incl. `stats`, `last_step`, `runner.lower`), `EngineConfig`,
  `SamplingParams`, `ModelConfig` (incl. `ModelConfig.from_hf`), `convert_hf_weights`, `load_hf_checkpoint`,
  `init_params`, `ops.*`, `testing.*`, and `batch.prepare_step_input` (kernel benchmark inputs). Engines are
  built with the §8.2 recipe, never from WP3/WP5 internals.
- **Acceptance:**
  - `test_e2e.py` (§8.2, `pytest.importorskip` of the model and scheduler modules): the reference matrix in
    under 60 s, and the `slow` interpret subset in under 90 s.
  - `python -m benchmarks.throughput --preset tiny --num-requests 8 --input-len 32 --output-len 16` finishes
    on CPU in under 60 s and prints the report (run from the repo root; no install needed, §2).
  - `python -m benchmarks.kernels --backend reference` runs on CPU.
  - README: install (optional `uv pip install -e .`), quickstart (`LLM.generate` on token ids and on an HF
    checkpoint dir), testing commands (`pytest`, `pytest -m "not slow"`, `pytest -m tpu` on a TPU VM), and
    benchmark commands for TPU (`python -m benchmarks.…`).

**`benchmarks/throughput.py`**

- **Flags:**
  - model: `--model PATH` | `--preset {tiny, llama3-1b, llama3-3b, llama3-8b, qwen3-0.6b}`. Presets are
    ModelConfigs with random weights via `init_params`, since HF is unreachable; llama3-8b needs v6e memory.
  - workload: `--num-requests`, `--input-len`, `--output-len` (an int or `a:b` for a uniform range),
    `--prefix-len` (a shared system prompt);
  - engine: `--max-num-batched-tokens`, `--max-num-seqs`, `--max-model-len`, `--block-size`,
    `--num-kv-blocks`, `--dtype`, `--backend`, `--kernel-overrides mlp=reference,…`, `--no-prefix-caching`,
    `--attn-block-q-small`;
  - run: `--seed`, `--profile DIR` (wraps the timed run in `jax.profiler.trace`), `--json`.
- **Report** (every number from the public API):
  - warmup/compile seconds (separately; `LLM.warmup()`);
  - wall time, req/s, output tok/s, total tok/s (`stats["generated_tokens"]`, `stats["prompt_tokens"]`);
  - mean step time for mixed steps and for decode-only steps: the loop calls `step()` itself and reads
    `llm.last_step` after each call (`seconds`; decode-only iff `num_decodes == num_seqs`);
  - prefix hit rate, preemptions, `num_traces` and `num_compiles` (they differ only if something recompiles
    without retracing, §5.3);
  - `llm.runner.lower(max_bucket).compile().memory_analysis()` of the largest bucket (shares the executable, no
    extra compile): alias bytes vs cache bytes, and temp bytes (R3).

**`benchmarks/kernels.py`** times, for each backend: the attention op on decode-only and mixed batches built
with `prepare_step_input` (reporting achieved KV GB/s), for `attn_block_q_small` ∈ {auto, `attn_block_q`}
(§5.4.6); `add_rms_norm`; and `swiglu` against the reference, printing the chosen `(bt, bh, bi)` and the weight
and h re-read factors `T/bt` and `I/bi`.

---

## 8. Test plan

### 8.1 Principles

- **Oracles, from most to least independent:**
  1. `testing.numpy_reference`: float64 NumPy on raw HF-named weights, parsing the HF dict itself.
  2. JAX's bundled `ref_ragged_paged_attention` (attention only).
  3. `flash_jax.ops.reference`, the pure-JAX twin of every kernel.
  4. The Pallas kernels in interpret mode.

  Each level is tested against the one above it.
- **Host logic runs without JAX.** It is tested through the FakeRunner's simulated KV store, which verifies the
  whole context of every chunk.
- **Invariants run after every step in every engine test** (`debug_checks=True`).
- **Tiny models are sharp:** logit std is about 4, so top-2 margins sit far above f32 noise. They are also
  diverse (the WP1 diversity guard), and float32.
- **Interpret mode is expensive** (about 0.15 s per attention call). The broad e2e matrix therefore runs on the
  reference backend. The kernels get dense unit tests plus two full e2e runs in interpret mode, marked `slow`.
- **Compilation is shared.** The module-level `step_fn` jit is shared by engines with identical static
  configs, which keeps the matrix cheap.

### 8.2 End-to-end equivalence (`tests/test_e2e.py`, WP6)

**Building an engine (the canonical recipe; every e2e test uses it).**

```python
hf, w = tiny_model(name)                                   # flash_jax.testing.tiny_models
cfg = ModelConfig.from_hf(hf)                              # or dataclasses.replace(cfg, eos_token_ids=(t,))
params = convert_hf_weights(w, cfg, "float32")             # a Mapping works; so does w.__getitem__
llm = LLM(model_config=cfg, params=params,
          engine_config=EngineConfig(dtype="float32", kernel_backend="reference", debug_checks=True, ...))
```

The oracle side uses the same `(hf, w)`: `check_greedy(out, w, hf, prompt)`, `greedy_generate(w, hf, prompt, n)`.

**Workload W.** Deterministic with `np.random.default_rng(0)`; token ids drawn from `[3, V)`.

- **Wave 1, submitted together:** prompts of lengths 3, 40, 17, 9, and two prompts made of a shared 16-token
  prefix plus 3 and 2 distinct tokens. `max_tokens` are 20, 6, 12, 4, 15 and 10.
- **Wave 2, after wave 1 finishes:** an exact duplicate of the 40-token prompt, and the shared prefix with a new
  5-token suffix, with `max_tokens` 8 and 8. Wave 2 guarantees prefix-cache hits when caching is on.
- **Sampling:** `temperature=0`, `ignore_eos=True`.
- **Model length:** `max_model_len=128`, except 64 for the tight-pool rows 2 and 5. There
  `num_kv_blocks == pages_per_seq`, the smallest legal pool.

**Matrix.** `kernel_backend="reference"` **passed explicitly** (so `FLASH_JAX_KERNEL_BACKEND` cannot switch the
matrix to interpret mode, §6.1), f32, `debug_checks=True`. Each row runs with prefix caching **off and on**, for
12 runs in total.

| # | model | block_size | budget | num_kv_blocks | max_num_seqs | max_prefill_chunk | expectation |
|---|---|---|---|---|---|---|---|
| 1 | llama | 4 | 16 | 64 | 8 | None | chunking |
| 2 | llama | 8 | 7 | 8 | 8 | None | **preemption** (tight pool), budget not a multiple of bs |
| 3 | qwen2 | 8 | 32 | 24 | 4 | None | bias, tied, G=4 |
| 4 | qwen3 | 16 | 64 | 64 | 8 | 5 | qk-norm, explicit head_dim, chunk cap |
| 5 | llama3 | 4 | 8 | 16 | 4 | None | **preemption**, rope scaling |
| 6 | llama3 | 16 | 256 | 64 | 2 | None | large bucket, seq cap |

**Assertions for each run:**

1. For every request, `check_greedy(engine_out, w, hf, prompt)` passes: no gap above 1e-3.
2. Total near-ties are at most 1% of generated tokens.
3. For every request whose oracle `greedy_generate` margins are all > 1e-3, the engine output **equals the oracle
   exactly** (the literal requirement).
4. `len(output) == max_tokens` and `finish_reason == "length"`.
5. After the run, every block is free. `debug_checks` guarantees I1–I7 and S1–S4 held every step.
6. Rows 2 and 5 have `stats["num_preemptions"] > 0`, so the test is not vacuous.
7. With caching on, `prefix_hit_tokens > 0`. With it off, `== 0`.
8. The outputs of the caching-off and caching-on runs are identical **for every request whose oracle margins are
   all > 1e-3** (the same filter as assertion 3). Caching on/off, like preemption, computes the same
   positions' K/V in different steps and buckets, so logits may differ at f32-rounding level; the remaining
   requests are counted and bounded by assertion 2. (The current workload has 0 near-ties, so today this
   filters nothing; it keeps the test honest when the workload, weight seed or bucket set changes.)

**Additional e2e tests:**

- **Stop conditions.** On the `llama` model with `ignore_eos=False`, set EOS through a `ModelConfig` copy
  (`dataclasses.replace(cfg, eos_token_ids=(t,))`), where t is the oracle's 3rd generated token for some prompt.
  Generation must stop right after the first occurrence, with `finish_reason == "stop"` and t included. Repeat
  the same with `stop_token_ids`.
- **Seeded sampling determinism.** `temperature=0.8`, `top_k=20`, `top_p=0.9`, fixed seeds. Run under row 1's
  config and under row 2's tight-pool config (with preemption). The outputs must be identical, because draws
  depend only on (logits, seed, gen_idx). Exact equality also relies on the logits agreeing to within f32
  rounding across bucket shapes: a flip needs two Gumbel-perturbed candidates (or a top-p cumsum and `top_p`)
  within ~1e-6, about 1e-6 per token, and on CPU the run is deterministic, so a failure is reproducible, not
  flaky. If a workload change ever makes it fail, triage the diverging position's perturbed-score gap before
  suspecting the engine.
- **Pallas interpret, `@pytest.mark.slow`:**
  - (a) `llama`, bs=8, budget 16, `num_kv_blocks=8`, `max_model_len=64` (preemption), `attn_block_kv=16`,
    caching on, `kernel_backend="interpret"` for all three ops;
  - (b) `qwen3`, bs=8, budget 32, `kv_head_dim_align=64` (padded cache head dim), caching on,
    `kernel_backend="interpret"`.

  Same assertions as above; (a) also asserts preemptions > 0. Each takes about 20 s with both waves (measured
  with the revised kernels: (a) 21 s with 4 preemptions, (b) 17 s).

### 8.3 How tests force Pallas-interpret

- Kernel unit tests call `ops.pallas.*(…, interpret=True)` directly, or pass
  `interpret=interpret_params(dma_execution_mode="eager")` / `interpret_params(detect_races=True)`.
- Model and e2e tests use `EngineConfig(kernel_backend="interpret")` or a `KernelConfig(...)` with
  `"interpret"` strings.
- `FLASH_JAX_KERNEL_BACKEND=interpret` flips every `"auto"` config. It is optional, for a full-suite run; tests
  that mean "reference" pass it explicitly, and resolution tests clear the variable (§6.1).

Interpret parameters come only from `ops.common.interpret_params(**overrides)`, so DMA semantics, NaN poisoning
and out-of-bounds checks are uniform everywhere; the only deviations are the explicit overrides above.

### 8.4 Time budget (4-core CPU)

| file(s) | estimate |
|---|---|
| test_core | 40 s |
| test_pallas_attention + test_pallas_norm_mlp + test_tpu_lowering | 190 s (eager + races + default-config cases) |
| test_rope + test_model + test_loader | 60 s |
| test_block_manager + test_scheduler | 30 s |
| test_sampler + test_model_runner + test_llm | 60 s |
| test_e2e (reference) | 40 s |
| test_e2e (slow interpret) | 45 s |
| **total** | **about 7.5 min**; `pytest -m "not slow"` about 6 min |

TPU-marked tests are skipped on CPU. On a TPU VM, `pytest -m tpu` compiles the kernels with Mosaic and compares
them with the reference. This is the first thing to run on hardware. It compiles the **same** Mosaic programs as
production: conftest's global `jax_default_matmul_precision="highest"` (kept so the XLA model path on TPU is
comparable with the f64 oracle) cannot reach the kernels, whose matmuls pin `precision=DEFAULT` (verified in
the jaxpr and the Mosaic text).

---

## 9. Risks and future work

### 9.1 Known risks

| # | risk | mitigation |
|---|---|---|
| R1 | Full Mosaic compilation of the kernels is unverified: there is no TPU here, only interpret runs and MLIR lowering. Lowering checks BlockSpec tiling but not scratch/DMA shapes, value reshapes, vector layouts or VMEM/SMEM (§0.3). | Those rules and budgets are asserted by `ops.tpu_rules` in `ModelRunner.__init__` (every bucket) and in the wrappers. The attention kernel uses the idioms of JAX's bundled TPU kernels (§5.4.5: whole-page leading-dim DMAs as in v1 `paged_attention`, full-shape iota masks, f32 selects, `[rows, 128]` m/l widened with `jnp.tile`, `fori_loop` with dynamic bounds, `precision=DEFAULT`), and its Mosaic MLIR was audited for i1 lane-broadcasts, bf16 selects and width-1 lane slices (none). `pytest -m tpu` is the first hardware step and compiles the production programs. Fallbacks: for the bf16 `[ppb, bs, Dc] → [bkv, Dc]` value reshape, upcast to f32 before reshaping; last resort `kernel_overrides=(("attention","reference"),)`, which is a **correctness fallback only**: the reference gathers the whole `max_model_len` context per token (Llama-3-8B, 2k context: 16 MiB per token row, ~8.6 GB of HBM traffic per layer at T=512, ~0.3 s per step on v5e), so its chunk C shrinks to keep the temporary ≤ 256 MiB and the runner warns on TPU. |
| R2 | XLA scatter for the KV write may be slow on TPU for large prefill steps (sub-tile row updates). | Measure with `benchmarks/throughput.py --profile`; future work #1. |
| R3 | The scan-carried cache might be copied by XLA on TPU. On CPU it is verified aliased. | `benchmarks/throughput.py` prints `runner.lower(max_bucket).compile().memory_analysis()` alias and temp bytes. Fallback: an unrolled layer loop (a static layer index works with the same kernel). |
| R4 | SMEM pressure from the `S*MP` page table at long context. | Startup check with an actionable error; raise `block_size`. Streaming the table is future work. |
| R5 | Greedy near-ties (f32 engine vs f64 oracle). | Sharp init, the diversity guard, a teacher-forced comparator with a 1e-3 tie band, and exact equality asserted wherever the oracle has no near-tie. |
| R6 | Interpret mode is slow. | Broad matrix on the reference backend; dense kernel unit tests; two slow interpret e2e runs. |
| R7 | The Pallas SwiGLU kernel may lose to XLA's dot on TPU. | `kernel_overrides=(("mlp","reference"),)`; the benchmark A/B decides the default. |
| R8 | Pallas API churn and private APIs (`tpu_info.registry` + `get_tpu_info.cache_clear`, `_cache_size`, `interpret_pallas_call.races`). | `jax>=0.10.2,<0.11`; private-API tests skip when the API is missing. Compile counting uses `step_fn._cache_size()` (executables); our trace counter alone would miss recompiles without retrace (§5.3). |
| R9 | Silent recompilation mid-run (committed/uncommitted or sharding mismatches recompile without retracing; at 8B that is tens of seconds per bucket). | Every input committed to `runner.device` (§5.3); the compile-bound test counts executables; `stats["num_compiles"]` and the benchmark report it next to `num_traces`. |
| R10 | The first large-bucket step OOMs after the KV pool is allocated (logits, Gumbel noise, activations are not in the pool budget). | Pool sizing subtracts the compiled step's `temp_size_in_bytes` + 256 MiB (§5.3 step 6). |

### 9.2 Future work

1. **A Pallas KV-update kernel, and fusing the write into attention.** The head-major layout makes per-token
   writes sub-tile. Plan: patch each sequence's tail page in VMEM (masked stores) and write whole pages back;
   fully overwritten prefill pages skip the read. Then fold this into the attention kernel's fetch of the last
   kv tile. This is the fix if profiling shows the XLA scatter is slow on TPU (R2).
2. **Async lookahead scheduling** (P2's design):
   - schedule step N+1 while step N runs;
   - feed the sampled tokens back on device through a `prev_src` gather;
   - hash commits stay in `update`;
   - a sequence with a pending token may decode but never prefill.
3. **Single packed int32 H2D buffer** for the StepInput, and preallocated numpy buffers in `prepare`.
4. **head_dim 64 without 2× KV waste:** lane-pack `[K|V]` into 128 lanes (P2's idea).
5. **Tuning:** tuned `(attn_block_q, attn_block_q_small, attn_block_kv, block_size)` tables per chip and
   bucket, a decode-only bucket flavor (a kernel variant with one sequence's G rows per q-tile, beyond the v1
   small decode tile), chip-aware VMEM budgets (v6e's scoped limit is 32 MiB), a megacore (v5p/v4)
   `"parallel"` split, and AOT executables with a persistent compilation cache.
6. **Tensor parallelism (single host).** `Mesh(("tp",))`:
   - `qkv` and `gate_up` sharded on their output dim (GQA-grouped, so a contiguous shard is whole groups;
     `gate_up` needs the loader to order columns `(tp, 2, I/tp)`);
   - `o` and `down` sharded on their input dim;
   - the KV cache sharded on its `Hkv` axis (axis 2);
   - attention wrapped in `jax.shard_map` with replicated metadata;
   - requires `Hkv % tp == 0`.
7. **Copy-on-write for a fully cached last block**, to avoid recomputing up to bs−1 tokens.
8. **Streaming the page table through SMEM** for very long contexts.
9. Per-request logprobs, fp8/int8 KV and weight quantization, sliding window, speculative decoding, LoRA,
   multimodal, and multi-host (all out of scope for v1).

---

## 10. Decision log

| # | Topic | P1 | P2 | P3 | **Decision** (why) |
|---|---|---|---|---|---|
| 1 | KV layout | head-major `[L,P,Hkv,2,ps,D]` | token-major interleaved `[L,P,ps,2Hkv,D]` (+u32 / lane-packed) | token-major `[L·NB,B,2Hkv,Dp]` | **P1 head-major.** Contiguous `[bs,D]` K/V tiles, no strided or bit-trick reads, no HBM padding for any Hkv; same DMA pattern as the TPU-proven v1 `paged_attention`; dtype-agnostic; every line interpretable. |
| 2 | KV write | XLA scatter next to attention | in-kernel prologue, page-run HBM→HBM DMAs | in-kernel per-token DMAs | **XLA scatter shared by both backends** (in place, `mode="drop"`, distinct OOB pads, `unique_indices`). Per-token DMAs are sub-tile in head-major; the fused write is future work #1. |
| 3 | Kernel control flow | host work plan, parity from offsets | SMEM cross-step state | SMEM cross-step state | **P1 host plan.** No mutable cross-step state, testable in numpy, per-pair causal kv-tile skipping. |
| 4 | Invalid pages | fetch page 0 (dummy) | `pl.when`-guarded start/wait | guarded | **Guarded (P2/P3).** No wasted DMA on short sequences, counts balanced. |
| 5 | q layout | `[bq,Hq,D]` + in-kernel fold (f32 upcast) | kv-head-major `[Hkv,T·G,D]` | `[Hq,T,D]` | **P2.** A dense `[bq·G, D]` MXU operand per KV head, no in-kernel relayout; XLA fuses the transposes. |
| 6 | Grid | `(T/bq,)` | `(T/bq,)` | `(Hkv/hb, T/bq)` | **1-D, static unroll over Hkv.** The stream crosses tiles, one TensorCore. |
| 7 | head_dim 64 | defer | lane packing | pad weights to 128 | **Pad the cache head dim in the attention op** (activations only, no loader or weight changes); lane packing is future work. |
| 8 | Bucketing / compile | T only, jit | T × decode_only, AOT dict | T + (S_smp × mode), two jits | **T only, one jit** with a module-level cache, an architecture-only static key, and every input committed to one device. At most `len(buckets)` compiles, counted as executables (`_cache_size`), not traces; lm_head overcompute on S_pad rows accepted. The q-tile varies per bucket (small decode tile for `T_pad <= next_pow2(max_num_seqs)`), which costs no extra compiles. |
| 9 | Sampling | per-row argsort, one step key | `top_k(64)` + per-request keys | sort thresholds + per-request keys | **Per-request keys `fold_in(seed, gen_idx)`**; full-vocab Gumbel for plain temperature; `lax.top_k(128)` candidates for top-k/p with exact full-vocab top-p mass; `lax.cond` all-greedy fast path. |
| 10 | Host/device overlap | sync | lookahead-1 async | sync | **Sync in v1.** The async state machine is future work #2. |
| 11 | H2D | pytree device_put | one packed i32 buffer | pytree | **Pytree** (clear, testable); packing is future work #3. |
| 12 | Prefix hash | sha256 chain | `hash()` + stored-token check | `(parent, tokens)` key | **sha256 chain from a fixed ROOT.** Deterministic across processes (P3's `hash(str)` root is not), collision-safe, no token storage. |
| 13 | Preemption victim | latest excluding self | `running[-1]`, may be self | max arrival not scheduled, else skip | **Max `seq_id` among unscheduled running, including self** (self → self-preempt). Progress is guaranteed (the lowest-`seq_id` unscheduled running seq always fits once all newer unscheduled seqs are preempted) and nothing starves (FIFO re-admission); it is **not** strict priority, because decodes scheduled in pass 1 are never un-scheduled (§5.2). |
| 14 | Freed unhashed blocks | LRU front | LRU front | LRU tail | **Front:** no reuse value. |
| 15 | Allocation order | hits first | hits first | hits first (+ test) | **Touch-before-pop, with a dedicated test.** |
| 16 | Waiting order on preempt | appendleft | appendleft | appendleft | **Insert sorted by `seq_id`**, so `waiting` stays in arrival order. |
| 17 | Oracle | dense JAX on HF dict | dense JAX (`reference.py`) | f64 NumPy, independent | **P3 f64 NumPy**, which parses the HF dict itself. |
| 18 | Comparator | stop at first near-tie | tie-tolerant at mismatch | teacher-forced + exact | **Teacher-forced tie-aware (1e-3) + exact equality** when the oracle has no near-tie. |
| 19 | Weight layout | 2-D grouped qkv; `[H,2I]` | 2-D plain concat; `[2,H,I]` | 4-D `[H,Hkv,G+2,Dp]`; `[H,2,I]` | **P1** (2-D: no TPU tile padding; grouped for TP). |
| 20 | SwiGLU fusion | Pallas epilogue | both, pick per bucket | XLA only | **Pallas epilogue behind the `mlp` op switch**; `auto` → pallas on TPU; the benchmark A/Bs it via `kernel_overrides`. |
| 21 | Norm numerics | f32 add | f32 stats | HF order | **HF order:** the residual rounded to dtype, f32 stats, cast, × weight. Bit-exact kernel vs reference. |
| 22 | RoPE | in-graph from positions | in-graph | cos/sin tables | **In-graph, once per step** outside the scan (no table memory). |
| 23 | Backend switch | global + per-op overrides | string spec + env | env + per-op | **Global + per-op overrides in EngineConfig**, and the env var replaces `auto`; resolved once into a static KernelConfig. |
| 24 | Interpret params | default | on_wait + eager | nan + OOB + races | **Defaults** (on_wait, nan, OOB raise) for every engine path; kernel unit tests additionally run eager DMAs and one race-detector case through `interpret_params(**overrides)` (P2/P3); races opt-in via env elsewhere. |
| 25 | Bundled-kernel backend | oracle only | TPU fallback | day-one TPU backend | **Oracle only.** It needs the token-major layout; the TPU fallback is `kernel_overrides=(("attention","reference"),)`. |
| 26 | Invariant checks | none | none | I1–I7, S1–S5, B1–B5 | **I1–I7, S1–S4 behind `debug_checks`**; B-rules enforced by builder asserts. |
| 27 | Scheduler test harness | unit tests | fake runner | FakeRunner (token hash) | **FakeRunner with a simulated paged KV store** (prefix-hash per slot); catches block-table and eviction bugs (mutation-verified). |
| 28 | Chunk cap | — | `max_prefill_chunk` | `long_prefill_token_threshold` | **`max_prefill_chunk`.** |
| 29 | Deadlock | — | — | `EngineDeadlock` | **Adopted.** |
| 30 | Layer addressing | `[L,P,…]` + layer scalar | same | global page ids | **`[L,P,…]` + layer scalar prefetch.** |
| 31 | `scan_layers=False` fallback | — | yes | yes | **Not in v1** (the scan carry verified in place on CPU); add it only if TPU copies the carry. |
| 32 | Pool sizing | memory_stats | + compiled temp size | memory_stats | **memory_stats × fraction, minus the compiled step's temp size (P2)** when the slack does not cover it, with a CPU byte budget fallback; validated ≥ `pages_per_seq`. |
| 33 | Defaults | bs 16, bq 32, bkv 128 | ps 64 | B 32 | **bs 16, bq 32, bkv 128** (SMEM-safe, fine-grained prefix reuse); long contexts are advised to raise bs. |
| 34 | `abort()` | — | — | yes | **Adopted** (scheduler + LLM); the abort output is returned by the next `step()`. |
| 35 | TPU rules | wrapper asserts | wrapper asserts | wrapper asserts | **One WP1 module (`ops.tpu_rules`)** used by the runner (all buckets, at construction) and the wrappers; kernels choose tiles from a VMEM budget instead of fixed sizes. |

---

## Appendix A: verbatim shared files (owned by WP1)

Create these files **exactly** as shown. Nobody else edits them. Every snippet was executed while writing
this spec: the e2e prototype, the scheduler stress test and a pytest smoke run all used these exact files.

### A.1 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "flash-jax"
version = "0.1.0"
description = "A mini vLLM-style LLM inference engine in JAX, built for TPUs"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = ["jax>=0.10.2,<0.11", "numpy", "safetensors>=0.4"]

[project.optional-dependencies]
tpu = ["jax[tpu]>=0.10.2,<0.11"]
tokenizer = ["tokenizers"]
dev = ["pytest>=7"]

[tool.setuptools.packages.find]
include = ["flash_jax*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]  # import flash_jax from a checkout without installing (pytest >= 7)
addopts = "-ra"
markers = [
    "slow: Pallas-interpret end-to-end tests (tens of seconds each on CPU)",
    "tpu: needs a real TPU (skipped elsewhere)",
]
```
### A.2 `flash_jax/__init__.py`

```python
"""flash-jax: a mini vLLM-style LLM inference engine in JAX, built for TPUs."""
from flash_jax.config import EngineConfig, KernelConfig, ModelConfig
from flash_jax.sampling_params import SamplingParams

__all__ = ["LLM", "RequestOutput", "EngineConfig", "KernelConfig", "ModelConfig", "SamplingParams"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy so that `import flash_jax.<anything>` works before/without the engine package (parallel development).
    if name in ("LLM", "RequestOutput"):
        from flash_jax.engine import llm

        return getattr(llm, name)
    raise AttributeError(f"module 'flash_jax' has no attribute {name!r}")
```
### A.3 `flash_jax/config.py`

```python
"""Static configuration: model architecture, engine knobs, resolved kernel settings, bucket helpers.

Every dataclass here is frozen (hashable) so it can be a static argument of `jax.jit`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

Backend = Literal["auto", "pallas", "interpret", "reference"]
KERNEL_OPS: tuple[str, ...] = ("attention", "norm", "mlp")
_BACKENDS = ("auto", "pallas", "interpret", "reference")


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def round_up(a: int, b: int) -> int:
    return cdiv(a, b) * b


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of a Llama-family decoder (Llama 2/3, Qwen2, Qwen3)."""

    model_type: str  # "llama" | "qwen2" | "qwen3"
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int  # explicit: Qwen3 has num_heads * head_dim != hidden_size
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # llama3 scaling: (factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)
    rope_scaling: tuple[float, float, float, int] | None = None
    tie_word_embeddings: bool = False
    qkv_bias: bool = False  # Qwen2 (always), Llama/Qwen3 when attention_bias=True
    o_bias: bool = False  # Llama/Qwen3 when attention_bias=True
    qk_norm: bool = False  # Qwen3 per-head RMSNorm on q and k (before RoPE)
    max_position_embeddings: int = 4096
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be even (RoPE)")

    @property
    def q_per_kv(self) -> int:
        """G: query heads per KV head."""
        return self.num_heads // self.num_kv_heads

    @property
    def qkv_width(self) -> int:
        """Output width of the fused QKV projection: Hkv * (G + 2) * D."""
        return self.num_kv_heads * (self.q_per_kv + 2) * self.head_dim

    @classmethod
    def from_hf(cls, hf: dict[str, Any], generation_config: dict[str, Any] | None = None) -> "ModelConfig":
        """Parse a Hugging Face `config.json` dict (+ optional `generation_config.json` dict)."""
        mt = hf.get("model_type", "llama")
        if mt not in ("llama", "qwen2", "qwen3"):
            raise NotImplementedError(f"model_type={mt!r}")
        if hf.get("use_sliding_window", False):
            raise NotImplementedError("sliding-window attention")
        if hf.get("mlp_bias", False):
            raise NotImplementedError("mlp_bias")
        if hf.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"hidden_act={hf.get('hidden_act')!r}")
        H, nh = hf["hidden_size"], hf["num_attention_heads"]
        rope_params = hf.get("rope_parameters") or {}
        theta = float(hf.get("rope_theta", rope_params.get("rope_theta", 10000.0)))
        rs = hf.get("rope_scaling") or rope_params or None
        scaling = None
        if rs:
            rtype = rs.get("rope_type", rs.get("type", "default"))
            if rtype == "llama3":
                scaling = (float(rs["factor"]), float(rs["low_freq_factor"]), float(rs["high_freq_factor"]),
                           int(rs["original_max_position_embeddings"]))
            elif rtype != "default":
                raise NotImplementedError(f"rope_type={rtype!r}")
        attn_bias = bool(hf.get("attention_bias", False))
        eos: list[int] = []
        for src in (hf, generation_config or {}):
            e = src.get("eos_token_id")
            for t in ([] if e is None else [e] if isinstance(e, int) else e):
                if t not in eos:
                    eos.append(int(t))
        return cls(
            model_type=mt,
            vocab_size=hf["vocab_size"],
            hidden_size=H,
            intermediate_size=hf["intermediate_size"],
            num_layers=hf["num_hidden_layers"],
            num_heads=nh,
            num_kv_heads=hf.get("num_key_value_heads") or nh,
            head_dim=hf.get("head_dim") or H // nh,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            rope_theta=theta,
            rope_scaling=scaling,
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            qkv_bias=True if mt == "qwen2" else attn_bias,
            o_bias=False if mt == "qwen2" else attn_bias,
            qk_norm=mt == "qwen3",
            max_position_embeddings=int(hf.get("max_position_embeddings", 4096)),
            eos_token_ids=tuple(eos),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        """Read `<path>/config.json` and, if present, `<path>/generation_config.json`."""
        with open(os.path.join(path, "config.json")) as f:
            hf = json.load(f)
        gen = None
        gpath = os.path.join(path, "generation_config.json")
        if os.path.exists(gpath):
            with open(gpath) as f:
                gen = json.load(f)
        return cls.from_hf(hf, gen)


@dataclass(frozen=True)
class EngineConfig:
    """User-facing engine knobs. Validated on construction (model-independent checks only)."""

    max_model_len: int = 2048  # max prompt + generated tokens per sequence
    max_num_seqs: int = 64  # max RUNNING sequences (and max sequences per step)
    max_num_batched_tokens: int = 512  # per-step token budget (chunked prefill)
    max_prefill_chunk: int | None = None  # optional per-sequence cap on new tokens in one step (non-decodes)
    block_size: int = 16  # tokens per KV page
    num_kv_blocks: int | None = None  # None: derived from device memory (see ModelRunner)
    kv_memory_fraction: float = 0.85  # of free device memory after params, when num_kv_blocks is None
    cpu_kv_cache_bytes: int = 256 << 20  # KV budget when device.memory_stats() is unavailable (CPU)
    enable_prefix_caching: bool = True
    dtype: str = "bfloat16"  # params, activations, KV cache: "bfloat16" | "float32"
    kernel_backend: Backend = "auto"
    kernel_overrides: tuple[tuple[str, Backend], ...] = ()  # e.g. (("mlp", "reference"),)
    attn_block_q: int = 32  # attention q-tile (tokens), power of two; effective = KernelConfig.attn_tile_q(T_pad)
    # q-tile for buckets T_pad <= next_pow2(max_num_seqs) (decode-only steps land there), power of two. None: on
    # "pallas" the smallest tile with block_q * G a multiple of the sublane tile (4 for G=4 in bf16), else attn_block_q.
    attn_block_q_small: int | None = None
    attn_block_kv: int = 128  # attention kv-tile (tokens), multiple of block_size; a multiple of 128 on TPU
    kv_head_dim_align: int | None = None  # None: 128 if the attention backend resolves to "pallas", else 1
    min_token_bucket: int = 16  # smallest T_pad, power of two
    sampler_max_candidates: int = 128  # candidates kept for top-k / top-p sampling
    seed: int = 0  # engine RNG (per-request seeds are drawn from it when not given)
    debug_checks: bool = False  # run invariant checks after every step (tests force True)

    def __post_init__(self) -> None:
        errs = []
        if self.max_model_len < 2:
            errs.append("max_model_len must be >= 2")
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            errs.append("max_num_seqs and max_num_batched_tokens must be >= 1")
        if self.max_prefill_chunk is not None and self.max_prefill_chunk < 1:
            errs.append("max_prefill_chunk must be >= 1 or None")
        if self.block_size < 1 or self.attn_block_kv % self.block_size:
            errs.append("attn_block_kv must be a positive multiple of block_size")
        if not (_is_pow2(self.attn_block_q) and _is_pow2(self.min_token_bucket)
                and (self.attn_block_q_small is None or _is_pow2(self.attn_block_q_small))):
            errs.append("attn_block_q, attn_block_q_small and min_token_bucket must be powers of two")
        if self.dtype not in ("bfloat16", "float32"):
            errs.append(f"dtype={self.dtype!r}")
        if self.kernel_backend not in _BACKENDS:
            errs.append(f"kernel_backend={self.kernel_backend!r}")
        for op, be in self.kernel_overrides:
            if op not in KERNEL_OPS or be not in _BACKENDS:
                errs.append(f"bad kernel override {(op, be)!r}")
        if self.num_kv_blocks is not None and self.num_kv_blocks < 1:
            errs.append("num_kv_blocks must be >= 1")
        if self.sampler_max_candidates < 1:
            errs.append("sampler_max_candidates must be >= 1")
        if errs:
            raise ValueError("invalid EngineConfig: " + "; ".join(errs))

    @property
    def pages_per_seq(self) -> int:
        """MP: fixed width of every sequence's row in the flattened page table."""
        return cdiv(self.max_model_len, self.block_size)

    @property
    def attn_pages_per_kv_tile(self) -> int:
        return self.attn_block_kv // self.block_size

    def token_buckets(self) -> tuple[int, ...]:
        """T_pad values: min_token_bucket * 2**i up to next_pow2(max_num_batched_tokens)."""
        top = max(self.min_token_bucket, next_pow2(self.max_num_batched_tokens))
        out, b = [], self.min_token_bucket
        while b <= top:
            out.append(b)
            b *= 2
        return tuple(out)

    def bucket_for(self, num_tokens: int) -> int:
        """Smallest token bucket >= num_tokens."""
        for b in self.token_buckets():
            if b >= num_tokens:
                return b
        raise ValueError(f"{num_tokens} tokens exceed the largest bucket")

    def num_seq_slots(self, t_pad: int) -> int:
        """S_pad for a token bucket: every sequence has >= 1 token, so S_pad = min(T_pad, max_num_seqs)."""
        return min(t_pad, self.max_num_seqs)


@dataclass(frozen=True)
class KernelConfig:
    """Resolved (never "auto") per-op backends and tile sizes; a static argument of the jitted step."""

    attention: str = "reference"  # "pallas" | "interpret" | "reference"
    norm: str = "reference"
    mlp: str = "reference"
    attn_block_q: int = 32  # q-tile of buckets > attn_small_bucket
    attn_block_q_small: int = 32  # q-tile of buckets <= attn_small_bucket
    attn_small_bucket: int = 0  # largest T_pad that uses attn_block_q_small (0: none)
    attn_pages_per_kv_tile: int = 8

    def attn_tile_q(self, t_pad: int) -> int:
        """THE effective attention q-tile of a bucket, used by both the work plan and the kernel (divides t_pad:
        both are powers of two)."""
        bq = self.attn_block_q_small if t_pad <= self.attn_small_bucket else self.attn_block_q
        return min(bq, t_pad)


def resolve_backend(requested: str, platform: str | None = None) -> str:
    """Map "auto" to a concrete backend. Env FLASH_JAX_KERNEL_BACKEND (if set) replaces "auto"."""
    if requested != "auto":
        return requested
    env = os.environ.get("FLASH_JAX_KERNEL_BACKEND")
    if env:
        if env not in _BACKENDS or env == "auto":
            raise ValueError(f"FLASH_JAX_KERNEL_BACKEND={env!r}")
        return env
    if platform is None:
        import jax

        platform = jax.default_backend()
    return "pallas" if platform == "tpu" else "reference"


def resolve_kernel_config(cfg: EngineConfig, platform: str | None = None, *, q_per_kv: int = 1) -> KernelConfig:
    """Resolve backends (once, in ModelRunner.__init__) and the per-bucket attention q-tiles. q_per_kv = G."""
    overrides = dict(cfg.kernel_overrides)
    be = {op: resolve_backend(overrides.get(op, cfg.kernel_backend), platform) for op in KERNEL_OPS}
    small = cfg.attn_block_q_small
    if small is None:
        small = cfg.attn_block_q
        if be["attention"] == "pallas":  # decode q-tiles: fill one sublane tile with block_q * G rows, no more
            rows = 16 if cfg.dtype == "bfloat16" else 8
            small = 1
            while small < cfg.attn_block_q and (small * q_per_kv) % rows:
                small *= 2
    return KernelConfig(attention=be["attention"], norm=be["norm"], mlp=be["mlp"], attn_block_q=cfg.attn_block_q,
                        attn_block_q_small=small, attn_small_bucket=next_pow2(cfg.max_num_seqs),
                        attn_pages_per_kv_tile=cfg.attn_pages_per_kv_tile)


def kv_head_dim(model: ModelConfig, cfg: EngineConfig, kc: KernelConfig) -> int:
    """Head dim of the KV cache (>= model.head_dim; the extra lanes are zero)."""
    align = cfg.kv_head_dim_align or (128 if kc.attention == "pallas" else 1)
    return round_up(model.head_dim, align)
```
### A.4 `flash_jax/sampling_params.py`

```python
"""Per-request sampling parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0  # <= 0 means greedy (exact argmax over the full vocab)
    top_k: int = 0  # 0 disables; otherwise keep the k most likely tokens
    top_p: float = 1.0  # nucleus mass in (0, 1]; 1.0 disables
    max_tokens: int = 16  # max generated tokens (clamped to max_model_len - prompt_len)
    seed: int | None = None  # None: drawn from the engine RNG at add_request
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_token_ids", tuple(int(t) for t in self.stop_token_ids))
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0
```
### A.5 `flash_jax/engine/sequence.py`

```python
"""Sequence: the host-side state of one request."""
from __future__ import annotations

import enum
from collections.abc import Sequence as Seq

from flash_jax.sampling_params import SamplingParams


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    """A request. `token_ids` = prompt + generated tokens; KV exists for token_ids[:num_computed_tokens]."""

    def __init__(self, seq_id: int, prompt_token_ids: Seq[int], params: SamplingParams, *, seed: int,
                 max_tokens: int) -> None:
        if len(prompt_token_ids) == 0:
            raise ValueError("empty prompt")
        self.seq_id = seq_id  # monotonically increasing: also the arrival order / priority (lower first)
        self.token_ids: list[int] = [int(t) for t in prompt_token_ids]
        self.num_prompt_tokens = len(self.token_ids)
        self.params = params
        self.seed = seed  # resolved per-request sampling seed
        self.max_tokens = max_tokens  # already clamped by the caller
        self.status = SeqStatus.WAITING
        self.block_table: list[int] = []  # physical KV block ids, in logical order
        self.block_hashes: list[bytes] = []  # memo: chained hashes of the first len() FULL blocks of token_ids
        self.num_sealed_blocks = 0  # block_table[:num_sealed_blocks] have been offered to the prefix cache
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0  # prefix-cache hit tokens at the latest admission (stats)
        self.num_preemptions = 0
        self.finish_reason: str | None = None  # "stop" | "length" | "abort"

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_remaining(self) -> int:
        """Tokens whose KV still has to be computed before the next token can be sampled."""
        return self.num_tokens - self.num_computed_tokens

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SeqStatus.FINISHED

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, {self.status.value}, tokens={self.num_tokens}, "
                f"computed={self.num_computed_tokens}, blocks={self.block_table})")
```
### A.6 `flash_jax/batch.py`

```python
"""Per-step batch contracts shared by scheduler, runner, model and kernels.

Host code builds these as numpy int32/float32 arrays; the runner `jax.device_put`s the whole StepInput
pytree once per step. All shapes depend only on the token bucket T_pad (see EngineConfig).
prepare_step_input is THE builder: the runner, model tests and benchmarks all use it.
"""
from __future__ import annotations

from collections.abc import Sequence as Seq
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from flash_jax.config import EngineConfig, KernelConfig, cdiv
from flash_jax.engine.sequence import Sequence


@dataclass(frozen=True, eq=False)
class ScheduledSeq:
    """One sequence's work in a step: compute KV (and hidden states) for token_ids[start_pos:start_pos+num_new_tokens]."""

    seq: Sequence
    start_pos: int  # == seq.num_computed_tokens when scheduled
    num_new_tokens: int  # >= 1
    samples: bool  # start_pos + num_new_tokens == seq.num_tokens: the sampled token is kept


class AttentionMetadata(NamedTuple):
    """Ragged batch description. Sizes: T=T_pad, S=S_pad, MP=pages_per_seq, NB=T_pad // block_q."""

    slot_mapping: Any  # i32[T]  page * block_size + pos % block_size for each real token; -1 for padding
    cu_q_lens: Any  # i32[S+1] cu[0]=0, cu[i+1]=cu[i]+q_len_i; entries past num_seqs repeat the total N
    kv_lens: Any  # i32[S]    context length AFTER this step (start_pos + q_len); 0 for padding seqs
    page_table: Any  # i32[S*MP] row i = block table of seq i (row-major, flattened); unused entries 0
    num_seqs: Any  # i32[1]
    # kernel work plan (see build_work_plan); P = NB + S
    pair_seq: Any  # i32[P]
    pair_num_kv_tiles: Any  # i32[P]
    pair_kv_offset: Any  # i32[P]
    tile_pair_start: Any  # i32[NB+1]


class StepInput(NamedTuple):
    """Everything the jitted step consumes (besides params and the KV cache)."""

    input_ids: Any  # i32[T]  0 for padding
    positions: Any  # i32[T]  start_pos + j for real tokens; 0 for padding
    logits_indices: Any  # i32[S] cu_q_lens[i+1]-1 (last new token of seq i); 0 for padding
    temperature: Any  # f32[S] <= 0 means greedy; padding 0.0
    top_k: Any  # i32[S]  0 = off
    top_p: Any  # f32[S]  1.0 = off
    seeds: Any  # i32[S]  per-request seed
    gen_idx: Any  # i32[S]  number of tokens generated so far (RNG counter)
    attn: AttentionMetadata


def build_work_plan(cu_q_lens: np.ndarray, kv_lens: np.ndarray, num_seqs: int, t_pad: int, block_q: int,
                    block_kv: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate (q-tile, sequence) overlaps ("pairs") for the attention kernel.

    q-tile b covers flat token rows [b*block_q, (b+1)*block_q). For every sequence s and every tile b it overlaps,
    one pair is emitted with the number of kv tiles it must stream: cdiv(last_pos + 1, block_kv), where last_pos
    is the largest position of seq s inside tile b (causal kv-tile skipping). Pairs are ordered by (s, b), which
    equals (b, s) order because sequences occupy contiguous, increasing token ranges.

    Returns (pair_seq, pair_num_kv_tiles, pair_kv_offset, tile_pair_start):
      pair_kv_offset[p] = exclusive prefix sum of pair_num_kv_tiles (global kv-tile index, gives buffer parity),
      tile_pair_start   = CSR row pointer: pairs of tile b are [tile_pair_start[b], tile_pair_start[b+1]).
    Unused pair slots are 0. Capacity NB + S is enough: sum over seqs of tiles spanned <= num_seqs + NB - 1.
    """
    assert t_pad % block_q == 0
    nb, s_pad = t_pad // block_q, len(kv_lens)
    cap = nb + s_pad
    pair_seq = np.zeros(cap, np.int32)
    pair_nkv = np.zeros(cap, np.int32)
    pair_off = np.zeros(cap, np.int32)
    tile_count = np.zeros(nb, np.int32)
    p = off = 0
    for s in range(num_seqs):
        lo, hi = int(cu_q_lens[s]), int(cu_q_lens[s + 1])
        assert hi > lo, "every scheduled sequence has >= 1 new token"
        ctx = int(kv_lens[s]) - (hi - lo)
        for b in range(lo // block_q, (hi - 1) // block_q + 1):
            last_pos = ctx + min(hi, (b + 1) * block_q) - 1 - lo
            n = cdiv(last_pos + 1, block_kv)
            pair_seq[p], pair_nkv[p], pair_off[p] = s, n, off
            tile_count[b] += 1
            off += n
            p += 1
    tile_start = np.zeros(nb + 1, np.int32)
    tile_start[1:] = np.cumsum(tile_count)
    return pair_seq, pair_nkv, pair_off, tile_start


def build_attention_metadata(q_lens: Seq[int], kv_lens: Seq[int], block_tables: Seq[Seq[int]], *, t_pad: int,
                             s_pad: int, block_size: int, pages_per_seq: int, block_q: int,
                             block_kv: int) -> AttentionMetadata:
    """Host-side (numpy) AttentionMetadata for sequences laid out back-to-back in the given order.

    Sequence i contributes q_lens[i] tokens at positions [kv_lens[i]-q_lens[i], kv_lens[i]); its KV pages are
    block_tables[i][: cdiv(kv_lens[i], block_size)].
    """
    n_seqs = len(q_lens)
    assert n_seqs == len(kv_lens) == len(block_tables) and n_seqs <= s_pad
    q = np.asarray(q_lens, np.int32).reshape(n_seqs)
    kv = np.asarray(kv_lens, np.int32).reshape(n_seqs)
    assert np.all(q >= 1) and np.all(kv >= q) and np.all(kv <= pages_per_seq * block_size)
    n_tok = int(q.sum())
    assert n_tok <= t_pad
    cu = np.full(s_pad + 1, n_tok, np.int32)
    cu[0] = 0
    cu[1:n_seqs + 1] = np.cumsum(q)
    kv_pad = np.zeros(s_pad, np.int32)
    kv_pad[:n_seqs] = kv
    page_table = np.zeros((s_pad, pages_per_seq), np.int32)
    slot = np.full(t_pad, -1, np.int32)
    for i in range(n_seqs):
        pages = np.asarray(block_tables[i][: cdiv(int(kv[i]), block_size)], np.int32)
        assert len(pages) == cdiv(int(kv[i]), block_size), "block table too short"
        page_table[i, : len(pages)] = pages
        pos = np.arange(kv[i] - q[i], kv[i])
        slot[cu[i]:cu[i + 1]] = pages[pos // block_size] * block_size + pos % block_size
    plan = build_work_plan(cu, kv_pad, n_seqs, t_pad, block_q, block_kv)
    return AttentionMetadata(slot, cu, kv_pad, page_table.reshape(-1), np.array([n_seqs], np.int32), *plan)


def prepare_step_input(batch: Seq[ScheduledSeq], cfg: EngineConfig, kc: KernelConfig, *,
                       t_pad: int | None = None) -> StepInput:
    """Numpy StepInput for a scheduled batch, laid out back to back in batch order (DESIGN.md 3.4).

    T_pad = cfg.bucket_for(N) unless given (warmup / dummy inputs: an empty batch in any bucket); the work plan
    uses the bucket's q-tile kc.attn_tile_q(T_pad) and kv-tile cfg.attn_block_kv. One Python iteration per
    sequence, never per token.
    """
    n = len(batch)
    q_lens = [b.num_new_tokens for b in batch]
    t_pad = cfg.bucket_for(max(sum(q_lens), 1)) if t_pad is None else t_pad
    s_pad = cfg.num_seq_slots(t_pad)
    md = build_attention_metadata(q_lens, [b.start_pos + b.num_new_tokens for b in batch],
                                  [b.seq.block_table for b in batch], t_pad=t_pad, s_pad=s_pad,
                                  block_size=cfg.block_size, pages_per_seq=cfg.pages_per_seq,
                                  block_q=kc.attn_tile_q(t_pad), block_kv=cfg.attn_block_kv)
    input_ids = np.zeros(t_pad, np.int32)
    positions = np.zeros(t_pad, np.int32)
    for i, b in enumerate(batch):
        lo, hi = int(md.cu_q_lens[i]), int(md.cu_q_lens[i + 1])
        input_ids[lo:hi] = b.seq.token_ids[b.start_pos:b.start_pos + b.num_new_tokens]
        positions[lo:hi] = np.arange(b.start_pos, b.start_pos + b.num_new_tokens, dtype=np.int32)
    logits_indices = np.zeros(s_pad, np.int32)
    logits_indices[:n] = md.cu_q_lens[1:n + 1] - 1
    temperature = np.zeros(s_pad, np.float32)
    top_k = np.zeros(s_pad, np.int32)
    top_p = np.ones(s_pad, np.float32)
    seeds = np.zeros(s_pad, np.int32)
    gen_idx = np.zeros(s_pad, np.int32)
    for i, b in enumerate(batch):
        p = b.seq.params
        temperature[i], top_k[i], top_p[i] = p.temperature, p.top_k, p.top_p
        seeds[i], gen_idx[i] = b.seq.seed, b.seq.num_output_tokens
    return StepInput(input_ids, positions, logits_indices, temperature, top_k, top_p, seeds, gen_idx, md)
```
### A.7 `flash_jax/ops/common.py`

```python
"""Constants and helpers shared by reference ops and Pallas kernels."""
from __future__ import annotations

import os

import numpy as np

# Finite "minus infinity" for masked attention scores (same constant as JAX's bundled TPU kernels).
MASK_VALUE: float = -0.7 * float(np.finfo(np.float32).max)


def interpret_params(**overrides):
    """InterpretParams used for backend == "interpret" (Pallas TPU semantics simulated on CPU).

    Defaults in jax 0.10.2: dma_execution_mode="on_wait" (a missing .wait() shows up as wrong data),
    uninitialized_memory="nan" (reads of never-written VMEM/scratch poison the result), out_of_bounds_reads="raise".
    FLASH_JAX_DETECT_RACES=1 additionally enables the (slow) race detector. Tests pass overrides, e.g.
    interpret_params(dma_execution_mode="eager") (exposes DMA write-after-read hazards that "on_wait" hides) or
    detect_races=True, and hand the result to a kernel's `interpret=` argument.
    """
    from jax.experimental.pallas import tpu as pltpu

    kw = {"detect_races": os.environ.get("FLASH_JAX_DETECT_RACES") == "1", **overrides}
    return pltpu.InterpretParams(**kw)
```
### A.8 `flash_jax/ops/reference.py`

```python
"""Pure-JAX reference implementations. They define the semantics every Pallas kernel must match."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from flash_jax.batch import AttentionMetadata
from flash_jax.ops.common import MASK_VALUE


def write_kv(kv_cache: jax.Array, layer: jax.Array, k: jax.Array, v: jax.Array,
             slot_mapping: jax.Array) -> jax.Array:
    """Scatter new K/V rows into the paged cache (in place when the cache is donated).

    kv_cache [L, P, Hkv, 2, ps, Dc]; k, v [T, Hkv, Dc]; slot_mapping i32[T] (-1 = padding, never written).
    Padding tokens are sent to distinct out-of-bounds pages (P + t) and dropped. NEVER use a negative index
    with mode="drop": jnp scatter wraps negative indices (x.at[-1] writes the last element).
    """
    _, P, _, _, ps, _ = kv_cache.shape
    t = jnp.arange(slot_mapping.shape[0], dtype=jnp.int32)
    real = slot_mapping >= 0
    page = jnp.where(real, slot_mapping // ps, P + t)
    off = jnp.where(real, slot_mapping % ps, 0)
    new = jnp.stack([k, v], axis=2).astype(kv_cache.dtype)  # [T, Hkv, 2, Dc]
    return kv_cache.at[layer, page, :, :, off, :].set(new, mode="drop", unique_indices=True)


def paged_attention(q: jax.Array, kv_cache: jax.Array, layer: jax.Array, md: AttentionMetadata, *,
                    sm_scale: float, chunk: int = 32, max_chunk_bytes: int = 256 << 20) -> jax.Array:
    """Causal ragged paged attention over an ALREADY WRITTEN cache.

    q [T, Hq, D] (D == cache head dim); returns o [T, Hq, D] in q.dtype. Token t of sequence i sits at position
    kv_lens[i] - q_len_i + (t - cu_q_lens[i]) and attends to cache positions [0, pos]. Rows t >= N (padding)
    are exactly 0. Masked K/V are zeroed before use, so garbage/NaN in unused cache slots cannot leak.
    Processes C = gcd(T, chunk) tokens at a time (lax.map), halved until the gathered f32 [C, MP*ps] K/V temporary
    fits max_chunk_bytes (C >= 1). It gathers the WHOLE max_model_len context per token: a semantics oracle and a
    correctness fallback, not a fast path.
    """
    T, Hq, D = q.shape
    _, _, Hkv, _, ps, _ = kv_cache.shape
    G = Hq // Hkv
    S = md.kv_lens.shape[0]
    MP = md.page_table.shape[0] // S
    page_table = md.page_table.reshape(S, MP)
    cu = md.cu_q_lens
    tok = jnp.arange(T, dtype=jnp.int32)
    seq = jnp.clip(jnp.searchsorted(cu[1:], tok, side="right"), 0, S - 1).astype(jnp.int32)
    pos = md.kv_lens[seq] - (cu[seq + 1] - cu[seq]) + tok - cu[seq]
    valid = tok < cu[md.num_seqs[0]]
    C = math.gcd(T, chunk)
    while C > 1 and C * MP * ps * Hkv * 2 * D * 4 > max_chunk_bytes:
        C //= 2
    col = jnp.arange(MP * ps, dtype=jnp.int32)

    def one_chunk(args):
        qc, sc, pc, vc = args  # [C, Hq, D], [C], [C], [C]
        kv = kv_cache[layer, page_table[sc]]  # [C, MP, Hkv, 2, ps, D]
        kv = kv.transpose(0, 2, 3, 1, 4, 5).reshape(C, Hkv, 2, MP * ps, D).astype(jnp.float32)
        mask = (col[None, :] <= pc[:, None]) & vc[:, None]  # [C, K]
        kv = jnp.where(mask[:, None, None, :, None], kv, 0.0)
        qf = qc.astype(jnp.float32).reshape(C, Hkv, G, D)
        s = jnp.einsum("chgd,chkd->chgk", qf, kv[:, :, 0], precision=lax.Precision.HIGHEST) * sm_scale
        s = jnp.where(mask[:, None, None, :], s, MASK_VALUE)
        p = jax.nn.softmax(s, axis=-1) * mask[:, None, None, :]
        o = jnp.einsum("chgk,chkd->chgd", p, kv[:, :, 1], precision=lax.Precision.HIGHEST)
        return o.reshape(C, Hq, D)

    o = lax.map(one_chunk, (q.reshape(T // C, C, Hq, D), seq.reshape(T // C, C), pos.reshape(T // C, C),
                            valid.reshape(T // C, C)))
    return o.reshape(T, Hq, D).astype(q.dtype)


def add_rms_norm(x: jax.Array, residual: jax.Array, weight: jax.Array, eps: float) -> tuple[jax.Array, jax.Array]:
    """(y, r) with r = x + residual (in x.dtype) and y = RMSNorm(r) * weight, HF numerics (f32 statistics)."""
    r = (x.astype(jnp.float32) + residual.astype(jnp.float32)).astype(x.dtype)
    return rms_norm(r, weight, eps), r


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """HF LlamaRMSNorm over the last axis: (x_f32 * rsqrt(mean(x_f32^2) + eps)).astype(dtype) * weight."""
    xf = x.astype(jnp.float32)
    y = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype) * weight.astype(x.dtype)


def swiglu(h: jax.Array, w_gate_up: jax.Array) -> jax.Array:
    """silu(h @ W_gate) * (h @ W_up) with w_gate_up = [W_gate | W_up] of shape [H, 2I]; f32 accumulate/epilogue."""
    inter = w_gate_up.shape[1] // 2
    gu = jnp.dot(h, w_gate_up, preferred_element_type=jnp.float32)
    g, u = gu[:, :inter], gu[:, inter:]
    return (jax.nn.silu(g) * u).astype(h.dtype)
```
### A.9 `flash_jax/ops/__init__.py`

```python
"""Backend-dispatching ops used by the model. `backend` is a resolved string: "pallas" | "interpret" | "reference".

Pallas implementations are imported lazily so the reference path works without flash_jax.ops.pallas.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from flash_jax.batch import AttentionMetadata
from flash_jax.ops import reference as ref
from flash_jax.ops.reference import rms_norm  # noqa: F401  (qk-norm: jnp only, XLA fuses it)


def paged_attention(q: jax.Array, k: jax.Array, v: jax.Array, kv_cache: jax.Array, layer: jax.Array,
                    md: AttentionMetadata, *, sm_scale: float, backend: str, block_q: int,
                    pages_per_kv_tile: int) -> tuple[jax.Array, jax.Array]:
    """Write this step's K/V into the paged cache, then causal ragged paged attention.

    q [T, Hq, D], k/v [T, Hkv, D], kv_cache [L, P, Hkv, 2, ps, Dc] with Dc >= D (extra lanes zero-padded here),
    layer: i32 scalar. Returns (o [T, Hq, D], kv_cache). `block_q` is the bucket's q-tile kc.attn_tile_q(T),
    the one md's work plan was built with (prepare_step_input).
    """
    D, Dc = q.shape[-1], kv_cache.shape[-1]
    if Dc != D:
        pad = [(0, 0), (0, 0), (0, Dc - D)]
        q, k, v = jnp.pad(q, pad), jnp.pad(k, pad), jnp.pad(v, pad)
    kv_cache = ref.write_kv(kv_cache, layer, k, v, md.slot_mapping)
    if backend == "reference":
        o = ref.paged_attention(q, kv_cache, layer, md, sm_scale=sm_scale)
    else:
        from flash_jax.ops.pallas.ragged_paged_attention import ragged_paged_attention

        o = ragged_paged_attention(q, kv_cache, layer, md, sm_scale=sm_scale, block_q=min(block_q, q.shape[0]),
                                   pages_per_kv_tile=pages_per_kv_tile, interpret=backend == "interpret")
    return o[..., :D], kv_cache


def add_rms_norm(x: jax.Array, residual: jax.Array, weight: jax.Array, eps: float, *,
                 backend: str) -> tuple[jax.Array, jax.Array]:
    """Fused residual add + RMSNorm: returns (RMSNorm(x + residual) * weight, x + residual)."""
    if backend == "reference":
        return ref.add_rms_norm(x, residual, weight, eps)
    from flash_jax.ops.pallas.rms_norm import add_rms_norm as pallas_add_rms_norm

    return pallas_add_rms_norm(x, residual, weight, eps, interpret=backend == "interpret")


def swiglu(h: jax.Array, w_gate_up: jax.Array, *, backend: str) -> jax.Array:
    """silu(h @ W_gate) * (h @ W_up) for w_gate_up = [W_gate | W_up] ([H, 2I]) -> [T, I]."""
    if backend == "reference":
        return ref.swiglu(h, w_gate_up)
    from flash_jax.ops.pallas.swiglu import swiglu as pallas_swiglu

    return pallas_swiglu(h, w_gate_up, interpret=backend == "interpret")
```
### A.10 Package docstring files

```python
# flash_jax/engine/__init__.py
"""Host-side engine: sequences, block manager, scheduler, model runner, LLM API."""
# flash_jax/model/__init__.py
"""Llama-family model (pure functions over a params pytree) and HF weight loading."""
# flash_jax/ops/pallas/__init__.py
"""Pallas TPU kernels. Each module mirrors a function in flash_jax.ops.reference."""
# flash_jax/testing/__init__.py
"""Test/benchmark support: tiny random HF models and an independent NumPy oracle."""
```
### A.11 `flash_jax/testing/tiny_models.py`

```python
"""Tiny random HF-format models for tests and benchmarks (no network access needed)."""
from __future__ import annotations

import functools
import json
import os
from typing import Any

import numpy as np

# HF config.json dicts. All f32-friendly and small enough for CPU; each exercises one family feature.
TINY_HF_CONFIGS: dict[str, dict[str, Any]] = {
    # GQA G=2, untied lm_head
    "llama": dict(model_type="llama", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=10000.0,
                  max_position_embeddings=1024, tie_word_embeddings=False, hidden_act="silu", eos_token_id=1),
    # QKV bias, tied embeddings, G=4 (single KV head)
    "qwen2": dict(model_type="qwen2", vocab_size=256, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=1, rms_norm_eps=1e-6, rope_theta=1000000.0,
                  max_position_embeddings=1024, tie_word_embeddings=True, hidden_act="silu", eos_token_id=1,
                  use_sliding_window=False, sliding_window=4096),
    # q/k-norm, explicit head_dim with num_heads * head_dim != hidden_size
    "qwen3": dict(model_type="qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
                  rope_theta=1000000.0, max_position_embeddings=1024, tie_word_embeddings=False,
                  hidden_act="silu", eos_token_id=1, attention_bias=False),
    # llama3 rope scaling (all three frequency bands are hit with D=16, theta=1e4, original 64), G=4
    "llama3": dict(model_type="llama", vocab_size=256, hidden_size=128, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=8, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                   rope_theta=10000.0, max_position_embeddings=1024, tie_word_embeddings=False, hidden_act="silu",
                   eos_token_id=[1, 2], rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                                                          high_freq_factor=4.0,
                                                          original_max_position_embeddings=64)),
}


def random_hf_weights(hf: dict[str, Any], seed: int = 0, logit_scale: float = 4.0) -> dict[str, np.ndarray]:
    """HF-named float32 weights ([out, in] Linear layout) for a config dict. Deterministic in `seed`.

    Scales keep activations O(1) and final logits with std ~= logit_scale (sharp: top-2 margins >> f32 noise).
    """
    rng = np.random.default_rng(seed)
    H, I, V, L = hf["hidden_size"], hf["intermediate_size"], hf["vocab_size"], hf["num_hidden_layers"]
    nh = hf["num_attention_heads"]
    nkv = hf.get("num_key_value_heads") or nh
    D = hf.get("head_dim") or H // nh
    mt = hf.get("model_type", "llama")
    qkv_bias = mt == "qwen2" or hf.get("attention_bias", False)
    o_bias = mt != "qwen2" and hf.get("attention_bias", False)

    def lin(out_f: int, in_f: int, gain: float = 1.0) -> np.ndarray:
        return (rng.standard_normal((out_f, in_f)) * gain / np.sqrt(in_f)).astype(np.float32)

    def norm(n: int) -> np.ndarray:
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def bias(n: int) -> np.ndarray:
        return (0.1 * rng.standard_normal(n)).astype(np.float32)

    w: dict[str, np.ndarray] = {"model.embed_tokens.weight": rng.standard_normal((V, H)).astype(np.float32)}
    for i in range(L):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = norm(H)
        w[p + "post_attention_layernorm.weight"] = norm(H)
        w[p + "self_attn.q_proj.weight"] = lin(nh * D, H, 2.0)
        w[p + "self_attn.k_proj.weight"] = lin(nkv * D, H, 2.0)
        w[p + "self_attn.v_proj.weight"] = lin(nkv * D, H)
        w[p + "self_attn.o_proj.weight"] = lin(H, nh * D, 2.0)
        if qkv_bias:
            w[p + "self_attn.q_proj.bias"] = bias(nh * D)
            w[p + "self_attn.k_proj.bias"] = bias(nkv * D)
            w[p + "self_attn.v_proj.bias"] = bias(nkv * D)
        if o_bias:
            w[p + "self_attn.o_proj.bias"] = bias(H)
        if mt == "qwen3":
            w[p + "self_attn.q_norm.weight"] = norm(D)
            w[p + "self_attn.k_norm.weight"] = norm(D)
        w[p + "mlp.gate_proj.weight"] = lin(I, H)
        w[p + "mlp.up_proj.weight"] = lin(I, H)
        w[p + "mlp.down_proj.weight"] = lin(H, I, 2.0)
    w["model.norm.weight"] = norm(H)
    if hf.get("tie_word_embeddings", False):
        w["model.embed_tokens.weight"] *= np.float32(logit_scale / np.sqrt(H))  # embed doubles as lm_head
    else:
        w["lm_head.weight"] = lin(V, H, logit_scale)
    return w


@functools.lru_cache(maxsize=None)
def _cached_weights(name: str, seed: int) -> dict[str, np.ndarray]:
    return random_hf_weights(TINY_HF_CONFIGS[name], seed)


def tiny_model(name: str, seed: int = 0) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """(hf_config_dict, hf_named_float32_weights) for a TINY_HF_CONFIGS entry, cached. Do not mutate the result."""
    return TINY_HF_CONFIGS[name], _cached_weights(name, seed)


def write_hf_checkpoint(path: str, hf: dict[str, Any], weights: dict[str, np.ndarray], *, num_shards: int = 1,
                        dtype: Any = None, generation_config: dict[str, Any] | None = None) -> None:
    """Write config.json (+ generation_config.json), safetensors shard(s) and, if sharded, the index json.

    dtype: None keeps float32; "bfloat16" stores ml_dtypes.bfloat16 tensors (safetensors BF16).
    """
    from safetensors.numpy import save_file

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(hf, f, indent=2)
    if generation_config is not None:
        with open(os.path.join(path, "generation_config.json"), "w") as f:
            json.dump(generation_config, f, indent=2)
    if dtype is not None:
        import ml_dtypes

        np_dtype = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
        weights = {k: v.astype(np_dtype) for k, v in weights.items()}
    names = sorted(weights)
    if num_shards == 1:
        save_file({k: weights[k] for k in names}, os.path.join(path, "model.safetensors"))
        return
    weight_map = {}
    for s in range(num_shards):
        fname = f"model-{s + 1:05d}-of-{num_shards:05d}.safetensors"
        part = names[s::num_shards]
        save_file({k: weights[k] for k in part}, os.path.join(path, fname))
        weight_map.update({k: fname for k in part})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
```
### A.12 `flash_jax/testing/numpy_reference.py`

```python
"""Independent float64 NumPy oracle for Llama / Qwen2 / Qwen3 on raw HF-named weights.

Deliberately shares NO code with flash_jax (it parses the HF config dict itself): no paging, no fusion, no JAX,
full recompute of the whole sequence for every generated token.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _inv_freq(hf: dict[str, Any], d: int) -> np.ndarray:
    theta = float(hf.get("rope_theta", (hf.get("rope_parameters") or {}).get("rope_theta", 10000.0)))
    inv = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
    rs = hf.get("rope_scaling") or hf.get("rope_parameters") or {}
    if rs.get("rope_type", rs.get("type", "default")) == "llama3":  # HF _compute_llama3_parameters
        factor, lo, hi = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
        old = rs["original_max_position_embeddings"]
        low_wl, high_wl = old / lo, old / hi
        wl = 2 * np.pi / inv
        scaled = np.where(wl > low_wl, inv / factor, inv)
        smooth = (old / wl - lo) / (hi - lo)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = ~(wl < high_wl) & ~(wl > low_wl)
        inv = np.where(medium, smoothed, scaled)
    return inv


def _rope(x: np.ndarray, pos: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """HF rotate_half convention. x [n, heads, d]."""
    freqs = pos[:, None].astype(np.float64) * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def dense_logits(w: dict[str, np.ndarray], hf: dict[str, Any], token_ids: Sequence[int]) -> np.ndarray:
    """float64 logits [len(token_ids), vocab] of a full causal forward pass."""
    g = lambda name: np.asarray(w[name], dtype=np.float64)  # noqa: E731
    H, nh = hf["hidden_size"], hf["num_attention_heads"]
    nkv = hf.get("num_key_value_heads") or nh
    d = hf.get("head_dim") or H // nh
    eps = float(hf.get("rms_norm_eps", 1e-6))
    mt = hf.get("model_type", "llama")
    ids = np.asarray(token_ids, dtype=np.int64)
    n = len(ids)
    pos = np.arange(n)
    inv = _inv_freq(hf, d)
    causal = np.tril(np.ones((n, n), dtype=bool))
    x = g("model.embed_tokens.weight")[ids]
    for i in range(hf["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = _rms_norm(x, g(p + "input_layernorm.weight"), eps)
        proj = {}
        for name in ("q", "k", "v"):
            y = h @ g(p + f"self_attn.{name}_proj.weight").T
            if p + f"self_attn.{name}_proj.bias" in w:
                y = y + g(p + f"self_attn.{name}_proj.bias")
            proj[name] = y
        q = proj["q"].reshape(n, nh, d)
        k = proj["k"].reshape(n, nkv, d)
        v = proj["v"].reshape(n, nkv, d)
        if mt == "qwen3":
            q = _rms_norm(q, g(p + "self_attn.q_norm.weight"), eps)
            k = _rms_norm(k, g(p + "self_attn.k_norm.weight"), eps)
        q, k = _rope(q, pos, inv), _rope(k, pos, inv)
        k = np.repeat(k, nh // nkv, axis=1)  # HF repeat_kv: q head j uses kv head j // G
        v = np.repeat(v, nh // nkv, axis=1)
        s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(d)
        s = np.where(causal[None], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", s, v).reshape(n, nh * d) @ g(p + "self_attn.o_proj.weight").T
        if p + "self_attn.o_proj.bias" in w:
            o = o + g(p + "self_attn.o_proj.bias")
        x = x + o
        h = _rms_norm(x, g(p + "post_attention_layernorm.weight"), eps)
        gate = h @ g(p + "mlp.gate_proj.weight").T
        up = h @ g(p + "mlp.up_proj.weight").T
        x = x + (gate / (1.0 + np.exp(-gate)) * up) @ g(p + "mlp.down_proj.weight").T
    x = _rms_norm(x, g("model.norm.weight"), eps)
    head = g("lm_head.weight") if "lm_head.weight" in w else g("model.embed_tokens.weight")
    return x @ head.T


def _top2_margin(row: np.ndarray) -> float:
    top2 = np.partition(row, -2)[-2:]
    return float(top2[1] - top2[0])


def greedy_generate(w: dict[str, np.ndarray], hf: dict[str, Any], prompt: Sequence[int], max_new_tokens: int,
                    eos_token_ids: Sequence[int] = ()) -> tuple[list[int], list[float]]:
    """Free-running greedy decoding (argmax = lowest index among exact ties, like jnp.argmax).

    Returns (generated tokens incl. a final EOS if hit, top-2 logit margin at every generated position).
    """
    toks, margins = list(prompt), []
    for _ in range(max_new_tokens):
        row = dense_logits(w, hf, toks)[-1]
        t = int(np.argmax(row))
        margins.append(_top2_margin(row))
        toks.append(t)
        if t in eos_token_ids:
            break
    return toks[len(prompt):], margins


@dataclass
class GreedyCheck:
    num_tokens: int  # generated tokens checked
    num_near_ties: int  # positions where the engine token != oracle argmax but within tie_tol
    exact: bool  # engine tokens == free-running oracle tokens


def check_greedy(engine_tokens: Sequence[int], w: dict[str, np.ndarray], hf: dict[str, Any],
                 prompt: Sequence[int], *, tie_tol: float = 1e-3) -> GreedyCheck:
    """Tie-aware, teacher-forced check of an engine's greedy continuation.

    One oracle forward over prompt + engine_tokens[:-1]; at every generated position the engine token must be the
    oracle argmax, or within `tie_tol` of the oracle max logit (a near-tie: f32 vs f64 may legitimately flip).
    Raises AssertionError on a real mismatch. Continues past near-ties (teacher forcing), so later positions are
    still verified on the engine's own trajectory.
    """
    engine_tokens = [int(t) for t in engine_tokens]
    if not engine_tokens:
        return GreedyCheck(0, 0, True)
    seq = list(prompt) + engine_tokens[:-1]
    logits = dense_logits(w, hf, seq)[len(prompt) - 1:]
    near = 0
    for i, t in enumerate(engine_tokens):
        row = logits[i]
        best = int(np.argmax(row))
        if t != best:
            gap = float(row[best] - row[t])
            if gap > tie_tol:
                raise AssertionError(f"token {i}: engine={t} oracle={best} logit gap {gap:.3e} > {tie_tol}; "
                                     f"prompt={list(prompt)} engine={engine_tokens}")
            near += 1
    return GreedyCheck(len(engine_tokens), near, near == 0)
```
### A.13 `tests/conftest.py`

```python
"""Shared pytest configuration. Tests run on CPU; Pallas kernels are exercised with backend="interpret"."""
from __future__ import annotations

import jax
import numpy as np
import pytest

from flash_jax.testing.tiny_models import TINY_HF_CONFIGS

ON_TPU = jax.default_backend() == "tpu"
if ON_TPU:  # make TPU f32 XLA matmuls comparable with the CPU/NumPy oracles. The Pallas kernels pin
    # precision=DEFAULT, so `pytest -m tpu` still compiles the production Mosaic programs.
    jax.config.update("jax_default_matmul_precision", "highest")


def pytest_collection_modifyitems(config, items):
    skip_tpu = pytest.mark.skip(reason="needs a TPU")
    for item in items:
        if "tpu" in item.keywords and not ON_TPU:
            item.add_marker(skip_tpu)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(params=sorted(TINY_HF_CONFIGS))
def tiny_name(request) -> str:
    """Parametrizes a test over all tiny model families: llama, llama3, qwen2, qwen3."""
    return request.param
```
### A.14 `flash_jax/ops/tpu_rules.py`

```python
"""TPU (Mosaic) shape rules and on-chip memory estimates, shared by the Pallas wrappers and the ModelRunner.

Pure Python and numpy, no JAX. CPU-side Mosaic lowering checks BlockSpec tiling, but NOT scratch shapes, DMA
slabs, value reshapes, vector layouts or the VMEM/SMEM budgets (DESIGN.md 0.3), so these rules are checked twice:
by ModelRunner.__init__ for every token bucket (before the KV pool is allocated) and by each Pallas wrapper when
interpret is False.
"""
from __future__ import annotations

from typing import Any

import numpy as np

LANES = 128
SMEM_BUDGET_BYTES = 512 << 10  # half of the 1 MiB SMEM of v5e/v6e
# Kernels stay under the DEFAULT scoped-VMEM limit (16 MiB on v5e, 32 MiB on v6e) with 4 MiB left for Mosaic's
# internal scratch. Raising the limit needs CompilerParams(vmem_limit_bytes=...) AND
# LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=N; v1 shrinks tiles instead.
VMEM_BUDGET_BYTES = 12 << 20


def itemsize(dtype: Any) -> int:
    """Bytes per element of "bfloat16" / "float32" (strings) or any numpy/JAX dtype."""
    if isinstance(dtype, str):
        return {"bfloat16": 2, "float32": 4}[dtype]
    return np.dtype(dtype).itemsize


def sublanes(dtype: Any) -> int:
    """Rows of one native (sublane, lane) tile: 8 for 32-bit, 16 for bf16 (two rows packed per sublane)."""
    return 32 // itemsize(dtype)


def attention_smem_bytes(s_pad: int, pages_per_seq: int, nb: int) -> int:
    """Scalar-prefetch bytes of one attention call (layer, cu_q_lens, kv_lens, page_table, work plan)."""
    return 4 * (1 + (s_pad + 1) + s_pad + s_pad * pages_per_seq + 3 * (nb + s_pad) + nb + 1)


def attention_vmem_bytes(*, num_kv_heads: int, rows: int, head_dim: int, block_size: int, pages_per_kv_tile: int,
                         cache_itemsize: int, q_itemsize: int) -> int:
    """VMEM estimate of the attention kernel. rows = block_q * G."""
    kv_buf = 2 * pages_per_kv_tile * num_kv_heads * 2 * block_size * head_dim * cache_itemsize
    q_o = 4 * num_kv_heads * rows * head_dim * q_itemsize  # q and o blocks, double-buffered
    m_l = 2 * num_kv_heads * rows * LANES * 4
    acc = num_kv_heads * rows * head_dim * 4
    values = 1 << 20  # scores, probabilities, masks and K/V tiles live in values
    return kv_buf + q_o + m_l + acc + values


def attention_problems(*, cache_dtype: Any, q_dtype: Any, num_kv_heads: int, q_per_kv: int, head_dim: int,
                       block_size: int, pages_per_kv_tile: int, block_q: int, s_pad: int, pages_per_seq: int,
                       nb: int) -> list[str]:
    """Every TPU rule the attention kernel violates for one call shape (empty list = OK)."""
    out = []
    bkv = pages_per_kv_tile * block_size
    rows = block_q * q_per_kv
    if head_dim % LANES:
        out.append(f"cache head dim {head_dim} must be a multiple of 128 (kv_head_dim_align=None or 128)")
    if block_size % sublanes(cache_dtype):
        out.append(f"block_size {block_size} must be a multiple of {sublanes(cache_dtype)} for {cache_dtype}")
    if bkv % LANES:
        out.append(f"attn_block_kv {bkv} must be a multiple of 128 (it is the lane dim of the scores)")
    if rows % sublanes(q_dtype):
        out.append(f"q tile rows block_q*G = {block_q}*{q_per_kv} must be a multiple of {sublanes(q_dtype)}: "
                   f"change attn_block_q / attn_block_q_small")
    smem = attention_smem_bytes(s_pad, pages_per_seq, nb)
    if smem > SMEM_BUDGET_BYTES:
        out.append(f"attention metadata needs {smem} B of SMEM (> {SMEM_BUDGET_BYTES}): raise block_size or "
                   f"lower max_num_seqs / max_model_len")
    vmem = attention_vmem_bytes(num_kv_heads=num_kv_heads, rows=rows, head_dim=head_dim, block_size=block_size,
                                pages_per_kv_tile=pages_per_kv_tile, cache_itemsize=itemsize(cache_dtype),
                                q_itemsize=itemsize(q_dtype))
    if vmem > VMEM_BUDGET_BYTES:
        out.append(f"attention needs ~{vmem / 2**20:.1f} MiB of VMEM (> {VMEM_BUDGET_BYTES / 2**20:.0f} MiB): "
                   f"lower attn_block_kv or attn_block_q")
    return out


def check_attention_config(model: Any, cfg: Any, kc: Any, head_dim: int) -> None:
    """ModelRunner validation for kc.attention == "pallas": every bucket of cfg (EngineConfig) with the resolved
    KernelConfig kc, ModelConfig model and cache head dim Dc. Raises ValueError listing every violation."""
    problems: list[str] = []
    for t in cfg.token_buckets():
        bq = kc.attn_tile_q(t)
        for p in attention_problems(cache_dtype=cfg.dtype, q_dtype=cfg.dtype, num_kv_heads=model.num_kv_heads,
                                    q_per_kv=model.q_per_kv, head_dim=head_dim, block_size=cfg.block_size,
                                    pages_per_kv_tile=kc.attn_pages_per_kv_tile, block_q=bq,
                                    s_pad=cfg.num_seq_slots(t), pages_per_seq=cfg.pages_per_seq, nb=t // bq):
            if p not in problems:
                problems.append(p)
    if problems:
        raise ValueError("Pallas attention cannot run this configuration on TPU:\n  " + "\n  ".join(problems))
```

---

## Appendix B: normative reference implementations (validated prototypes)

These files belong to the named work packages. The code below is what the §0.3 experiments ran (B.2, B.3,
B.5b and B.6 as revised in the review, §11, and re-validated). Treat its behavior as normative: same algorithm,
same ordering rules, same numerics, and for the kernels the same Mosaic idioms (§5.4.5). Each package may restructure
it, add docstrings, and must add its tests. Where §4/§5 add requirements not shown here, §4/§5 win. Examples
are the loader, `param_shapes`/`init_params` and the runner, which are specified in prose.

### B.1 `flash_jax/engine/block_manager.py` (WP4)

```python
"""Paged KV block allocator with hash-based prefix caching (ref counts + LRU eviction of cached free blocks)."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Sequence as Seq

import numpy as np

from flash_jax.config import cdiv
from flash_jax.engine.sequence import Sequence, SeqStatus

ROOT_HASH: bytes = hashlib.sha256(b"flash-jax/prefix-cache/v1").digest()


def hash_block(parent: bytes, tokens: Seq[int]) -> bytes:
    """Chained block hash: equal tokens after a different prefix hash differently."""
    return hashlib.sha256(parent + np.asarray(tokens, dtype=np.int32).tobytes()).digest()


class BlockManager:
    """Owns KV blocks [0, num_blocks). A block is either referenced (ref_count > 0) or in `free_blocks`.

    `free_blocks` is ONE LRU (front = evicted first) holding every ref-0 block, sealed or not; a sealed (hashed)
    free block stays reusable through `hash_to_block` until it is popped for reuse (= evicted).
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_counts: list[int] = [0] * num_blocks
        self.block_hash: list[bytes | None] = [None] * num_blocks
        self.hash_to_block: dict[bytes, int] = {}
        self.free_blocks: OrderedDict[int, None] = OrderedDict((b, None) for b in range(num_blocks))
        self.num_query_tokens = 0  # prompt(+output) tokens of admitted sequences
        self.num_hit_tokens = 0  # of which served from the prefix cache
        self.num_evictions = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _hash_chain(self, seq: Sequence, n_blocks: int) -> list[bytes]:
        """Hashes of the first n_blocks FULL blocks of seq.token_ids (memoized on seq.block_hashes)."""
        bs = self.block_size
        while len(seq.block_hashes) < n_blocks:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[-1] if i else ROOT_HASH
            seq.block_hashes.append(hash_block(parent, seq.token_ids[i * bs:(i + 1) * bs]))
        return seq.block_hashes[:n_blocks]

    def find_cached_prefix(self, seq: Sequence) -> list[int]:
        """Longest run of cached full blocks, capped so that >= 1 token is always computed. No mutation."""
        if not self.enable_prefix_caching:
            return []
        hits: list[int] = []
        for h in self._hash_chain(seq, (seq.num_tokens - 1) // self.block_size):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hits.append(b)
        return hits

    def can_allocate(self, seq: Sequence, cached: list[int], num_new_tokens: int) -> bool:
        need = cdiv(len(cached) * self.block_size + num_new_tokens, self.block_size) - len(cached)
        revived = sum(1 for b in cached if self.ref_counts[b] == 0)  # hits that currently sit in free_blocks
        return len(self.free_blocks) - revived >= need

    def allocate(self, seq: Sequence, cached: list[int], num_new_tokens: int) -> None:
        """Admit a WAITING seq: take refs on the prefix hits, then fresh blocks for num_new_tokens."""
        assert not seq.block_table and seq.num_computed_tokens == 0
        assert self.can_allocate(seq, cached, num_new_tokens)
        for b in cached:  # touch hits BEFORE popping fresh blocks (a pop could otherwise evict a hit)
            if self.ref_counts[b] == 0:
                del self.free_blocks[b]
            self.ref_counts[b] += 1
            seq.block_table.append(b)
        seq.num_computed_tokens = seq.num_cached_tokens = len(cached) * self.block_size
        seq.num_sealed_blocks = len(cached)
        self.num_query_tokens += seq.num_tokens
        self.num_hit_tokens += seq.num_cached_tokens
        self.append_slots(seq, num_new_tokens)

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        need = cdiv(seq.num_computed_tokens + num_new_tokens, self.block_size) - len(seq.block_table)
        return len(self.free_blocks) >= need

    def append_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        """Grow seq.block_table to cover positions [0, num_computed_tokens + num_new_tokens)."""
        need = cdiv(seq.num_computed_tokens + num_new_tokens, self.block_size) - len(seq.block_table)
        assert need <= len(self.free_blocks)
        for _ in range(max(0, need)):
            seq.block_table.append(self._pop_free())

    def _pop_free(self) -> int:
        b, _ = self.free_blocks.popitem(last=False)  # LRU front
        h = self.block_hash[b]
        if h is not None:  # evict from the prefix cache
            del self.hash_to_block[h]
            self.block_hash[b] = None
            self.num_evictions += 1
        self.ref_counts[b] = 1
        return b

    def free(self, seq: Sequence) -> None:
        """Drop seq's refs (tail first). Hashed blocks stay cached; unhashed ones go to the LRU front."""
        for b in reversed(seq.block_table):
            self.ref_counts[b] -= 1
            assert self.ref_counts[b] >= 0
            if self.ref_counts[b] == 0:
                self.free_blocks[b] = None  # MRU end: a seq's head blocks end up newest, evicted last
                if self.block_hash[b] is None:
                    self.free_blocks.move_to_end(b, last=False)
        seq.block_table = []
        seq.num_sealed_blocks = 0

    def seal_computed_blocks(self, seq: Sequence) -> None:
        """Publish newly FULL blocks whose KV is computed. Called after every step, before freeing."""
        if not self.enable_prefix_caching:
            return
        n_full = seq.num_computed_tokens // self.block_size
        hashes = self._hash_chain(seq, n_full)
        for i in range(seq.num_sealed_blocks, n_full):
            b, h = seq.block_table[i], hashes[i]
            if h not in self.hash_to_block and self.block_hash[b] is None:
                self.hash_to_block[h] = b
                self.block_hash[b] = h
            # else: an identical block was published concurrently; this one stays private (unhashed)
        seq.num_sealed_blocks = max(seq.num_sealed_blocks, n_full)

    def check_invariants(self, live: Iterable[Sequence]) -> None:
        """I1-I6 (see DESIGN.md). `live` = every WAITING and RUNNING sequence."""
        counts = [0] * self.num_blocks
        for s in live:
            assert len(set(s.block_table)) == len(s.block_table), f"I4 duplicate block in {s}"
            for b in s.block_table:
                counts[b] += 1
            if s.status is SeqStatus.RUNNING:
                assert len(s.block_table) == cdiv(s.num_computed_tokens, self.block_size), f"I5 {s}"
                if self.enable_prefix_caching:
                    chain = self._hash_chain(s, s.num_sealed_blocks)
                    for i in range(s.num_sealed_blocks):
                        h = self.block_hash[s.block_table[i]]
                        assert h is None or h == chain[i], f"I6 hash mismatch in {s}"
            else:
                assert not s.block_table, f"I4 waiting seq holds blocks: {s}"
        assert counts == self.ref_counts, "I1 ref counts"
        for b in range(self.num_blocks):
            assert (b in self.free_blocks) == (self.ref_counts[b] == 0), f"I2 block {b}"
            h = self.block_hash[b]
            assert h is None or self.hash_to_block.get(h) == b, f"I3 block {b}"
        assert len(self.hash_to_block) == sum(h is not None for h in self.block_hash), "I3 bijection"
```
### B.2 `flash_jax/engine/scheduler.py` (WP4)

```python
"""Continuous-batching scheduler: token budget, decode priority, chunked prefill, recompute preemption."""
from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Sequence as Seq

from flash_jax.batch import ScheduledSeq
from flash_jax.config import EngineConfig, cdiv
from flash_jax.engine.block_manager import BlockManager
from flash_jax.engine.sequence import Sequence, SeqStatus


class EngineDeadlock(RuntimeError):
    """schedule() found no runnable work although requests are pending (a bug or an undersized KV pool)."""


class Scheduler:
    def __init__(self, config: EngineConfig, num_kv_blocks: int, eos_token_ids: Seq[int] = ()) -> None:
        self.config = config
        self.block_manager = BlockManager(num_kv_blocks, config.block_size, config.enable_prefix_caching)
        self.eos_token_ids = frozenset(int(t) for t in eos_token_ids)
        self.waiting: deque[Sequence] = deque()  # always sorted by seq_id (= arrival order)
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        assert seq.status is SeqStatus.WAITING and (not self.waiting or self.waiting[-1].seq_id < seq.seq_id)
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _chunk(self, remaining: int, budget: int) -> int:
        return min(remaining, budget, self.config.max_prefill_chunk or budget)

    def schedule(self) -> list[ScheduledSeq]:
        """Pick this step's work. Order: running decodes, running partial prefills, then FIFO admissions."""
        cfg, bm = self.config, self.block_manager
        budget = cfg.max_num_batched_tokens
        out: list[ScheduledSeq] = []
        scheduled: set[int] = set()
        preempted = False
        for decodes in (True, False):
            for seq in sorted(self.running, key=lambda s: s.seq_id):
                if budget == 0:
                    break
                if seq.seq_id in scheduled or seq.status is not SeqStatus.RUNNING:
                    continue
                if (seq.num_remaining == 1) != decodes:
                    continue
                n = self._chunk(seq.num_remaining, budget)
                while not bm.can_append(seq, n):
                    # victim: latest-arrived running seq not scheduled this step (possibly seq itself)
                    victim = max((s for s in self.running if s.seq_id not in scheduled), key=lambda s: s.seq_id)
                    self._preempt(victim)
                    preempted = True
                    if victim is seq:
                        break
                else:
                    bm.append_slots(seq, n)
                    out.append(ScheduledSeq(seq, seq.num_computed_tokens, n, n == seq.num_remaining))
                    scheduled.add(seq.seq_id)
                    budget -= n
        # Admission is skipped in any step that preempted (anti-thrash). Head-of-line blocking (FIFO fairness).
        while not preempted and self.waiting and budget > 0 and len(self.running) < cfg.max_num_seqs:
            seq = self.waiting[0]
            cached = bm.find_cached_prefix(seq)
            n = self._chunk(seq.num_tokens - len(cached) * cfg.block_size, budget)
            if not bm.can_allocate(seq, cached, n):
                break
            self.waiting.popleft()
            bm.allocate(seq, cached, n)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            out.append(ScheduledSeq(seq, seq.num_computed_tokens, n, n == seq.num_remaining))
            scheduled.add(seq.seq_id)
            budget -= n
        if not out and self.has_unfinished():
            raise EngineDeadlock(f"nothing schedulable: waiting={list(self.waiting)} running={self.running}")
        return out

    def _preempt(self, seq: Sequence) -> None:
        """Recompute-style preemption. Sealed blocks stay in the prefix cache, so re-admission usually hits."""
        self.block_manager.free(seq)
        seq.num_computed_tokens = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.running.remove(seq)
        keys = [s.seq_id for s in self.waiting]
        self.waiting.insert(bisect.bisect_left(keys, seq.seq_id), seq)

    def update(self, batch: list[ScheduledSeq], sampled_tokens: Seq[int]) -> list[Sequence]:
        """Apply a finished step (in batch order). Returns sequences that finished in this step."""
        finished: list[Sequence] = []
        for item, tok in zip(batch, sampled_tokens, strict=True):
            seq = item.seq
            seq.num_computed_tokens += item.num_new_tokens
            self.block_manager.seal_computed_blocks(seq)
            if not item.samples:
                continue  # partial prefill chunk: the sampled token is discarded
            tok = int(tok)
            seq.token_ids.append(tok)
            if (tok in self.eos_token_ids and not seq.params.ignore_eos) or tok in seq.params.stop_token_ids:
                seq.finish_reason = "stop"
            elif seq.num_output_tokens >= seq.max_tokens or seq.num_tokens >= self.config.max_model_len:
                seq.finish_reason = "length"
            if seq.finish_reason is not None:
                seq.status = SeqStatus.FINISHED
                self.block_manager.free(seq)
                self.running.remove(seq)
                finished.append(seq)
        return finished

    def abort(self, seq_id: int) -> Sequence | None:
        for seq in [*self.waiting, *self.running]:
            if seq.seq_id == seq_id:
                if seq.status is SeqStatus.RUNNING:
                    self.running.remove(seq)
                    self.block_manager.free(seq)
                else:
                    self.waiting.remove(seq)
                seq.status, seq.finish_reason = SeqStatus.FINISHED, "abort"
                return seq
        return None

    def check_batch(self, batch: list[ScheduledSeq]) -> None:
        """S1-S4 + I7 (see DESIGN.md), run by the engine when debug_checks=True."""
        cfg, bm = self.config, self.block_manager
        assert sum(b.num_new_tokens for b in batch) <= cfg.max_num_batched_tokens, "S1 budget"
        assert len(batch) <= cfg.max_num_seqs, "S1 seqs"
        assert len({b.seq.seq_id for b in batch}) == len(batch), "S2 duplicate"
        for b in batch:
            s = b.seq
            assert s.status is SeqStatus.RUNNING
            assert b.start_pos == s.num_computed_tokens and 1 <= b.num_new_tokens <= s.num_tokens - b.start_pos, "S3"
            assert b.samples == (b.start_pos + b.num_new_tokens == s.num_tokens), "S4"
            end = b.start_pos + b.num_new_tokens
            assert len(s.block_table) == cdiv(end, cfg.block_size), "S3 block table"
            for blk in {s.block_table[p // cfg.block_size] for p in range(b.start_pos, end)}:
                assert bm.ref_counts[blk] == 1 and bm.block_hash[blk] is None, f"I7 write into shared block {blk}"
```
### B.3 `flash_jax/ops/pallas/ragged_paged_attention.py` (WP2)

```python
"""Ragged paged attention (Pallas TPU): one call per layer serves a flat mix of prefill chunks and decodes.

Cache [L, P, Hkv, 2, bs, Dc] stays in HBM (pl.ANY); pages are DMA'd into a double-buffered VMEM tile driven by the
host-built work plan in AttentionMetadata (see DESIGN.md 5.4). Read-only on the cache.

In-kernel idioms follow JAX's bundled TPU kernels (flash_attention.py, ragged_paged_attention): masks come from
full-shape 2-D iotas (no [N, 1] bool broadcasts), selects run in f32 (v5e has no bf16 VPU), m/l stay
lane-replicated [rows, 128] and are widened with jnp.tile (no width-1 lane slices), and every matmul pins
precision=DEFAULT so a global jax_default_matmul_precision cannot change the Mosaic program.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flash_jax.batch import AttentionMetadata
from flash_jax.ops import tpu_rules
from flash_jax.ops.common import MASK_VALUE, interpret_params

_DEFAULT = lax.Precision.DEFAULT


def _lanes(x: jax.Array, n: int) -> jax.Array:
    """Widen a lane-replicated [rows, 128] f32 value to [rows, n]. On TPU n % 128 == 0 (tpu_rules), so this is a
    plain jnp.tile; other widths only occur in interpret mode."""
    if n % 128 == 0:
        return x if n == 128 else jnp.tile(x, (1, n // 128))
    return jnp.tile(x, (1, pl.cdiv(n, 128)))[:, :n]


def _kernel(layer_ref, cu_ref, kvlen_ref, pt_ref, pseq_ref, pntiles_ref, poff_ref, tstart_ref,  # SMEM
            q_ref, cache_hbm,  # VMEM block [Hkv, bq*G, D], HBM [L, P, Hkv, 2, ps, D]
            o_ref,  # VMEM block [Hkv, bq*G, D]
            kv_buf, sems, m_ref, l_ref, acc_ref,  # scratch
            *, sm_scale: float, block_q: int, pages_per_seq: int):
    hkv, rows, d = q_ref.shape
    g = rows // block_q
    _, ppb, _, _, ps, _ = kv_buf.shape
    bkv = ppb * ps
    b = pl.program_id(0)
    layer = layer_ref[0]
    total_pairs = tstart_ref[pl.num_programs(0)]

    def page_copy(p, j, slot, i):
        s = pseq_ref[p]
        page = pt_ref[s * pages_per_seq + j * ppb + i]
        return pltpu.make_async_copy(cache_hbm.at[layer, page], kv_buf.at[slot, i], sems.at[slot])

    def for_valid_pages(p, j, fn):
        n_pages = pl.cdiv(kvlen_ref[pseq_ref[p]], ps)
        for i in range(ppb):  # static unroll; start and wait use the SAME predicate
            pl.when(j * ppb + i < n_pages)(functools.partial(fn, i))

    def start_fetch(p, j, slot):
        for_valid_pages(p, j, lambda i: page_copy(p, j, slot, i).start())

    def wait_fetch(p, j, slot):
        for_valid_pages(p, j, lambda i: page_copy(p, j, slot, i).wait())

    @pl.when((b == 0) & (total_pairs > 0))
    def _prologue():
        start_fetch(0, 0, 0)  # global kv-tile 0 -> slot 0

    m_ref[...] = jnp.full(m_ref.shape, MASK_VALUE, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    def pair_body(p, carry):
        s, nkv, base = pseq_ref[p], pntiles_ref[p], poff_ref[p]
        q_lo, q_hi, kv_len = cu_ref[s], cu_ref[s + 1], kvlen_ref[s]
        # full-shape [rows, bkv] index grids: row r of the tile is token b*bq + r // G
        row_tok = b * block_q + lax.broadcasted_iota(jnp.int32, (rows, bkv), 0) // g
        row_ok = (row_tok >= q_lo) & (row_tok < q_hi)
        row_pos = kv_len - (q_hi - q_lo) + (row_tok - q_lo)

        def kv_body(j, carry):
            slot = (base + j) % 2
            last = j + 1 == nkv
            nxt_p = jnp.where(last, p + 1, p)
            nxt_j = jnp.where(last, 0, j + 1)

            @pl.when(nxt_p < total_pairs)
            def _():
                start_fetch(nxt_p, nxt_j, 1 - slot)

            wait_fetch(p, j, slot)
            col = j * bkv + lax.broadcasted_iota(jnp.int32, (rows, bkv), 1)
            mask = row_ok & (col <= row_pos)  # [rows, bkv]; mask implies col < kv_len
            v_ok = j * bkv + lax.broadcasted_iota(jnp.int32, (bkv, d), 0) < kv_len  # [bkv, D]
            for h in range(hkv):  # static unroll over KV heads
                # K is not masked: a NaN/stale K row only poisons its own score column, which `mask` replaces.
                k = kv_buf[slot, :, h, 0].reshape(bkv, d)  # value reshape: [ppb, ps, D] -> [bkv, D]
                # V must be zeroed (0 * NaN = NaN in P@V); the select runs in f32.
                v = kv_buf[slot, :, h, 1].reshape(bkv, d)
                v = jnp.where(v_ok, v.astype(jnp.float32), 0.0).astype(kv_buf.dtype)
                sc = lax.dot_general(q_ref[h], k, (((1,), (1,)), ((), ())), precision=_DEFAULT,
                                     preferred_element_type=jnp.float32)  # [rows, bkv]
                sc = jnp.where(mask, sc * sm_scale, MASK_VALUE)
                m_prev = m_ref[h]  # [rows, 128], lane-replicated
                # rows of other sequences are fully masked: max(m_prev, MASK_VALUE) == m_prev, alpha == 1, p == 0
                m_next = jnp.maximum(m_prev, sc.max(axis=1, keepdims=True))
                pexp = jnp.where(mask, jnp.exp(sc - _lanes(m_next, bkv)), 0.0)
                alpha = jnp.exp(m_prev - m_next)
                l_ref[h] = alpha * l_ref[h] + pexp.sum(axis=1, keepdims=True)
                m_ref[h] = m_next
                pv = jnp.dot(pexp.astype(kv_buf.dtype), v, precision=_DEFAULT, preferred_element_type=jnp.float32)
                acc_ref[h] = _lanes(alpha, d) * acc_ref[h] + pv
            return carry

        lax.fori_loop(0, nkv, kv_body, 0)
        return carry

    lax.fori_loop(tstart_ref[b], tstart_ref[b + 1], pair_body, 0)
    for h in range(hkv):
        l = _lanes(l_ref[h], d)
        o_ref[h] = (acc_ref[h] / jnp.where(l == 0.0, 1.0, l)).astype(o_ref.dtype)  # never-valid rows -> exactly 0


def ragged_paged_attention(q: jax.Array, kv_cache: jax.Array, layer: jax.Array, md: AttentionMetadata, *,
                           sm_scale: float, block_q: int, pages_per_kv_tile: int,
                           interpret: bool | pltpu.InterpretParams = False) -> jax.Array:
    T, Hq, D = q.shape
    L, P, Hkv, two, ps, Dc = kv_cache.shape
    assert two == 2 and Dc == D and Hq % Hkv == 0 and T % block_q == 0
    G = Hq // Hkv
    nb = T // block_q
    S = md.kv_lens.shape[0]
    MP = md.page_table.shape[0] // S
    assert md.tile_pair_start.shape == (nb + 1,), (md.tile_pair_start.shape, nb)
    ppb = pages_per_kv_tile
    if interpret is False:
        problems = tpu_rules.attention_problems(
            cache_dtype=kv_cache.dtype, q_dtype=q.dtype, num_kv_heads=Hkv, q_per_kv=G, head_dim=D, block_size=ps,
            pages_per_kv_tile=ppb, block_q=block_q, s_pad=S, pages_per_seq=MP, nb=nb)
        if problems:
            raise ValueError("ragged_paged_attention on TPU: " + "; ".join(problems))
    if interpret is True:
        interpret = interpret_params()
    q_hm = q.reshape(T, Hkv, G, D).transpose(1, 0, 2, 3).reshape(Hkv, T * G, D)
    prefetch = (jnp.reshape(layer, (1,)).astype(jnp.int32), md.cu_q_lens, md.kv_lens, md.page_table,
                md.pair_seq, md.pair_num_kv_tiles, md.pair_kv_offset, md.tile_pair_start)
    qspec = pl.BlockSpec((Hkv, block_q * G, D), lambda b, *_: (0, b, 0))
    out = pl.pallas_call(
        functools.partial(_kernel, sm_scale=sm_scale, block_q=block_q, pages_per_seq=MP),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            grid=(nb,),
            in_specs=[qspec, pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=qspec,
            scratch_shapes=[
                pltpu.VMEM((2, ppb, Hkv, 2, ps, D), kv_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM((Hkv, block_q * G, 128), jnp.float32),
                pltpu.VMEM((Hkv, block_q * G, 128), jnp.float32),
                pltpu.VMEM((Hkv, block_q * G, D), jnp.float32),
            ]),
        out_shape=jax.ShapeDtypeStruct((Hkv, T * G, D), q.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )(*prefetch, q_hm, kv_cache)
    return out.reshape(Hkv, T, G, D).transpose(1, 0, 2, 3).reshape(T, Hq, D)
```
### B.4 `flash_jax/ops/sampler.py` (WP5)

```python
"""On-device sampling: exact greedy, temperature, top-k, top-p; per-request, batch-position-independent RNG."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def sample(logits: jax.Array, temperature: jax.Array, top_k: jax.Array, top_p: jax.Array, seeds: jax.Array,
           gen_idx: jax.Array, *, max_candidates: int) -> jax.Array:
    """logits f32[S, V]; temperature/top_p f32[S]; top_k/seeds/gen_idx i32[S] -> tokens i32[S]."""
    S, V = logits.shape
    greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def stochastic(_):
        K = min(max_candidates, V)
        base = jax.random.key(0)
        keys = jax.vmap(lambda s, g: jax.random.fold_in(jax.random.fold_in(base, s), g))(seeds, gen_idx)
        x = logits / jnp.where(temperature > 0, temperature, 1.0)[:, None]
        # plain temperature sampling: exact Gumbel-max over the full vocab
        g_full = jax.vmap(lambda k: jax.random.gumbel(jax.random.fold_in(k, 0), (V,), jnp.float32))(keys)
        tok_full = jnp.argmax(x + g_full, axis=-1).astype(jnp.int32)
        # top-k / top-p: restrict to the K best candidates
        vals, idx = lax.top_k(x, K)  # descending
        rank = jnp.arange(K, dtype=jnp.int32)[None, :]
        keep_k = rank < jnp.where(top_k > 0, jnp.minimum(top_k, K), K)[:, None]
        lse = jnp.where(top_k > 0, jax.nn.logsumexp(jnp.where(keep_k, vals, -jnp.inf), axis=-1),
                        jax.nn.logsumexp(x, axis=-1))  # top-k renormalizes (HF order); else exact full-vocab mass
        probs = jnp.where(keep_k, jnp.exp(vals - lse[:, None]), 0.0)
        keep = keep_k & ((jnp.cumsum(probs, axis=-1) - probs) < top_p[:, None])  # rank 0 always kept
        g_c = jax.vmap(lambda k: jax.random.gumbel(jax.random.fold_in(k, 1), (K,), jnp.float32))(keys)
        choice = jnp.argmax(jnp.where(keep, vals + g_c, -jnp.inf), axis=-1)
        tok_c = jnp.take_along_axis(idx, choice[:, None], axis=-1)[:, 0].astype(jnp.int32)
        filtered = (top_k > 0) | (top_p < 1.0)
        return jnp.where(temperature > 0, jnp.where(filtered, tok_c, tok_full), greedy)

    return lax.cond(jnp.all(temperature <= 0), lambda _: greedy, stochastic, None)
```
### B.5a `flash_jax/model/rope.py` (WP3)

```python
"""Rotary position embedding (HF rotate_half convention) with optional llama3 frequency scaling."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from flash_jax.config import ModelConfig


def rope_inv_freq(cfg: ModelConfig) -> np.ndarray:
    """f32[D/2] inverse frequencies (computed in float64; a trace-time constant)."""
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    if cfg.rope_scaling is not None:  # HF _compute_llama3_parameters
        factor, low, high, old = cfg.rope_scaling
        wl = 2 * math.pi / inv
        scaled = np.where(wl > old / low, inv / factor, inv)
        smooth = (old / wl - low) / (high - low)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = ~(wl < old / high) & ~(wl > old / low)
        inv = np.where(medium, smoothed, scaled)
    return inv.astype(np.float32)


def rope_cos_sin(positions: jax.Array, inv_freq: np.ndarray) -> tuple[jax.Array, jax.Array]:
    freqs = positions.astype(jnp.float32)[:, None] * jnp.asarray(inv_freq)[None, :]
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """x [T, heads, D]; cos/sin [T, D/2]. out = x*cos + rotate_half(x)*sin, computed in f32."""
    half = x.shape[-1] // 2
    xf = x.astype(jnp.float32)
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
```
### B.5b `flash_jax/model/llama.py`: forward and compute_logits (WP3; add param_shapes and init_params per §4.3)

```python
"""Llama-family decoder (Llama 2/3, Qwen2, Qwen3) as pure functions over a stacked params pytree."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from flash_jax import ops
from flash_jax.batch import StepInput
from flash_jax.config import KernelConfig, ModelConfig
from flash_jax.model.rope import apply_rope, rope_cos_sin, rope_inv_freq

Params = dict[str, Any]


def forward(params: Params, kv_cache: jax.Array, inp: StepInput, cfg: ModelConfig,
            kc: KernelConfig) -> tuple[jax.Array, jax.Array]:
    T = inp.input_ids.shape[0]
    Hkv, G, D, eps = cfg.num_kv_heads, cfg.q_per_kv, cfg.head_dim, cfg.rms_norm_eps
    x = params["embed"][inp.input_ids]
    cos, sin = rope_cos_sin(inp.positions, rope_inv_freq(cfg))  # once per step, shared by all layers

    def layer(carry, lp):
        x, res, kv_cache, li = carry
        h, res = ops.add_rms_norm(x, res, lp["attn_norm"], eps, backend=kc.norm)
        qkv = h @ lp["qkv"]  # ONE matmul for q, k and v
        if "qkv_bias" in lp:
            qkv = qkv + lp["qkv_bias"]
        qkv = qkv.reshape(T, Hkv, G + 2, D)
        q = qkv[:, :, :G].reshape(T, Hkv * G, D)
        k, v = qkv[:, :, G], qkv[:, :, G + 1]
        if cfg.qk_norm:
            q, k = ops.rms_norm(q, lp["q_norm"], eps), ops.rms_norm(k, lp["k_norm"], eps)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o, kv_cache = ops.paged_attention(q, k, v, kv_cache, li, inp.attn, sm_scale=D ** -0.5,
                                          backend=kc.attention, block_q=kc.attn_tile_q(T),
                                          pages_per_kv_tile=kc.attn_pages_per_kv_tile)
        x = o.reshape(T, Hkv * G * D) @ lp["o"]
        if "o_bias" in lp:
            x = x + lp["o_bias"]
        h, res = ops.add_rms_norm(x, res, lp["mlp_norm"], eps, backend=kc.norm)
        x = ops.swiglu(h, lp["gate_up"], backend=kc.mlp) @ lp["down"]
        return (x, res, kv_cache, li + 1), None

    init = (x, jnp.zeros_like(x), kv_cache, jnp.int32(0))
    (x, res, kv_cache, _), _ = jax.lax.scan(layer, init, params["layers"])
    idx = inp.logits_indices  # gather BEFORE the final norm and the LM head
    h, _ = ops.add_rms_norm(x[idx], res[idx], params["final_norm"], eps, backend=kc.norm)
    return h, kv_cache


def compute_logits(params: Params, hidden: jax.Array, cfg: ModelConfig) -> jax.Array:
    if cfg.tie_word_embeddings:
        return jnp.einsum("sh,vh->sv", hidden, params["embed"], preferred_element_type=jnp.float32)
    return jnp.dot(hidden, params["lm_head"], preferred_element_type=jnp.float32)
```
### B.6a `flash_jax/ops/pallas/rms_norm.py` (WP2)

```python
"""Fused residual-add + RMSNorm (Pallas TPU), residual updated in place (input_output_aliases={1: 1})."""
from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flash_jax.ops import reference as ref
from flash_jax.ops import tpu_rules
from flash_jax.ops.common import interpret_params


def _kernel(x_ref, r_ref, w_ref, y_ref, r_out_ref, *, eps: float):
    r = (x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)).astype(r_out_ref.dtype)
    r_out_ref[...] = r
    rf = r.astype(jnp.float32)
    y = rf * lax.rsqrt(jnp.mean(rf * rf, axis=-1, keepdims=True) + eps)
    y_ref[...] = y.astype(y_ref.dtype) * w_ref[...].astype(y_ref.dtype)


def vmem_bytes(bt: int, h: int, itemsize: int) -> int:
    """x, residual, y, new_residual (bt, H) blocks double-buffered + ~3 f32 (bt, H) temporaries."""
    return 8 * bt * h * itemsize + 3 * bt * h * 4


def row_tile(t: int, h: int, dtype) -> int | None:
    """Largest legal row tile that fits tpu_rules.VMEM_BUDGET_BYTES: T itself (a full dim is always legal) if
    T <= 1024, else a power-of-two divisor of T that is a multiple of the sublane tile. None if nothing fits."""
    sub, isz = tpu_rules.sublanes(dtype), tpu_rules.itemsize(dtype)
    cands = ([t] if t <= 1024 else []) + [b for b in (1024, 512, 256, 128, 64, 32, 16, 8)
                                          if b < t and t % b == 0 and b % sub == 0]
    return next((bt for bt in cands if vmem_bytes(bt, h, isz) <= tpu_rules.VMEM_BUDGET_BYTES), None)


def add_rms_norm(x: jax.Array, residual: jax.Array, weight: jax.Array, eps: float, *,
                 interpret: bool | pltpu.InterpretParams = False) -> tuple[jax.Array, jax.Array]:
    T, H = x.shape
    bt = row_tile(T, H, x.dtype)
    if bt is None:
        warnings.warn(f"add_rms_norm: no row tile of (T={T}, H={H}) fits VMEM; using the XLA reference")
        return ref.add_rms_norm(x, residual, weight, eps)
    assert T % bt == 0
    spec = pl.BlockSpec((bt, H), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        grid=(T // bt,),
        in_specs=[spec, spec, pl.BlockSpec((1, H), lambda i: (0, 0))],
        out_specs=[spec, spec],
        out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype)],
        input_output_aliases={1: 1},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret_params() if interpret is True else interpret,
    )(x, residual.astype(x.dtype), weight.reshape(1, H))
```
### B.6b `flash_jax/ops/pallas/swiglu.py` (WP2)

```python
"""Fused gate_up matmul + SiLU*mul epilogue (Pallas TPU). The [T, 2I] intermediate never reaches HBM."""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flash_jax.ops import reference as ref
from flash_jax.ops import tpu_rules
from flash_jax.ops.common import interpret_params

_DEFAULT = lax.Precision.DEFAULT  # pinned: a global jax_default_matmul_precision must not change the kernel


def _kernel(h_ref, wg_ref, wu_ref, o_ref, acc_g, acc_u):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_g[...] = jnp.zeros(acc_g.shape, jnp.float32)
        acc_u[...] = jnp.zeros(acc_u.shape, jnp.float32)

    h = h_ref[...]
    acc_g[...] += jnp.dot(h, wg_ref[...], precision=_DEFAULT, preferred_element_type=jnp.float32)
    acc_u[...] += jnp.dot(h, wu_ref[...], precision=_DEFAULT, preferred_element_type=jnp.float32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = (jax.nn.silu(acc_g[...]) * acc_u[...]).astype(o_ref.dtype)


def vmem_bytes(bt: int, bh: int, bi: int, itemsize: int) -> int:
    """h, gate, up and out blocks double-buffered + 2 f32 accumulators + 2 f32 matmul results."""
    return 2 * (bt * bh + 2 * bh * bi + bt * bi) * itemsize + 4 * bt * bi * 4


def _divisor_tiles(n: int, full_max: int, prefs: tuple[int, ...]) -> list[int]:
    return ([n] if n <= full_max else []) + [p for p in prefs if p < n and n % p == 0]


def tiles(t: int, h: int, i: int, dtype) -> tuple[int, int, int] | None:
    """(bt, bh, bi). bi: I if I <= 512 else the first of 512/256/128 dividing I. Then the LARGEST row tile bt
    (T itself if <= 1024, else a power-of-two divisor that is a multiple of the sublane tile) and, for it, the largest
    bh (H if <= 512, else 512/256/128 dividing H) whose VMEM estimate fits tpu_rules.VMEM_BUDGET_BYTES. A large bt
    matters most: the whole [H, 2I] weight is streamed T/bt times. None if nothing fits."""
    sub, isz = tpu_rules.sublanes(dtype), tpu_rules.itemsize(dtype)
    bis = _divisor_tiles(i, 512, (512, 256, 128))
    if not bis:
        return None
    bi = bis[0]
    bts = [b for b in _divisor_tiles(t, 1024, (1024, 512, 256, 128, 64, 32, 16, 8)) if b == t or b % sub == 0]
    for bt in bts:
        for bh in _divisor_tiles(h, 512, (512, 256, 128)):
            if vmem_bytes(bt, bh, bi, isz) <= tpu_rules.VMEM_BUDGET_BYTES:
                return bt, bh, bi
    return None


def swiglu(h: jax.Array, w_gate_up: jax.Array, *, interpret: bool | pltpu.InterpretParams = False) -> jax.Array:
    T, H = h.shape
    I = w_gate_up.shape[1] // 2
    tl = tiles(T, H, I, h.dtype)
    tpu_ok = tl is not None and tl[2] % 128 == 0 and (tl[1] % 128 == 0 or tl[1] == H)
    if tl is None or (interpret is False and not tpu_ok):
        # warnings' default filter shows this once per call site and message; pytest.warns always sees it
        warnings.warn(f"swiglu: shape (T={T}, H={H}, I={I}) not tileable for TPU; using the XLA reference")
        return ref.swiglu(h, w_gate_up)
    bt, bh, bi = tl
    assert T % bt == 0 and H % bh == 0 and I % bi == 0
    nj = I // bi
    return pl.pallas_call(
        _kernel,
        grid=(T // bt, nj, H // bh),
        in_specs=[pl.BlockSpec((bt, bh), lambda i, j, k: (i, k)),
                  pl.BlockSpec((bh, bi), lambda i, j, k: (k, j)),  # gate columns
                  pl.BlockSpec((bh, bi), lambda i, j, k: (k, j + nj))],  # up columns (same array)
        out_specs=pl.BlockSpec((bt, bi), lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct((T, I), h.dtype),
        scratch_shapes=[pltpu.VMEM((bt, bi), jnp.float32), pltpu.VMEM((bt, bi), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret_params() if interpret is True else interpret,
    )(h, w_gate_up, w_gate_up)
```
### B.7 `flash_jax/testing/fake_runner.py` (WP4)

```python
"""Model-free stand-in for ModelRunner: exercises scheduler + block manager without JAX.

It simulates a paged KV store whose entry for position p is a rolling hash of token_ids[:p+1] (i.e. it depends
on the whole prefix, like real KV). Before "computing" a chunk it asserts that every context position
< start_pos is present at (block_table[p // bs], p % bs) with the right prefix hash, which catches wrong block
tables, premature eviction/reuse, hash-chain bugs and touch-before-pop bugs. The "model" is deterministic:
next token = hash(context) % vocab, so preemption + recompute must reproduce the same outputs.
"""
from __future__ import annotations

from collections.abc import Sequence as Seq

import numpy as np

from flash_jax.batch import ScheduledSeq

_MOD = (1 << 61) - 1


def _roll(h: int, tok: int) -> int:
    return (h * 1_000_003 + int(tok) + 1) % _MOD


class FakeRunner:
    def __init__(self, num_kv_blocks: int, block_size: int, vocab_size: int = 1000) -> None:
        self.num_kv_blocks = num_kv_blocks
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.kv: dict[tuple[int, int], int] = {}  # (block, offset) -> prefix hash
        self.num_steps = 0
        self.num_tokens_computed = 0

    @staticmethod
    def next_token(context: Seq[int], vocab_size: int) -> int:
        h = 0
        for t in context:
            h = _roll(h, t)
        return int(h % vocab_size)

    def run(self, batch: list[ScheduledSeq]) -> np.ndarray:
        """Same contract as ModelRunner.run: one sampled token per scheduled sequence, in batch order."""
        out = np.zeros(len(batch), np.int32)
        bs = self.block_size
        for i, item in enumerate(batch):
            seq, start, n = item.seq, item.start_pos, item.num_new_tokens
            h = 0
            for p in range(start + n):
                h = _roll(h, seq.token_ids[p])
                blk = seq.block_table[p // bs]
                assert 0 <= blk < self.num_kv_blocks
                if p < start:
                    assert self.kv.get((blk, p % bs)) == h, f"stale or missing KV at position {p} of {seq}"
                else:
                    self.kv[(blk, p % bs)] = h
            out[i] = self.next_token(seq.token_ids[:start + n], self.vocab_size)
            self.num_tokens_computed += n
        self.num_steps += 1
        return out

    def reference_output(self, prompt: Seq[int], max_tokens: int, stop_token_ids: Seq[int] = ()) -> list[int]:
        """What a correct engine must generate for `prompt` (greedy, no batching effects)."""
        toks, out = [int(t) for t in prompt], []
        for _ in range(max_tokens):
            t = self.next_token(toks, self.vocab_size)
            toks.append(t)
            out.append(t)
            if t in stop_token_ids:
                break
        return out
```

---

## 11. Review resolutions

Each issue was verified by an experiment in the jax 0.10.2 venv (CPU; TPU lowering through the §4.2 hook), then
fixed or rejected. After the fixes, Appendix A and B were re-extracted from this document and re-run: the §3.4
example through `prepare_step_input`, the 8-case attention fuzz (on_wait, eager, race detector), the
TPU-default-config interpret case, norm/SwiGLU numerics and v5e/v6e lowering, the Mosaic idiom audit, the
scheduler stress (600 workloads), the full §8.2 reference matrix and both slow interpret configs (§0.3). A plain
`pytest` from the extracted tree passes without an install.

**Verdicts:** 35 issues, of which 2 are duplicates (#21 = #5, #18 = #11). Of the 33 distinct issues, 32 are
accepted (8 with a modified fix: #1, #7, #12, #13, #14, #17, #29, #30) and 1 is partially accepted (#16).
Nothing is rejected outright; the one rejected part is the second half of #16.

| # | sev. | issue | verification | resolution | sections changed |
|---|---|---|---|---|---|
| 1 | major | `add_rms_norm` row tile overflows v5e scoped VMEM | Arithmetic confirmed: bt=256 at H=4096 bf16 is 16 MiB of pipelined blocks + 12 MiB f32 temps; CPU lowering accepts it; the `vmem_limit_bytes` docstring requires `--xla_tpu_scoped_vmem_limit_kib`. | **Accepted, modified budget.** `row_tile` takes the largest legal tile (T, or power-of-two divisors ≥ sublane tile) with `8*bt*H*isz + 3*bt*H*4 <= VMEM_BUDGET_BYTES` (12 MiB, one budget for all kernels, not 8 MiB): bt=64 for H=4096 in bf16 **and** f32 (11 MiB), 32 at H=8192; reference fallback if nothing fits. Tests pin the tiles; `tpu` tests at (512, 4096) bf16/f32. SwiGLU gets the same budget (#13). | §4.2, §5.5.1, WP2, B.6a, A.14 |
| 2 | major | Attention kernel uses Mosaic idioms the bundled kernels avoid; no `bkv % 128` rule | Mosaic MLIR captured with `pallas_call(debug=True)`: old B.3 had 17 i1 lane-broadcasts, 16 bf16 selects, 24 width-1 lane slices; bundled RPA / `flash_attention.py` idioms confirmed in the venv sources. | **Accepted.** B.3 rewritten: (a) full-shape 2-D iota masks; (b) K unmasked, V masked in f32; (c) m/l `[rows,128]` widened with `jnp.tile`; (d) `bkv % 128` in `tpu_rules`; (e) interpret test at the exact TPU default config. Also dropped the `row_valid` select on m (a fully masked row leaves m unchanged). New MLIR: 0/0/0; all fuzz cases, eager, races, default config, bkv=Dc=256, bq=4 and e2e (a)/(b) pass. | §0.3, §5.4.5, §5.4.6, §9.1 R1, WP2, B.3 |
| 3 | major | Global `matmul_precision="highest"` changes the kernels under `pytest -m tpu` | Confirmed: 16 `contract_precision<fp32>` matmuls under "highest"; "high" raised `NotImplementedError` in lowering. | **Accepted (first alternative).** Every kernel matmul pins `precision=lax.Precision.DEFAULT` (B.3, B.6b): 0 fp32-precision matmuls under "highest", and "high" lowers. The conftest global stays for the XLA model path (comment updated); the lowering test asserts no fp32 contract under "highest"; `tpu` kernel tests use bf16-level tolerance. | §5.4.5, §5.5.2, §8.4, WP2, A.13, B.3, B.6b |
| 4 | major | Compile bound counts traces; committed/uncommitted mix recompiles silently | Reproduced: `_cache_size()` 2 vs 1 trace; explicit device → 1. | **Accepted.** `runner.device`; params, cache (`jnp.zeros(..., device=)`) and StepInputs committed to it; compile-bound test asserts `step_fn._cache_size()`; `stats["num_compiles"]`; benchmark reports both. | §0.3, §4.5, §5.3, WP5, WP6, §9.1 R8/R9, §10 #8 |
| 5, 21 | minor | `lower_for_tpu` recipe broken (string chip; memoized `get_tpu_info`) — reported twice | Both confirmed (`AttributeError`; v5e info returned after switching to v6e). | **Accepted, merged.** `ChipVersion(chip)`, `get_tpu_info.cache_clear()` after installing and after restoring (try/finally), chip assertion inside. | §4.2, WP2 |
| 6 | minor | "Lowering does not check alignment" is wrong | Confirmed: `_check_block_mappings` raises for (8,64) and (4,128) blocks of a (16,256) array. | **Accepted.** §0.3 lists exactly what lowering does and does not check; a misaligned-BlockSpec case expected to raise from lowering. | §0.3, §9.1 R1, WP2, A.14 docstring |
| 7, 30 | minor | VMEM rule never enforced; rules only at trace time; SMEM formula private to WP2; "Dc%128 automatic" false | Confirmed by reading B.3/§5.3; `kv_head_dim_align=64` gives Dc=64 with pallas. | **Accepted, merged; location modified.** New WP1 verbatim `flash_jax/ops/tpu_rules.py` (not `config.py`, which would need kernel math and cannot import `ops`): budgets, `attention_smem_bytes`, `attention_vmem_bytes` (+1 MiB values), `attention_problems`, `check_attention_config` over every bucket, called in `ModelRunner.__init__` before the pool; wrappers call the same function. Tiles shrink instead of raising the limit; the LIBTPU flag is documented. "automatic" dropped. | §2, §4.1, §4.2, §5.3, §5.4.6, WP1, WP2, WP5, A.14, B.3 |
| 8 | minor | Kernel tests only run `dma_execution_mode="on_wait"` | B.3 (new) passes all 8 cases in eager; `races_found` False with `detect_races=True`. | **Accepted.** `interpret_params(**overrides)`; kernels accept an `InterpretParams`; WP2 parametrizes random and edge cases over both modes plus one race case. | §4.1, §4.2, §6.1, §8.3, WP2, §10 #24, A.7 |
| 9 | minor | Reference asserts `T % min(32, T) == 0` | `AssertionError` at T=48 confirmed. | **Accepted.** `C = gcd(T, chunk)`; test_core covers T=48. | A.8, WP1, WP2 |
| 10 | minor | Reference attention is unusable as a production fallback | Confirmed 512 MiB per chunk at 2k and 8 GiB at 32k (Llama-3-8B). | **Accepted.** `max_chunk_bytes=256 MiB` halves C down to 1; R1 calls it a correctness fallback only; the runner warns when the reference attention is used on TPU. | §5.3, §9.1 R1, A.8, WP1 |
| 11, 18 | major | Sampler TV test fails with unit-scale logits (reported twice) | Confirmed: nucleus median 23, > 16 in 99.5% of draws, TV 0.21; with `3·N(0,1)` max 12 and TV ≤ 0.013 on all four cases. | **Accepted, merged.** Inputs pinned (`3.0 * rng.standard_normal(50)`), plus a wide-nucleus test against the K-truncated expectation (TV 0.0095). | §0.3 item 7, §5.6, WP5 |
| 12 | minor | Decode tokens share a q-tile: ~32× wasted attention compute on decode steps | Arithmetic confirmed (1.1 TFLOP ≈ 5.6 ms vs ≈ 10 ms of KV reads, 64 decodes at 1k, v5e). | **Accepted, modified.** `KernelConfig.attn_tile_q(T_pad)` is the single q-tile authority (work plan and kernel); buckets `T_pad <= next_pow2(max_num_seqs)` use `attn_block_q_small` (auto on pallas: smallest power of two with `bq*G` a multiple of the sublane tile, e.g. 4 for G=4; on reference/interpret it equals `attn_block_q`, so CPU test cost is unchanged). The prefill-remainder trade-off is documented; `benchmarks/kernels.py` A/Bs it; `EngineConfig.attn_tile_q` removed. | §3.1, §4.1, §5.3, §5.4.6, §5.7, §6.1, §6.2, WP1–WP3, WP6, §9.2 #5, A.3, A.6, A.9, B.5b |
| 13 | minor | SwiGLU re-streams the whole weight T/bt times (bt ≤ 256) | Confirmed; with our estimate (which also counts the two f32 matmul results), bt=1024/bh=512 is 14 MiB, over the budget. | **Accepted, modified.** `tiles()` maximises bt (≤ 1024, VMEM-checked), then bh: Llama-3-8B gets `(T,512,512)` up to 512 and `(1024,256,512)` at 1024, i.e. one weight pass per step; lowered for v5e/v6e. The benchmark prints the weight and h re-read factors. | §5.5.2, WP2, WP6, B.6b |
| 14 | minor | KV pool sizing ignores the step's temporaries | Confirmed; and `lower().compile()` shares the executable with later calls **only for identical shapes** (verified both orders), so a provisional-size compile is not free. | **Accepted, modified.** P from free × fraction; AOT-compile the largest bucket at that P against an abstract cache; shrink only if the slack is below `temp + 256 MiB`. No extra compile when P is unchanged, one extra when it shrinks. | §4.5 (`lower`), §5.3, §9.1 R10, §10 #32 |
| 15 | minor | `ModelConfig` as the static key recompiles on EOS changes | Confirmed by construction; after the fix 3 EOS variants added 0 executables. | **Accepted.** Static key `arch = replace(mcfg, eos_token_ids=(), max_position_embeddings=0)`; WP5 test. | §0.3, §4.5, §5.3, WP5 |
| 16 | minor | Assertion 8 and the seeded test assume bitwise-equal logits across steps/buckets | Reasoned: both compute the same positions in different buckets. | **Partially accepted.** Assertion 8 now uses assertion 3's margin filter. **Rejected:** the "allow one divergence if teacher-forced logits differ < 1e-5" rule for the seeded test: the run is deterministic on CPU (reproducible, not flaky), a flip needs perturbed scores within ~1e-6 (~1e-6 per token), and engine logits are not exposed by the API; the doc now documents the triage instead. | §8.2 |
| 17 | major | Nothing puts the repo root on `sys.path` | Confirmed: `ModuleNotFoundError` from conftest and from `python benchmarks/b.py`. | **Accepted, modified.** `pythonpath = ["."]`, `dev = ["pytest>=7"]`; benchmarks run as `python -m benchmarks.<name>` with `benchmarks/__init__.py`. No mandatory `uv pip install -e .`: the venv has no pip and an editable build fetches setuptools; it is optional. Verified `pytest` from the extracted tree with no install. | §2, §7, WP1, WP6, A.1 |
| 19 | minor | `check_invariants(running + waiting)` is a `TypeError` | Confirmed. | **Accepted.** `[*scheduler.running, *scheduler.waiting]`. | §4.5 |
| 20 | minor | "No priority inversion" is false | Confirmed: 19 steps in 600 workloads preempted an older seq while a newer one ran. | **Accepted, option (a).** B.2 kept; §5.2 and decision #13 now say progress is guaranteed, not strict priority, and argue no starvation (FIFO re-admission, no admissions in preempting steps). | §0.3, §5.2, §10 #13, WP4 |
| 22 | minor | SwiGLU leaves rows unwritten for T=384 | Confirmed: 16384 NaNs. | **Accepted.** Tile rule divides T (T=384 → bt=384), asserted; T=384 in WP2 tests. | §4.2, §5.5.2, WP2, B.6b |
| 23 | minor | `_warned` makes the fallback test order-dependent | Confirmed by reading. | **Accepted (second alternative).** Global removed; plain `warnings.warn`. The only global mutable state left is `TRACE_COUNTER`. | §4.2, §5.5.2, §7, WP2, B.6b |
| 24 | minor | `convert_hf_weights` Callable vs `convert_hf_weights(dict)` | Confirmed inconsistency. | **Accepted.** Accepts a Mapping or a callable; WP3 test covers both; recipes pass the dict. | §4.3, §8.2, WP3 |
| 25 | minor | WP6 lacks a canonical engine recipe and dependencies | Confirmed. | **Accepted.** Recipe in §8.2; WP6 depends on `ModelConfig.from_hf` and `convert_hf_weights`. | §8.2, WP6 |
| 26 | minor | Benchmark report needs APIs §4.5 does not expose | Confirmed. | **Accepted.** `ModelRunner.dummy_input`, `ModelRunner.lower(t_pad, num_blocks)`, `LLM.last_step`, defined `prompt_tokens` / `generated_tokens`, `num_compiles`. | §4.5, §5.3, WP5, WP6 |
| 27 | minor | `finish_reason="abort"` is unreachable | Confirmed (abort bypasses `update`). | **Accepted.** `abort()` queues the output; the next `step()` returns it; `generate` treats it as finished. | §4.5, WP5, §10 #34 |
| 28 | minor | `step()` runs a dummy device step when idle | Confirmed (`bucket_for(0) = 16`). | **Accepted.** Empty batch → no runner call. | §4.5, §5.3, WP5 |
| 29 | minor | Seeds and token ids are not range-checked | Confirmed: numpy `OverflowError` for `2**40`; the gather clamps id 12 of 10 to the last row. | **Accepted, modified.** `add_request` reduces the seed mod 2**31 (SamplingParams unchanged) and rejects out-of-vocab ids before touching the scheduler. | §4.5, WP5 |
| 31 | minor | `zip(..., strict=False)` hides runner bugs | — | **Accepted.** `strict=True`; WP4 test. | §5.2, WP4, B.2 |
| 32 | minor | WP3 hand-builds StepInputs with unstated constraints; duplicate layout logic | Confirmed (T=40/48 fail the plan/reference asserts). | **Accepted.** `prepare_step_input(batch, cfg, kc, *, t_pad=None)` moved into WP1's verbatim `batch.py` (the §3.4 test moves to test_core); WP3 tests, the runner and benchmarks share it. | §2, §3.4, §4.1, §4.5, §5.3, WP1, WP3, WP5, WP6, A.6 |
| 33 | minor | "integration" marker is not registered | Confirmed (A.1 has only `slow`, `tpu`). | **Accepted.** Integration tests use `pytest.importorskip` and carry no marker. | §7, WP5, WP6 |
| 34 | minor | `FLASH_JAX_KERNEL_BACKEND` breaks test_core and slows the e2e matrix | Confirmed by reading. | **Accepted.** Resolution tests `delenv` first; reference tests pass `kernel_backend="reference"` explicitly. | §6.1, §8.2, §8.3, WP1, WP5 |
| 35 | minor | `pair_num_kv_blocks` counts kv tiles, not blocks | Confirmed naming clash. | **Accepted.** Renamed `pair_num_kv_tiles`; §3.1 defines block = page vs kv tile vs q-tile. | §3.1, §3.4, §5.4.3–5.4.5, A.6, B.3 |
