# Fusion Advisor

Fusion Advisor is a static analyzer for PyTorch inference graphs. It traces an
`nn.Module` into `torch.fx`, finds the operator groups a compiler would fuse,
generates a real Triton kernel for each one, verifies the kernel against eager
PyTorch, and maps it back to the exact lines of your source.

`torch.compile` fuses operators internally but never tells you what it fused,
why, or whether the fusion actually helped. Fusion Advisor is built to expose those internals, i.e.
typed clusters, named rejections, readable Triton, per-cluster cost regime, and
source-mapped diffs, so you can audit and act on what the compiler hides.

```text
model.py ──[trace]──▶ FX graph ──[detect]──▶ clusters ──[lower]──▶ Triton ──[validate]──▶ speedup + diff
```

---

## How it works

Fusion Advisor operates on a specialized pre-IR compiler pipeline.

| Pass | Input | Output |
| --- | --- | --- |
| **Selective trace** | `nn.Module` | FX graph (SSA-like, source provenance per node) |
| **Shape propagation** | FX graph + example input | shape / dtype / stride per edge |
| **Classification** | each node | pointwise-unary, pointwise-binary, reduction, opaque |
| **Fusion detection** | classified graph | fusable clusters + rejected candidates with reasons |
| **Traffic estimation** | cluster + shapes | unfused vs fused DRAM bytes |
| **Triton lowering** | cluster | kernel source + wrapper (no `triton` import) |
| **Validation** | kernel | compiled, numerics-checked, benchmarked |
| **Source mapping** | cluster + provenance | line range, quality tag, red/green diff |

Everything up to validation is CPU-only (`triton` is emitted as text, never
imported), so detection, costing, and codegen are testable on a laptop.

### Tracing

FX gives an SSA-like def-use graph: each node produces one value, `node.args`
are its inputs, `node.users` its consumers, and graph order is a valid
topological schedule. Plain `symbolic_trace` collapses every `nn.Module` to one
opaque node, so the custom `FusionTracer` traces *through* fusion-relevant modules
(`GELU`, `Dropout`, `LayerNorm`, etc.) while keeping everything else opaque,
and records stack traces for the final source diff.

### Detection

Connected components of absorbable (pointwise + reduction) nodes form
candidates. Fusion detection finds maximal absorbable components in the FX graph,
then uses a greedy pruning heuristic to recover legal fusable subDAGs.

Consider a transformer residual where `relu` feeds both `mul` and an external
`layer_norm`:

```text
         x
         |
       relu ─────────┐
        |             |
  [ mul, add ]    layer_norm   (external consumer)
```

The component `{relu, mul, add}` fails fan-out because `relu` escapes. Rather
than reject everything, the pass drops `relu` and retries on `{mul, add}` -
the fusable tail survives. This is how real transformer blocks produce clusters
despite residual connections.

Fan-out is **not** `len(node.users) > 1` - a reconverging diamond like
`x * 2.0 + x` has two users *inside* the group, so `x` stays in a register.
The real predicate is whether any consumer lies outside.

**Legality checks** (run in order, first failure wins):

| Check | Rejects |
| --- | --- |
| **Fan-out** | value consumed outside the group (must be materialized) |
| **Convexity** | dependency path leaves the subDAG and re-enters (unschedulable) |
| **Shapes** | operands not broadcast-compatible (can't co-index) |
| **Aliasing** | views or in-place mutation on cluster values (FX models dataflow, not mutation order) |
| **Reduction** | multi-reduction or non-last-axis (one row-per-program kernel can't express it) |
| **Lowering** | node without a Triton lowering rule |

### Cost model

```text
unfused = Σ (operand reads + result writes) per op
fused   = external inputs read once + escaping outputs written once
```

DRAM upper bound, no cache model. Measurements are tagged by regime:
`launch-bound` (wall time is launch overhead - no speedup printed),
`l2-resident` (fits in L2, ratio overstates DRAM savings), or
`dram-bound` (traffic estimate applies directly).

### Lowering

The emitter walks the cluster topologically, binds one Triton temporary per SSA
value, and substitutes each op for its target expression. Two skeletons:

- **Elementwise**: flat grid, one program per block, masked tail loads/stores.
- **Reduction**: one program per row, power-of-two block, tail lanes neutralized
  with per-op identity (`-inf` for softmax, `0` for sum) before the reduce.

Broadcast operands get independent index expressions via right-aligned div/mod
on the propagated shapes.

### Validation

Four gates before a kernel is reported as usable: cluster numerics (vs eager
subgraph), model numerics (patched into the full graph), benchmark (median-timed,
regime-tagged), and bandwidth (achieved GB/s vs detected peak, override with
`--peak-gbps`). Kernels compile from disk (`@triton.jit` needs
`inspect.getsourcelines`). Each is wrapped in `torch.autograd.Function` whose
`backward` raises (currently only inference is supported).

### Source mapping

Each node's stack trace resolves to a source line. A cluster maps `exact` when
every node is in the same function, no outside node falls in the range, and
every input name exists in source. Otherwise it degrades to a pointer (no diff).
`--apply` rewrites only exact ranges, bottom-up, after confirming the file is
unchanged.

---

## Installing

Fusion Advisor is not yet published to PyPI. Install it from a checkout:

```bash
git clone <repo-url> fusion-advisor
cd fusion-advisor
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,dev]'      # 'gpu' pulls in triton; drop it for CPU-only analysis
```

Python 3.12+ is required. CUDA and Triton are needed only for the validation and
benchmarking stages; tracing, detection, costing, and code generation run on
CPU.

```bash
pytest -q                 # full suite (GPU tests auto-skip without a device)
pytest -m gpu -v -s       # kernel numerics + benchmarks, on a CUDA box
```

| Package | Responsibility |
| --- | --- |
| `fusion_advisor/ir/` | FX tracing, op classification, tensor metadata |
| `fusion_advisor/analysis/` | detection, legality, traffic estimation |
| `fusion_advisor/codegen/` | FX → Triton lowering and emission |
| `fusion_advisor/validate/` | compile, numerics, benchmark |
| `fusion_advisor/sourcemap/` | provenance and guarded source rewriting |

---

## Using it

Point the CLI at a file that defines an `nn.Module` and give it an example input
shape. The repo ships a showcase model in [`examples/example_net.py`](examples/example_net.py):

```python
class ExampleNet(nn.Module):
    def forward(self, x):
        x = x * 2.0                     # ┐
        x = F.relu(x)                   # │ cluster 0 (elementwise)
        x = x + 1.0                     # ┘

        x = self.proj1(x)               # opaque barrier (Linear)
        x = F.gelu(x)                   # ┐
        x = torch.sigmoid(x)            # │ cluster 1 (elementwise)
        x = x * 0.5                     # ┘

        x = self.proj2(x)               # opaque barrier (Linear)
        x = torch.tanh(x)               # ┐
        x = x * 3.0                     # │ cluster 2 (reduction)
        return F.softmax(x, dim=-1)     # ┘
```

### Analyze

```bash
fusion-advisor --model examples/example_net.py --input-shape 16384,1024 \
    --out-dir generated/example_net
```

The two `nn.Linear` calls split the graph into three fusable regions. Each
generated kernel is written to `generated/`, and the analysis is printed:

```text
ExampleNet (17 nodes, 3 fusable cluster(s))

Fusable clusters (estimated memory traffic)
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ # ┃ category                  ┃ ops ┃   traffic ┃  traffic ┃ traffic ┃ map   ┃ lines  ┃
┃   ┃                           ┃     ┃ (unfused) ┃  (fused) ┃   saved ┃       ┃        ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ 0 │ elementwise-chain         │ 3   │  384.0 MB │ 128.0 MB │     67% │ exact │ L15-17 │
│ 1 │ elementwise-chain         │ 3   │  192.0 MB │  64.0 MB │     67% │ exact │ L21-23 │
│ 2 │ single-reduction-boundary │ 3   │  192.0 MB │  64.0 MB │     67% │ exact │ L27-29 │
└───┴───────────────────────────┴─────┴───────────┴──────────┴─────────┴───────┴────────┘
Across 3 cluster(s): 512.0 MB of 768.0 MB cluster-local traffic avoided.
Estimates are a DRAM upper bound with no cache model.
```

Reading a row: cluster 0 is the `x*2 → relu → +1` chain on lines 15-17. Unfused
it moves 384 MB (each op reads and writes a 64 MB tensor); fused it moves 128 MB
(read the input once, write the result once). The `exact` map means a diff is
available.

### The generated kernel

`generated/cluster0.py` is real, dynamically emitted, runnable Triton:

```python
@triton.jit
def cluster0_kernel(in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr):
    # shared boilerplate (elementwise)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load input tensors and chain operations
    v0 = tl.load(in_ptr0 + (offs), mask=mask)
    v1 = (v0 * 2.0)
    v2 = tl.maximum(v1, 0.0)
    v3 = (v2 + 1.0)

    # store output
    tl.store(out_ptr0 + (offs), v3, mask=mask)
```

Cluster 2 ends in a softmax, so it uses the reduction skeleton. There's one program per
row, tail lanes neutralized before the reduce, and `tanh` lowered through
`sigmoid` since Triton has no `tl.tanh`:

```python
@triton.jit
def cluster2_kernel(in_ptr0, out_ptr0, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # shared boilerplate (reduction)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    base = row * n_cols + offs

    # load input tensors and chain operations
    v0 = tl.load(in_ptr0 + (base), mask=mask)
    v1 = (2.0 * tl.sigmoid(2.0 * v0) - 1.0)
    v2 = (v1 * 3.0)
    v3 = tl.where(mask, v2, -float('inf'))
    v4 = tl.softmax(v3, dim=0)  # reduction operation

    # store output
    tl.store(out_ptr0 + (base), v4, mask=mask)
```

### Measure (on a GPU)

With CUDA and Triton available, the same command compiles each kernel, checks it
against eager, and benchmarks it. Example run on an RTX 4060, `16384 x 1024`
input:

```text
Measured
cluster 0: verified (max abs err 0.00e+00)
  dram-bound  128.0 MB  fused cluster speedup 2.97x  model speedup 1.17x  fused cluster bandwidth 214 GB/s (83% of 256 GB/s detected peak memory bandwidth)
cluster 1: verified (max abs err 8.94e-08)
  dram-bound  64.0 MB   fused cluster speedup 2.81x  model speedup 1.08x  fused cluster bandwidth 201 GB/s (78% of 256 GB/s detected peak memory bandwidth)
cluster 2: verified (max abs err 1.12e-08)
  dram-bound  64.0 MB   fused cluster speedup 2.65x  model speedup 1.09x  fused cluster bandwidth 194 GB/s (76% of 256 GB/s detected peak memory bandwidth)
```

The cost model predicts a 3x traffic reduction for a three-op chain; the
measured 2.97x at 83% of peak bandwidth confirms it. Model speedup is much
smaller (Amdahl - these clusters are a fraction of the whole network), and is
reported separately so the cluster-local number is never mistaken for an
end-to-end claim.

### See the diff and apply it

Every `exact` cluster prints the swap it enables:

```diff
examples/example_net.py:15-17
- x = x * 2.0
- x = F.relu(x)
- x = x + 1.0
+ x = cluster0(x)
```

`--apply` writes these edits into the model file and drops the kernels into a
sibling module, after confirmation.

### A realistic model

[`examples/microgpt.py`](examples/microgpt.py) is a complete decoder-only
transformer: token/position embeddings, causal multi-head attention, pre-norm
blocks, MLPs, tied LM head. It takes integer token IDs, so the input is `int64`:

```bash
fusion-advisor --model examples/microgpt.py \
    --input-shape 2,128 --dtype int64 --out-dir generated/microgpt
```

### Flags

| Flag | Effect |
| --- | --- |
| `--model PATH` | file defining the `nn.Module` (required) |
| `--model-class NAME` | pick a class when the file defines several |
| `--input-shape N,M` | example input shape; repeat for multi-input models (required) |
| `--dtype` | input dtype (default `float32`; use `int64` for token IDs) |
| `--out-dir DIR` | where generated kernels are written |
| `--explain-rejections` | list candidates that failed legality, and why |
| `--vs-inductor` | also benchmark `torch.compile` per cluster |
| `--peak-gbps` | override the auto-detected peak memory bandwidth |
| `--apply` | rewrite the model file in place, after confirmation |
| `--json FILE` | write machine-readable results (versioned schema) |
