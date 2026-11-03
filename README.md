# quantized-gemv-bandwidth

When a model decodes one token at a time (batch 1), every weight is read once per token, so the
speed limit is memory bandwidth divided by bytes per weight. Quantizing weights only buys speed if
the kernel can unpack them as fast as memory delivers them. I wanted to see where that breaks on
my laptop's CPU, so I wrote a small benchmark.

`gemv.c` times `y = W x` for one 8192 x 16384 matrix (134M weights) stored as fp32, fp16, int8
and int4 (two weights per byte, one fp32 scale per group of 32), using a few kernels:

| name    | weights | activations | kernel |
|---------|---------|-------------|--------|
| fp32    | fp32    | fp32        | plain C loop |
| fp16    | fp16    | fp32        | plain C loop |
| int8    | int8, per-row scale | fp32 | plain C loop |
| int4    | int4, interleaved nibbles | fp32 | plain C loop |
| int4s   | int4, split layout (like llama.cpp Q4_0) | fp32 | plain C loop |
| i8q8    | int8 | int8, per-group scale | integer dot per group, plain C |
| i4q8    | int4, split layout | int8 | integer dot per group, plain C |
| i4q8n   | int4, split layout | int8 | NEON `vdotq_s32` (SDOT) |

It reports the effective weight bandwidth and what that means in tokens/s for an 8B-parameter
model, and checks every kernel against a double-precision reference (max relative error is
printed). `stream.c` is a STREAM-style copy/triad test for the bandwidth ceiling.

## Results (Apple Silicon laptop, 12 threads, best of 15 runs)

STREAM copy: about 231 GB/s.

| kernel | bytes per weight | weight GB/s | % of STREAM | tok/s for 8B params |
|--------|------------------|-------------|-------------|---------------------|
| fp32   | 4      | 216 | 93% | 6.8 |
| fp16   | 2      | 197 | 85% | 12.3 |
| int8   | 1      | 152 | 66% | 18.9 |
| int4   | 0.625  | 34  | 15% | 6.7 |
| int4s  | 0.625  | 57  | 25% | 11.3 |
| i8q8   | 1      | 190 | 82% | 23.7 |
| i4q8   | 0.625  | 21  | 9%  | 4.1 |
| i4q8n  | 0.625  | 130 | 56% | 25.9 |

Full output for 1, 4, 8 and 12 threads is in `results-run1.txt`.

What I took from it:

- fp32 and fp16 are bandwidth-bound: going from 4 to 2 bytes per weight nearly doubles tokens/s.
- A naive int4 kernel is slower than fp16 even though it reads 3.2x fewer bytes. Unpacking
  nibbles and converting each one to float is the bottleneck, not memory.
- Just changing the nibble layout so both halves of a group are contiguous lets the compiler
  vectorize better (34 to 57 GB/s), but it's still compute-bound.
- Quantizing the activations to int8 and doing integer dot products fixes int8 (190 GB/s, close to
  the ceiling). For int4 it only works when written with SDOT intrinsics: 130 GB/s, 2.1x the
  tokens/s of fp16, but still only about 56% of the bandwidth ceiling, so int4 doesn't give the
  full 3.2x here.
- On a CPU, int4 needs most of the cores just to keep up with memory: single-threaded, i4q8n
  reaches 17 GB/s versus 130 GB/s for STREAM.

## Caveats

- One matrix, not a full model: no attention, KV cache, or activation traffic, so the tokens/s
  column is an upper bound from weight reads only.
- The plain C kernels rely on clang `-O3 -ffast-math` auto-vectorization; they are not tuned like
  llama.cpp or a GPU kernel. Results on another machine or compiler will differ.
- Other processes were running on the machine during these runs, so numbers vary between runs
  (most for the int4 SDOT kernel: 130 to 150 GB/s across runs at 8 to 12 threads). The ordering of
  the kernels stayed the same.

## Run it

```sh
./run.sh                       # Apple Silicon only (uses NEON dotprod)
THREADS="8" MODEL_PARAMS=70e9 ./run.sh
```
