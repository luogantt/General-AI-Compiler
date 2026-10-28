# NVIDIA FP16 MSDA arithmetic regression

`capture.npz` contains five synthetic inputs and the output of NVIDIA's
`MultiscaleDeformableAttnPlugin_TRT`, version 2, from TensorRT **10.13.3.9** on
an **RTX 3070 Ti Laptop GPU**, using PyTorch 2.7.0+cu126 / CUDA 12.6 buffers.
It contains no images, model weights, or private data.

The inputs came from `scripts/benchmark_msda.py` with seed `20261021`:
`B=2, S=20, H=2, C=5, Q=7, L=3, P=3`, spatial shapes `[[3,5],[2,2],[1,1]]`.
Floating tensors and `expected` are FP16; `shapes` and `starts` are INT32.
`expected` is captured official-plugin output, not output from the CPU oracle.

This fixture exposed rounding differences in the previous unfused CPU oracle.
The regression compares every FP16 bit and protects the observed half-FMA
coordinate, interpolation-pair, and accumulation order. It does not validate
GPU execution or full-model accuracy on the machine running the CPU tests.

Derived from `build/msda-optimization-20261021/check-04.oracle-failure.npz`
(SHA-256 `7880c53749dc0793e5d02152bec163a8fdbdfcad17b98d9d32ef4738ebd9b952`).
Only `value`, `shapes`, `starts`, `locations`, `weights`, and `expected` were
copied; the obsolete `oracle` array was dropped. The accompanying GPU report
was `check-04.json`.

Fixture SHA-256:
`aaad4df202527505d521330f844c9361b3cd5da910b7220e1a50d0aa1350c818`.
