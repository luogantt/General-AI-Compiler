# Remote Vortex full-graph validation — 2026-10-15

The test ran on `robot@192.168.115.128` using one Vortex SimX device. It executed the tiny `bevformer_streaming_step` graph with 811 dispatches covering all 33 currently registered GAC operators, including explicit history selection and warp outputs. This uses synthetic weights and reduced shapes; it does not establish full-size BEVFormer checkpoint accuracy or chip latency.

- Remote artifact: `/home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015`
- Artifact manifest SHA-256: `405f41a0bf3114b9fd3710b99ae0c1de56f209fc00c5b530334632871d868adc`
- Device vxbin SHA-256: `888c8fba432bb4e0eb4e16eb819cc94d9d5dfbf02582eef9651bd0def7e3b398`
- Device code compiled by pinned VOLT/MLIR LLVM 20.1.8 in WSL.
- Host source compiled on the remote Ubuntu22 machine against its existing Vortex2 Runtime.
- Remote Vortex source and installed toolchains were not modified.
- Runtime input: frozen cold-frame inputs; one device, 4 threads per CTA, no warmup, one repetition.

All nine outputs pass both remote NumPy validation and an independent comparison of downloaded raw outputs against the local artifact goldens. Maximum absolute error is `7.62939453125e-06` on bbox_regression; FP16 next_bev matches exactly. Tolerances are rtol=0.002 and atol=0.0003. The reported `4540.340167 ms` is simulator/driver host wall time and is not hardware inference latency.

Run from the configured Vortex build directory, as required by its AGENTS.md. Every command below has a 120-second cap.

```bash
cd /home/robot/project/vortex/build

timeout 120s python3 /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/verify_artifact.py

timeout 120s g++ -std=c++17 -O2 -Wall -Wextra -pedantic -Werror -I/home/robot/project/vortex/sw/runtime/include /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/main.cpp -L/home/robot/project/vortex/build/sw/runtime -Wl,-rpath,/home/robot/project/vortex/build/sw/runtime -lvortex -o /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/gac_bevformer_full_vortex

timeout 120s env LD_LIBRARY_PATH=/home/robot/project/vortex/build/sw/runtime VORTEX_DRIVER=simx /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/gac_bevformer_full_vortex --data-dir /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015 --kernel /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/kernel.vxbin --threads 4 --repeat 1 --warmup 0

timeout 120s /home/robot/anaconda3/bin/python /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/verify_outputs.py --data-dir /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015 --report /home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015/cold-correctness.json
```

Evidence files:

- `remote-deployment.json`: exact commands, hashes, exit codes and timings.
- `remote-environment.json`: CPU/OS, hardware configuration, toolchain revisions and Runtime hashes.
- `remote-cold-simx.stdout.log`: unmodified simulator output.
- `remote-cold-device-check.json`: comparison performed on the remote machine.
- `remote-cold-correctness.json`: independent local comparison, including shapes and per-output error.
- `remote-output-*.bin`: downloaded device-produced raw outputs.


The same binary was subsequently tested with two additional invocation fixtures:

1. Warm frame: `runtime_inputs(small_config(), warm=True)` with `prev_bev` replaced by the **actual remote cold-frame** FP16 `next_bev`. This checks state transfer between separate host invocations.
2. History-route probe: warm fixture with FP16 `prev_bev = arange(shape) + 3` and final `can_bus_delta` angle set to zero. The device's `selected_prev_bev` output exactly matches this nonzero history.

Expected outputs were evaluated locally by the independent NumPy interpreter using the optimized `02-graph.json` plus frozen raw constants. Original/optimized graph equivalence was separately verified by the WSL validation runner. For each invocation, all old output files were removed first, preventing stale-output passes. Remote Anaconda NumPy and independently downloaded-output comparisons both passed. Invocation inputs/goldens/outputs were restored to the original cold fixture afterward; source hashes and cold output correctness were checked again.

See `remote-summary.json` for all three frame results, `remote-state-validation-commands.json` for exact execution commands, and `remote-warm-correctness.json` / `remote-history-route-probe-correctness.json` for per-output shape, error and SHA-256 evidence. Fixture generation follows `tools/validate_vortex_full_graph.py`; command logs assume the corresponding fixture has already been installed, since the frozen remote artifact intentionally remains a cold fixture.
