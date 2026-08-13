# U0 Local Validation Record

Date: 2026-08-13

This record establishes local code readiness only. It is not a confirmatory run,
does not support a performance claim, and does not by itself close U0 criteria
that lack an explicit test.

## Unit suite

- Interpreter: `C:\lintao\envs\fixmatch\python.exe`
- Python: 3.10.19
- PyTorch: 2.5.1
- torchvision: 0.20.1
- CUDA runtime reported by PyTorch: 11.8
- Command: `python -m pytest -q -p no:cacheprovider`
- Result: 42 tests run, 42 passed, 0 failed, 0 skipped
- Elapsed time reported by final pytest regression: 4.59 seconds
- Source-bundle SHA256: `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd`

The source-bundle hash covers `train.py`, every Python/Bash file under
`scripts/`, and all Python files under `tangs/` and `tests/`, sorted by relative
path and hashed as a newline-delimited `path<TAB>sha256` manifest.

The minimal-gate, development, required-run, and smoke Bash scripts passed
`bash -n` under `C:\msys64\usr\bin\bash.exe`. The Windows WSL shim was not used.
The development and required launchers were also executed against isolated
empty result roots and returned their registered exit code 4 without starting
training. The smoke wrapper correctly skipped a completed run.

## P0 runtime and spending gate

- Persistent DataLoader workers were added after an invalid unversioned pilot
  repeatedly respawned Windows workers when cycling short loaders.
- The invalid pilot remains in `results/pilot-c10-100-fixmatch-20000step-seed0/`.
- The valid paired runs use the `perf-v2` suffix and identical worker policy.
- Worker PIDs remained stable across loader exhaustion and sampled RTX 4060
  utilization stayed between 83% and 96% during training windows.
- Gate result: PASS; artifact `results/minimal-gate-c10-100-20000step.json`.
- P0 is development-only and is not paper evidence.

## Current v4.5 GPU smoke and resume preflight

- Fresh run ID: `u0-server-preflight-v45-final-20260813`
- Protocol/method: P-C100-100 / TANGS
- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- Result: completed 5 of 5 optimizer steps, final evaluation, checkpoint,
  diagnostics, status, manifest, and summary creation
- Config SHA256: `b2fa6bd9e9017c8b0845ca5c281d13270e9f1e52338c5fb5885a0efca338fedb`
- Manifest protocol version: 4.5
- Local artifact: `results/u0-server-preflight-v45-final-20260813/`

The first real CUDA `--resume auto` audit exposed and then fixed a device-mapped
RNG-state bug: checkpoint CPU RNG byte tensors were being loaded onto CUDA before
`torch.set_rng_state`. The fixed implementation explicitly moves CPU and CUDA RNG
state tensors back to CPU before calling the restoration APIs.

A separate controlled mid-run test `u0-resume-midrun-v45-20260813` was terminated
after a step-4 rolling checkpoint and resumed through optimizer steps 5–20. It
completed with a final checkpoint and summary, and the run manifest contains one
`resume_history` entry. Config SHA256:
`972e9483d739a0598e614477fa7c59cb77f36abf2e25f4c0afdfdf8d89e05877`.
Loader continuation remains operationally safe but is explicitly not bit-exact.

The required STL branch was validated separately with run ID
`u0-stl10-20-server-preflight-v45-20260813`. It completed five CUDA optimizer
steps and the full 8,000-example test evaluation. The split manifest records the
official 100,000-image unlabeled pool, 1,528 appended labeled images, and final
unlabeled-loader cardinality 101,528. Peak CUDA allocation was 914.74 MiB and
the config SHA256 was
`1e9b9ee0193bfee04b9ffcb422239023b439376911ed600b542ffcdf307f930e`.

## Historical GPU smoke

- Run ID: `u0-smoke-tangs-cdmad-table-v44-env4`
- Protocol/method: P-C100-100 / TANGS
- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- Result: completed 5 of 5 optimizer steps
- Training-loop elapsed time: 33.019 seconds
- Config SHA256: `4c2abb7fb4fd174e24ccf9fa9a5729fee001d676dd87acf4d40ac6753c7134fb`
- Local artifact: `results/u0-smoke-tangs-cdmad-table-v44-env4/`
- E1 instrumentation: 5 synchronized timing rows with non-null extra-gradient,
  surgery, diagnostics, allocated/reserved memory, and known gradient/anchor
  tensor-footprint fields
- Pinned CDMAD commit: `7cd732b4615b9d94934a9197e69c6775496fb5ee`
