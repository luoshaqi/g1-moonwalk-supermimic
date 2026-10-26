# Rebuild the motion and portrait video

Run from the repository root in the installed environment, with `SUPERMIMIC_ROOT` set as in README.

## CMU 90_32 → human joints → G1 → mjlab NPZ

The bundled NPZ is the exact experimental numeric data with only its private `source` metadata string replaced. Original and release SHA-256 values are recorded in `data/provenance.json`; container hashes differ because metadata differs. Do not compare these as if they were different trajectories.

To rebuild from source (no network or training is performed by the IK script):

```bash
mkdir -p outputs/cmu
curl --fail --location https://mocap.cs.cmu.edu/subjects/90/90.asf -o outputs/cmu/90.asf
curl --fail --location https://mocap.cs.cmu.edu/subjects/90/90_32.amc -o outputs/cmu/90_32.amc
.venv/bin/python scripts/cmu_to_xyz.py --source-dir outputs/cmu
.venv/bin/python scripts/retarget_cmu.py \
  --input outputs/cmu/human_reference_xyz.npz --out outputs/retarget
```

The decoder implements ASF/AMC forward kinematics for this particular CMU clip: 120 Hz, unit factor `.0254/.45`, Y-up to Z-up rotation and one constant floor offset. This is not a general ASF parser.

Retargeting uses cubic resampling to 50 Hz, smoothing, robot limb-length scaling, bounded per-frame whole-body IK, calibrated foot pitch, both-foot floor constraints and temporal regularization. Root XY follows the scaled original path; yaw is normalized to the initial heading. Foot yaw relative to body heading is limited to ±0.35 rad; joint speed bounds are 6 rad/s (hip yaw/roll 4 rad/s after the first frame). The final common vertical correction preserves relative foot heights. This is a clip-specific heuristic, not a general retargeting benchmark or proof of dynamic balance.

Outputs include `cmu90_32_moonwalk.npz`, `g1_moonwalk_50hz.csv`, `g1_moonwalk_reference.npz` and `retarget_report.json`. The CSV uses XYZ + XYZW quaternion + 29 joints; qpos uses WXYZ; the upstream converter writes the full mjlab motion fields.

Use the bundled motion for the published results. Numerical differences can occur when re-running IK with other NumPy/SciPy/MuJoCo builds. A rebuilt NPZ also contains its new CSV source path; motion checks should compare numeric arrays, not file hashes alone.

## Evaluation without training

```bash
.venv/bin/python scripts/pipeline.py --stage evaluate \
  --run-dir outputs/moonwalk/logs/moonwalk \
  --output outputs/reeval
```

Evaluation: 1 environment, seed 0, the same 300 transitions of the same 6-second training clip. No random held-out split is claimed. The pipeline stores actual simulated root/joint states and matched reference states, including the terminal state. Rendering replays these states with forward kinematics; it does not run a policy or synthesize successful motion. The camera tracks their shared midpoint without translating actual and reference independently.

For detailed metric definitions, see the pinned upstream `scripts/evaluate_hard_suite.py`. The completion, finite checks, termination and root/joint errors come directly from that evaluator. The normalized full-clip return penalizes incomplete rollouts via the expected full-clip length.

## Portrait comparison (1080 × 1920)

Install FFmpeg and a Chinese font; on Debian/Ubuntu, `ffmpeg` and `fonts-noto-cjk` packages provide them. Then:

```bash
.venv/bin/python scripts/make_portrait.py \
  --baseline media/baseline.mp4 \
  --finetuned media/finetuned.mp4 \
  --report results/comparison.json \
  --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
  --output outputs/comparison_portrait.mp4
```

Use your local font path if different (the supplied video was composed with macOS STHeiti Medium). The fixed crop is designed for this moonwalk follow-camera view; inspect heads/feet before using it for another motion. The videos start together and stop at the shorter input; no speed change, music or interpolation is added. Labels come from the provided metrics JSON. Font rendering can differ across machines.

## Validation scope

The original 3,000-iteration experiment ran on A800. The portable publication scripts are additionally checked by a short GPU optimization run and an evaluation/render pass reusing the original trained checkpoint. Fresh dependency installation on arbitrary platforms is not guaranteed by these checks. See `results/release_validation.json` for actual checks performed, rather than treating publication as a new full-training run.

## Environment notes

`source scripts/env.sh` adds the active environment’s pip CUDA libraries and Python include path for Warp/JIT compilation. Set `MOONWALK_PYTHON` before sourcing it if reusing another Python environment. It does not modify the system driver. Missing `Python.h` requires the development headers matching your Python 3.12 interpreter.

Our A800 container had stale NVIDIA libraries on its initial loader path; the experiment selected the matching host driver libraries before adding pip CUDA libraries. If `nvidia-smi` reports a driver/library version mismatch, fix the container driver-library path for your machine. Do not copy a private server’s library paths blindly. Headless rendering also requires a working EGL or OSMesa installation.

A historical, sanitized package snapshot is included in `results/environment-reference.txt` for troubleshooting; `scripts/setup.sh` and `requirements.txt` are the installation entry points.

Publication re-evaluation on 2026-10-26 completed 300/300 steps for both policies. Baseline normalized return was 0.83415 versus the historical 0.83293; fine-tuned return was 0.945327 versus 0.945328. The historical table/media remain unchanged. Repeated GPU evaluation is not asserted to be bitwise deterministic.
