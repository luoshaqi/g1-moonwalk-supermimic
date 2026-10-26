"""CPU MuJoCo replay rendering of ``evaluate_hard_suite.py`` trajectory NPZ files.

  .venv/bin/python scripts/render_eval_trajectory.py \
      --trajectory logs/validation/hard_suite_ckpt_latest_traj/flip_360_001__A304.npz \
      --out logs/validation/flip_smoke.mp4

Input: the NPZ written by ``scripts/evaluate_hard_suite.py --trajectory-clips`` —
closed-loop recorded sim states (``sim_root_pos`` / ``sim_root_quat`` wxyz /
``sim_joint_pos``, paired with ``sim_ref_frame_idx``) and the matching reference motion
(``ref_root_pos`` / ``ref_root_quat`` wxyz / ``ref_joint_pos``, paired with
``ref_frame_idx``), plus ``dt``, ``joint_names`` and the termination bookkeeping
(``steps`` / ``expected_steps`` / ``failed`` / ``failure_step`` / ``failure_time_s`` /
``termination_reasons`` / ``terminal_state_recorded``). Current-schema NPZs record the
true terminal state (``terminal_state_recorded=True``, sim rows = step-0 initial state +
every counted transition including the terminal one); pre-terminal-recording NPZs that
only carry ``sim_rows_stop_before_final_state`` stay readable and are labelled exactly
as before.

This script only *replays* the recorded states on the CPU with the fixed vendor G1 model
from ``supermimic.data.motion_convert.load_g1_model()`` (joint order resolved by name
against the model; ``--robot-variant`` selects the geometry-only no-hands spec). No
policy is loaded and no physics step is taken: every frame's qpos
is written straight from the recording and only forward kinematics runs, for rendering.

Output: one MP4 (libx264 via imageio-ffmpeg). ``--layout side-by-side`` (default) shows
the actual rollout on the left and the reference motion on the right. Both panels use
the same fixed world-space camera,
centred on the joint bounds of both recordings. A render-only z=0 plane and metre grid
show ground height and translation; neither recording is shifted or re-centred. PIL text at the bottom states the clip, the sim/reference roles, the
time, the failure step and explicitly that this is a closed-loop recorded trajectory
replay. If the recording failed, the video ends after at most ``--failure-tail-seconds``
(default 0.5) past the failure (the actual tail may be shorter when the reference clip
or ``--max-seconds`` ends first); both panels freeze at the failure moment with a clear
FAILED / TERMINAL STATE banner and the footer reports requested vs actual tail — the
long static tail that looked like a stuck policy is gone.
The reference is never shown as if it were the actual and the recording is never
extended past its terminal state.

``--layout overlay`` renders one full-width camera view instead: the actual rollout's
opaque visual meshes plus the reference motion's meshes at their true saved
world-space pose appended into the same MuJoCo scene (``mjv_addGeoms`` from a second
MjData holding the recorded reference state) and tinted cyan at ``--reference-alpha``
(default 0.35) opacity. The two recordings are never image-blended, so neither the
actual nor the ground ghosts. Overlay requires ``--iteration`` (drawn top-left, e.g.
"Iteration 45,800"), carries an actual/reference legend and a footer without the
left/right panel role labels; the reference/time pairing, terminal state and frozen
failure tail are unchanged.

Rendering backend: ``--backend osmesa|egl|glfw`` (default osmesa) maps to MUJOCO_GL and
is applied before mujoco is imported; without the flag, an existing MUJOCO_GL wins.
Per-backend requirements (nothing is installed by this script):
  * osmesa - CPU headless; needs the system library libOSMesa.so.8 (Debian/Ubuntu
    package ``libosmesa6``) so PyOpenGL can bind GL.
  * egl - headless EGL device context; needs Mesa/NVIDIA EGL device enumeration
    (``eglQueryDevicesEXT``) and selects a device via MUJOCO_EGL_DEVICE_ID. For a GPU
    render node the caller must verify which physical device index the enumeration
    returns; with ``__EGL_VENDOR_LIBRARY_FILENAMES`` pointing at 50_mesa.json and
    GALLIUM_DRIVER=llvmpipe it stays CPU software rendering.
  * glfw - X11 GLX; needs a display (e.g. Xvfb) whose server returns GLXFBConfigs.
"""

from __future__ import annotations

import os

# MuJoCo reads MUJOCO_GL at import time; default to the CPU headless backend.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Fixed camera geometry shared by both panels (only the lookat XY anchor differs).
CAM_AZIMUTH_DEG = 140.0
CAM_ELEVATION_DEG = -20.0
CAM_FOVY_DEG = 45.0

REQUIRED_KEYS = (
    "dt",
    "joint_names",
    "sim_root_pos",
    "sim_root_quat",
    "sim_joint_pos",
    "sim_ref_frame_idx",
    "ref_root_pos",
    "ref_root_quat",
    "ref_joint_pos",
    "ref_frame_idx",
)

LABEL_COLOR = (235, 235, 235)
STATUS_OK_COLOR = (170, 220, 140)
STATUS_FAIL_COLOR = (240, 80, 70)
NOTE_COLOR = (245, 190, 70)
MUTED_COLOR = (165, 165, 175)

# Overlay layout: cyan tint for the appended reference geoms (float rgb for mjvGeom.rgba).
REFERENCE_TINT = (0.0, 0.85, 1.0)
TEXT_BG = (20, 20, 24)

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


# --------------------------------------------------------------------------
# trajectory loading / validation (pure numpy, no mujoco import needed)
# --------------------------------------------------------------------------


@dataclass
class Trajectory:
    name: str
    dt: float
    joint_names: list[str]
    sim_root_pos: np.ndarray  # (Ns, 3)
    sim_root_quat: np.ndarray  # (Ns, 4) wxyz
    sim_joint_pos: np.ndarray  # (Ns, 29)
    sim_time: np.ndarray  # (Ns,) seconds from clip start
    ref_root_pos: np.ndarray  # (Nr, 3)
    ref_root_quat: np.ndarray  # (Nr, 4) wxyz
    ref_joint_pos: np.ndarray  # (Nr, 29)
    ref_time: np.ndarray  # (Nr,) seconds from clip start
    steps: int
    expected_steps: int
    failed: bool
    failure_step: int
    failure_time: float  # nan when not failed
    terminal_state_recorded: bool  # False only for pre-terminal-recording NPZs
    termination_reasons: list[str]  # e.g. ["root_tilt", "clip_end"]; empty when unknown

    @property
    def sim_end_time(self) -> float:
        return float(self.sim_time[-1])

    @property
    def ref_end_time(self) -> float:
        return float(self.ref_time[-1])


def _scalar(npz, key: str, default):
    if key not in npz.files:
        return default
    return np.asarray(npz[key]).item()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _normalize_quat(q: np.ndarray, what: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1)
    _require(bool(np.isfinite(norm).all()), f"{what}: non-finite quaternion")
    _require(float(norm.min()) > 1e-8, f"{what}: zero-norm quaternion")
    return q / norm[:, None]


def load_trajectory(path: Path) -> Trajectory:
    """Read one trajectory NPZ, validating shapes, finiteness and frame/time alignment."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"trajectory not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        missing = [k for k in REQUIRED_KEYS if k not in npz.files]
        _require(not missing, f"{path}: missing keys {missing}")

        dt = float(_scalar(npz, "dt", float("nan")))
        _require(math.isfinite(dt) and dt > 0.0, f"{path}: invalid dt {dt}")

        joint_names = [str(n) for n in np.asarray(npz["joint_names"]).reshape(-1)]
        n_joints = len(joint_names)
        _require(n_joints > 0, f"{path}: empty joint_names")
        _require(len(set(joint_names)) == n_joints, f"{path}: duplicate joint_names")

        sim_root_pos = np.asarray(npz["sim_root_pos"], dtype=np.float64)
        sim_root_quat = np.asarray(npz["sim_root_quat"], dtype=np.float64)
        sim_joint_pos = np.asarray(npz["sim_joint_pos"], dtype=np.float64)
        ref_root_pos = np.asarray(npz["ref_root_pos"], dtype=np.float64)
        ref_root_quat = np.asarray(npz["ref_root_quat"], dtype=np.float64)
        ref_joint_pos = np.asarray(npz["ref_joint_pos"], dtype=np.float64)
        sim_frame_idx = np.asarray(npz["sim_ref_frame_idx"], dtype=np.int64).reshape(-1)
        ref_frame_idx = np.asarray(npz["ref_frame_idx"], dtype=np.int64).reshape(-1)

        for arr, what, cols in (
            (sim_root_pos, "sim_root_pos", 3),
            (sim_root_quat, "sim_root_quat", 4),
            (ref_root_pos, "ref_root_pos", 3),
            (ref_root_quat, "ref_root_quat", 4),
        ):
            _require(arr.ndim == 2 and arr.shape[1] == cols, f"{path}: {what} must be (N, {cols}), got {arr.shape}")
            _require(bool(np.isfinite(arr).all()), f"{path}: {what} has non-finite values")
        for arr, what in ((sim_joint_pos, "sim_joint_pos"), (ref_joint_pos, "ref_joint_pos")):
            _require(arr.ndim == 2 and arr.shape[1] == n_joints, f"{path}: {what} must be (N, {n_joints}), got {arr.shape}")
            _require(bool(np.isfinite(arr).all()), f"{path}: {what} has non-finite values")

        n_sim = sim_root_pos.shape[0]
        n_ref = ref_root_pos.shape[0]
        _require(n_sim >= 1, f"{path}: empty sim trajectory")
        _require(n_ref >= 1, f"{path}: empty reference trajectory")
        _require(
            sim_joint_pos.shape[0] == n_sim and sim_frame_idx.shape[0] == n_sim,
            f"{path}: sim arrays disagree on length "
            f"({sim_root_pos.shape[0]}, {sim_joint_pos.shape[0]}, {sim_frame_idx.shape[0]})",
        )
        _require(
            ref_joint_pos.shape[0] == n_ref and ref_frame_idx.shape[0] == n_ref,
            f"{path}: ref arrays disagree on length "
            f"({ref_root_pos.shape[0]}, {ref_joint_pos.shape[0]}, {ref_frame_idx.shape[0]})",
        )

        sim_root_quat = _normalize_quat(sim_root_quat, f"{path}: sim_root_quat")
        ref_root_quat = _normalize_quat(ref_root_quat, f"{path}: ref_root_quat")

        t0 = int(ref_frame_idx[0])
        sim_time = (sim_frame_idx - t0).astype(np.float64) * dt
        ref_time = (ref_frame_idx - t0).astype(np.float64) * dt
        _require(bool((np.diff(sim_time) >= -1e-12).all()), f"{path}: sim_ref_frame_idx not monotonic")
        _require(bool((np.diff(ref_time) > 0).all()), f"{path}: ref_frame_idx not strictly increasing")

        steps = int(_scalar(npz, "steps", max(n_sim - 1, 0)))
        expected_steps = int(_scalar(npz, "expected_steps", max(n_ref - 1, 0)))
        failed = bool(_scalar(npz, "failed", False))
        failure_step = int(_scalar(npz, "failure_step", -1))
        _require(
            not failed or failure_step >= 0,
            f"{path}: failed=True but failure_step={failure_step}",
        )
        failure_time = float(_scalar(npz, "failure_time_s", float("nan")))
        if not math.isfinite(failure_time):
            failure_time = failure_step * dt if (failed and failure_step >= 0) else float("nan")
        # Terminal schema: current NPZs carry terminal_state_recorded=True; older ones
        # (env auto-reset era) only carry sim_rows_stop_before_final_state=True.
        if "terminal_state_recorded" in npz.files:
            terminal_state_recorded = bool(_scalar(npz, "terminal_state_recorded", True))
        else:
            terminal_state_recorded = not bool(_scalar(npz, "sim_rows_stop_before_final_state", False))
        if "termination_reasons" in npz.files:
            termination_reasons = [str(s) for s in np.asarray(npz["termination_reasons"]).reshape(-1) if str(s)]
        else:
            termination_reasons = []

    return Trajectory(
        name=path.stem,
        dt=dt,
        joint_names=joint_names,
        sim_root_pos=sim_root_pos,
        sim_root_quat=sim_root_quat,
        sim_joint_pos=sim_joint_pos,
        sim_time=sim_time,
        ref_root_pos=ref_root_pos,
        ref_root_quat=ref_root_quat,
        ref_joint_pos=ref_joint_pos,
        ref_time=ref_time,
        steps=steps,
        expected_steps=expected_steps,
        failed=failed,
        failure_step=failure_step,
        failure_time=failure_time,
        terminal_state_recorded=terminal_state_recorded,
        termination_reasons=termination_reasons,
    )


# --------------------------------------------------------------------------
# mujoco / model
# --------------------------------------------------------------------------


def _import_mujoco():
    """Import mujoco + the project's G1 loader, naming the GL dependency on failure."""
    try:
        import mujoco

        from supermimic.data.motion_convert import load_g1_model
    except Exception as exc:  # pragma: no cover - depends on the system GL stack
        raise RuntimeError(
            f"cannot initialise MuJoCo rendering with MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}: {exc}\n"
            "Backend requirements (this script never installs system libraries):\n"
            "  osmesa: system library libOSMesa.so.8 (Debian/Ubuntu package 'libosmesa6') for PyOpenGL;\n"
            "  egl:    EGL device enumeration (eglQueryDevicesEXT) must return a device; select one with\n"
            "          MUJOCO_EGL_DEVICE_ID and verify which physical device the enumeration indexes;\n"
            "          __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json with\n"
            "          GALLIUM_DRIVER=llvmpipe keeps it CPU software rendering;\n"
            "  glfw:   an X display (e.g. Xvfb) whose GLX server returns GLXFBConfigs.\n"
            "Pick one explicitly with --backend osmesa|egl|glfw (or set MUJOCO_GL)."
        ) from exc
    return mujoco, load_g1_model


def _joint_qpos_addrs(mujoco, model, joint_names: list[str]) -> np.ndarray:
    """Map NPZ joint columns to model qpos addresses by joint name (order-independent)."""
    addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' from the trajectory NPZ does not exist in the G1 model")
        addrs.append(int(model.jnt_qposadr[jid]))
    _require(len(set(addrs)) == len(addrs), "joint name -> qpos address mapping is not unique")
    _require(model.nq == 7 + len(addrs), f"G1 model nq={model.nq} != 7 + {len(addrs)} trajectory joints")
    return np.asarray(addrs, dtype=np.int64)


def _set_qpos(data, addrs: np.ndarray, root_pos, root_quat, joint_pos) -> None:
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[addrs] = joint_pos


# --------------------------------------------------------------------------
# cameras / layout / text
# --------------------------------------------------------------------------


def camera_setup(traj: Trajectory) -> tuple[dict, dict, list[float]]:
    """One fixed world-space camera covering both unmodified trajectories."""
    xy = np.concatenate([traj.sim_root_pos[:, :2], traj.ref_root_pos[:, :2]])
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    sim_xy = ref_xy = center
    zs = np.concatenate([traj.sim_root_pos[:, 2], traj.ref_root_pos[:, 2]])
    lookat_z = float(np.clip(np.median(zs), 0.35, 1.1))
    sim_radius = float(np.max(np.linalg.norm(traj.sim_root_pos[:, :2] - center, axis=1)))
    ref_radius = float(np.max(np.linalg.norm(traj.ref_root_pos[:, :2] - center, axis=1)))
    radius = max(sim_radius, ref_radius, 0.8) + 0.65
    distance = max(3.5, radius / math.tan(math.radians(CAM_FOVY_DEG / 2.0)) * 1.05)
    shared = {"lookat_z": lookat_z, "distance": distance}
    return (
        {"xy": sim_xy, **shared},
        {"xy": ref_xy, **shared},
        [sim_radius, ref_radius, distance],
    )


def add_ground_grid(mujoco, scene, center: np.ndarray, radius: float) -> None:
    """Render-only z=0 plane and metre grid; no model or recorded state changes."""
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_PLANE, np.array([0.0, 0.0, 0.1]),
        np.zeros(3), np.eye(3).ravel(), np.array([0.32, 0.36, 0.40, 1.0]),
    )
    geom.emission = 0.3
    scene.ngeom += 1
    lo = np.floor(center - radius - 2).astype(int)
    hi = np.ceil(center + radius + 2).astype(int)
    for axis in (0, 1):
        for value in range(lo[axis], hi[axis] + 1):
            start = np.array([float(lo[0]), float(lo[1]), 0.001])
            end = np.array([float(hi[0]), float(hi[1]), 0.001])
            start[axis] = end[axis] = value
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                np.zeros(3), np.eye(3).ravel(), np.array([0.5, 0.55, 0.60, 1.0]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, start, end)
            scene.ngeom += 1


def tint_reference_geoms(scene, first_geom: int, alpha: float) -> None:
    """Cyan ghost tint for the reference geoms appended after ``first_geom`` only.

    The actual rollout's geoms (everything before ``first_geom``) keep their original
    materials, textures and opaque colours. matid/texid are dropped on the tinted
    geoms so no material or texture colour can override the cyan/alpha, and
    ``transparent`` marks them for the renderer's blended pass.
    """
    for i in range(first_geom, int(scene.ngeom)):
        geom = scene.geoms[i]
        geom.rgba[0] = REFERENCE_TINT[0]
        geom.rgba[1] = REFERENCE_TINT[1]
        geom.rgba[2] = REFERENCE_TINT[2]
        geom.rgba[3] = alpha
        geom.matid = -1
        geom.texid = -1
        geom.texcoord = -1
        geom.emission = 0.15
        geom.specular = 0.0
        geom.shininess = 0.0
        geom.reflectance = 0.0
        geom.transparent = 1


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in _FONT_CANDIDATES:
        if Path(cand).is_file():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    return ImageFont.load_default()


def footer_lines(
    traj: Trajectory,
    t: float,
    sim_frozen: bool,
    truncated: bool,
    in_tail: bool = False,
    failure_tail_s: float = 0.0,
    actual_failure_tail_s: float | None = None,
    layout: str = "side-by-side",
) -> list[tuple[str, tuple[int, int, int]]]:
    """Bottom-band text: clip / roles / time / step, failure state, replay caveats."""
    actual_tail = float(failure_tail_s) if actual_failure_tail_s is None else float(actual_failure_tail_s)
    shown_step = min(int(round(t / traj.dt)), traj.steps)
    line1 = f"clip={traj.name}   t={t:5.2f}s / {traj.ref_end_time:.2f}s   step={shown_step}/{traj.expected_steps}"
    if layout != "overlay":  # overlay has one view: the legend states the roles
        line1 += "   [left=sim actual | right=reference]"
    reasons = f" [{'+'.join(traj.termination_reasons)}]" if traj.termination_reasons else ""
    # Legacy (pre-terminal-recording) NPZs never captured the terminal state: never
    # call that frozen row a terminal state.
    state_word = "TERMINAL STATE" if traj.terminal_state_recorded else "LAST RECORDED STATE"
    if traj.failed:
        if in_tail:
            line2 = (
                f"FAILED at step {traj.failure_step} (t={traj.failure_time:.2f}s){reasons}  -  "
                f"{state_word}; comparison frozen; video ends after {actual_tail:.2f}s actual tail"
            )
        elif t + 1e-9 >= traj.failure_time or sim_frozen:
            line2 = (
                f"FAILED at step {traj.failure_step} (t={traj.failure_time:.2f}s){reasons}  -  "
                f"ACTUAL HAS FAILED ({state_word}), not playing on"
            )
        else:
            line2 = f"recorded failure: step {traj.failure_step} (t={traj.failure_time:.2f}s){reasons}"
        color2 = STATUS_FAIL_COLOR
    elif traj.steps >= traj.expected_steps and t + 1e-9 >= traj.sim_end_time:
        line2 = f"clip completed ({traj.steps}/{traj.expected_steps} steps)"
        color2 = STATUS_OK_COLOR
    elif sim_frozen:
        line2 = f"actual frozen at last recorded state ({traj.steps} steps); reference keeps playing"
        color2 = NOTE_COLOR
    else:
        line2 = f"running ({traj.steps} recorded steps, {traj.expected_steps} expected)"
        color2 = LABEL_COLOR
    parts = ["closed-loop recorded trajectory replay (no policy recompute)"]
    if not traj.terminal_state_recorded:
        parts.append("terminal state not recorded (env auto-reset)")
    if truncated:
        parts.append("reference truncated at --max-seconds")
    if traj.failed and failure_tail_s >= 0.0:
        if actual_tail + 1e-9 < failure_tail_s:
            parts.append(
                f"failed clip: requested {failure_tail_s:.2f}s tail; actual {actual_tail:.2f}s after failure"
            )
        else:
            parts.append(f"failed clip: video ends {actual_tail:.2f}s after failure")
    line3 = " | ".join(parts)
    return [(line1, LABEL_COLOR), (line2, color2), (line3, MUTED_COLOR if traj.terminal_state_recorded else NOTE_COLOR)]


def actual_failure_tail(video_duration: float, failure_time: float | None) -> float:
    """Frozen tail actually present in the video (may be clipped by ref/max duration)."""
    if failure_time is None or not math.isfinite(float(failure_time)):
        return 0.0
    return max(0.0, float(video_duration) - float(failure_time))


def compose_frame(
    left: np.ndarray,
    right: np.ndarray | None,
    width: int,
    height: int,
    panel_h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    lines: list[tuple[str, tuple[int, int, int]]],
    actual_badge: str | None = None,
    badge_bg: tuple[int, int, int] = (140, 20, 20),
    layout: str = "side-by-side",
    iteration: int | None = None,
    reference_alpha: float = 0.35,
) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(Image.fromarray(left), (0, 0))
    draw = ImageDraw.Draw(canvas)
    top = 4
    if iteration is not None:
        # Prominent checkpoint label, top-left over the scene.
        title_font = _load_font(max(20, (font.size if hasattr(font, "size") else 16) + 8))
        title_size = getattr(title_font, "size", 20)
        title = f"Iteration {iteration:,}"
        draw.rectangle([4, top, 14 + int(title_font.getlength(title)), top + title_size + 12], fill=TEXT_BG)
        draw.text((10, top + 6), title, font=title_font, fill=LABEL_COLOR)
        top += title_size + 18
    if layout == "overlay":
        # Legend top-right (never over the robot): opaque actual vs the cyan reference
        # ghost; the swatch shows the tint at the true --reference-alpha.
        pct = int(round(reference_alpha * 100))
        ghost_swatch = tuple(
            round(reference_alpha * c * 255.0 + (1.0 - reference_alpha) * b)
            for c, b in zip(REFERENCE_TINT, TEXT_BG)
        )
        entries = [((170, 175, 185), "actual opaque"), (ghost_swatch, f"reference cyan {pct}% opacity")]
        swatch = 12
        widths = [swatch + 5 + int(small_font.getlength(text)) for _, text in entries]
        x = width - 4 - (10 + sum(widths) + 12 * (len(entries) - 1))
        draw.rectangle([x, 4, width - 4, 26], fill=TEXT_BG)
        for (color, text), entry_w in zip(entries, widths):
            draw.rectangle([x + 5, 8, x + 5 + swatch, 20], fill=color)
            draw.text((x + 5 + swatch + 5, 8), text, font=small_font, fill=LABEL_COLOR)
            x += entry_w + 12
    else:
        canvas.paste(Image.fromarray(right), (left.shape[1], 0))
        draw = ImageDraw.Draw(canvas)
        # Panel role labels (also repeated in the footer so the roles are unambiguous).
        for x, text in ((4, "ACTUAL (sim, closed-loop recorded)"), (left.shape[1] + 4, "REFERENCE")):
            draw.rectangle([x, top, x + int(small_font.getlength(text)) + 10, top + 22], fill=TEXT_BG)
            draw.text((x + 5, top + 4), text, font=small_font, fill=LABEL_COLOR)
        top += 26
    # Badge on the actual view so its frozen/failed state can never be mistaken for
    # the reference's later motion.
    if actual_badge:
        draw.rectangle([4, top, int(small_font.getlength(actual_badge)) + 14, top + 22], fill=badge_bg)
        draw.text((9, top + 4), actual_badge, font=small_font, fill=(255, 235, 235))
    # Footer band.
    draw.rectangle([0, panel_h, width, height], fill=(24, 24, 30))
    y = panel_h + 6
    for text, color in lines:
        draw.text((6, y), text, font=font, fill=color)
        y += font.size + 6 if hasattr(font, "size") else 22
    return np.asarray(canvas)


# --------------------------------------------------------------------------
# rendering loop
# --------------------------------------------------------------------------


def render_video(
    traj: Trajectory,
    out_path: Path,
    width: int,
    height: int,
    max_seconds: float,
    fps: float,
    failure_tail_seconds: float = 0.5,
    layout: str = "side-by-side",
    iteration: int | None = None,
    reference_alpha: float = 0.35,
    robot_variant: str = "g1",
) -> dict:
    if layout not in ("overlay", "side-by-side"):
        raise ValueError(f"unknown layout {layout!r} (expected 'overlay' or 'side-by-side')")
    if not (math.isfinite(reference_alpha) and 0.0 < reference_alpha < 1.0):
        raise ValueError(f"reference_alpha must be in (0, 1), got {reference_alpha}")
    if layout == "overlay" and iteration is None:
        raise ValueError("layout 'overlay' requires an iteration for the top-left video label")
    mujoco, load_g1_model = _import_mujoco()
    model = load_g1_model(robot_variant=robot_variant)
    addrs = _joint_qpos_addrs(mujoco, model, traj.joint_names)
    identical_order = traj.joint_names == [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)) for i in range(1, model.njnt)
    ]

    font_size = max(13, min(20, height // 40))
    footer_h = 3 * (font_size + 6) + 12
    panel_h = height - footer_h
    if panel_h < 64:
        raise ValueError(f"--height {height} too small for the text footer ({footer_h}px); use >= 240")
    panel_w = (width,) if layout == "overlay" else (width // 2, width - width // 2)
    font = _load_font(font_size)
    small_font = _load_font(max(12, font_size - 3))

    sim_cam_cfg, ref_cam_cfg, cam_info = camera_setup(traj)
    cams = []
    for cfg in (sim_cam_cfg, ref_cam_cfg):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [float(cfg["xy"][0]), float(cfg["xy"][1]), float(cfg["lookat_z"])]
        cam.distance = float(cfg["distance"])
        cam.azimuth = CAM_AZIMUTH_DEG
        cam.elevation = CAM_ELEVATION_DEG
        cams.append(cam)
    model.vis.global_.fovy = CAM_FOVY_DEG  # per-camera FOV does not exist on MjvCamera
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.65

    scene_opt = mujoco.MjvOption()
    scene_opt.geomgroup[3] = 0  # hide the collision capsules, keep the visual meshes

    # Offscreen framebuffer must cover the panels (in-memory render setting only).
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), *panel_w)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), panel_h)

    # Failed clips: end --failure-tail-seconds after the failure so the viewer is not
    # left staring at a long frozen actual (it looked like a stuck policy). During the
    # tail both panels freeze at the failure moment with an explicit banner.
    capped_by_max_seconds = traj.ref_end_time > float(max_seconds) + 1e-9
    tail_s = 0.0
    duration_caps = [traj.ref_end_time, float(max_seconds)]
    if traj.failed and math.isfinite(traj.failure_time):
        tail_s = max(float(failure_tail_seconds), 0.0)
        duration_caps.append(traj.failure_time + tail_s)
    video_duration = min(duration_caps)
    actual_tail_s = actual_failure_tail(video_duration, traj.failure_time if traj.failed else None)
    n_frames = max(1, int(math.floor(video_duration * fps + 1e-9)) + 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    # Overlay: the reference ghost is appended from its own MjData (the saved
    # reference state), so the two recordings never share or overwrite each other.
    ref_data = mujoco.MjData(model) if layout == "overlay" else None
    ref_pert = mujoco.MjvPerturb()
    renderers = [mujoco.Renderer(model, height=panel_h, width=w) for w in panel_w]
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    try:
        for i in range(n_frames):
            t = i / fps
            ref_i = int(np.searchsorted(traj.ref_time, t, side="right")) - 1
            ref_i = min(max(ref_i, 0), len(traj.ref_time) - 1)
            sim_i = int(np.searchsorted(traj.sim_time, t, side="right")) - 1
            sim_i = min(max(sim_i, 0), len(traj.sim_time) - 1)
            sim_frozen = t > traj.sim_end_time + 1e-9
            in_tail = traj.failed and math.isfinite(traj.failure_time) and t > traj.failure_time + 1e-9
            if in_tail:
                # Failure tail: both panels freeze at the failure moment (explicit
                # left/right comparison freeze; the recording never rolls on).
                ref_i = int(np.searchsorted(traj.ref_time, traj.failure_time, side="right")) - 1
                ref_i = min(max(ref_i, 0), len(traj.ref_time) - 1)
                sim_i = len(traj.sim_time) - 1

            if layout == "overlay":
                renderer = renderers[0]
                _set_qpos(data, addrs, traj.sim_root_pos[sim_i], traj.sim_root_quat[sim_i], traj.sim_joint_pos[sim_i])
                mujoco.mj_forward(model, data)
                # Display-only shared camera follows the midpoint of the two ORIGINAL world-space roots.
                midpoint = (traj.sim_root_pos[sim_i] + traj.ref_root_pos[ref_i]) * 0.5
                target = np.array([midpoint[0], midpoint[1], np.clip(midpoint[2] + 0.2, 0.6, 1.1)])
                separation = float(np.linalg.norm(traj.sim_root_pos[sim_i, :2] - traj.ref_root_pos[ref_i, :2]))
                desired_distance = max(3.6, 2.8 + separation / math.tan(math.radians(CAM_FOVY_DEG / 2)))
                if i == 0:
                    cams[0].lookat[:] = target
                    cams[0].distance = desired_distance
                else:
                    cams[0].lookat[:] = 0.85 * cams[0].lookat + 0.15 * target
                    cams[0].distance = 0.85 * cams[0].distance + 0.15 * desired_distance
                renderer.update_scene(data, scene_option=scene_opt, camera=cams[0])
                # Reference ghost: its own MjData at the saved reference pose, appended
                # as scene geometry and tinted; no image blending, no recording changes.
                first_ref_geom = renderer.scene.ngeom
                _set_qpos(ref_data, addrs, traj.ref_root_pos[ref_i], traj.ref_root_quat[ref_i], traj.ref_joint_pos[ref_i])
                mujoco.mj_forward(model, ref_data)
                mujoco.mjv_addGeoms(
                    model, ref_data, scene_opt, ref_pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene
                )
                tint_reference_geoms(renderer.scene, first_ref_geom, reference_alpha)
                add_ground_grid(mujoco, renderer.scene, cams[0].lookat[:2], 5.0)
                panels = [renderer.render()]
            else:
                panels = []
                for stream, cam, renderer in (
                    (0, cams[0], renderers[0]),
                    (1, cams[1], renderers[1]),
                ):
                    if stream == 0:
                        _set_qpos(data, addrs, traj.sim_root_pos[sim_i], traj.sim_root_quat[sim_i], traj.sim_joint_pos[sim_i])
                    else:
                        _set_qpos(data, addrs, traj.ref_root_pos[ref_i], traj.ref_root_quat[ref_i], traj.ref_joint_pos[ref_i])
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, scene_option=scene_opt, camera=cam)
                    add_ground_grid(mujoco, renderer.scene, cams[0].lookat[:2], 5.0)
                    panels.append(renderer.render())
            lines = footer_lines(
                traj, t, sim_frozen=sim_frozen, truncated=capped_by_max_seconds,
                in_tail=in_tail, failure_tail_s=tail_s, actual_failure_tail_s=actual_tail_s,
                layout=layout,
            )
            state_word = "TERMINAL STATE" if traj.terminal_state_recorded else "LAST RECORDED STATE"
            badge = None
            badge_bg = (140, 20, 20)
            if traj.failed and in_tail:
                badge = f"ACTUAL: FAILED at step {traj.failure_step} - {state_word}, video ends"
            elif traj.failed and (t + 1e-9 >= traj.failure_time or sim_frozen):
                badge = f"ACTUAL: FAILED at step {traj.failure_step} - {state_word}, not playing on"
            elif traj.steps >= traj.expected_steps and t + 1e-9 >= traj.sim_end_time:
                badge = f"ACTUAL: completed ({traj.steps}/{traj.expected_steps} steps)"
                badge_bg = (30, 90, 40)
            elif sim_frozen:
                badge = "ACTUAL: frozen at last recorded state"
            frame = compose_frame(
                panels[0], panels[1] if layout != "overlay" else None,
                width, height, panel_h, font, small_font, lines,
                actual_badge=badge, badge_bg=badge_bg,
                layout=layout, iteration=iteration, reference_alpha=reference_alpha,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        for r in renderers:
            r.close()

    return {
        "out": str(out_path),
        "frames": n_frames,
        "duration_s": n_frames / fps,
        "fps": fps,
        "size": (width, height),
        "panel_size": (panel_w[0], panel_h),
        "layout": layout,
        "iteration": iteration,
        "reference_alpha": reference_alpha,
        "robot_variant": robot_variant,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "joint_order_matches_model": identical_order,
        "n_sim_rows": int(len(traj.sim_time)),
        "n_ref_frames": int(len(traj.ref_time)),
        "failed": traj.failed,
        "failure_step": traj.failure_step,
        "termination_reasons": list(traj.termination_reasons),
        "terminal_state_recorded": traj.terminal_state_recorded,
        "failure_tail_seconds": tail_s,
        "requested_failure_tail_seconds": tail_s,
        "actual_failure_tail_seconds": actual_tail_s,
        "truncated_at_max_seconds": capped_by_max_seconds,
        "camera_mode": "shared midpoint follow; original world-space trajectories preserved",
    }


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory", type=Path, required=True, help="trajectory NPZ from evaluate_hard_suite.py")
    p.add_argument("--out", type=Path, required=True, help="output MP4 path")
    p.add_argument("--label", default=None, help="explicit clip and evaluation label displayed in the video")
    p.add_argument(
        "--layout",
        choices=("overlay", "side-by-side"),
        default="side-by-side",
        help="overlay: one full-width camera view, the actual's opaque meshes plus the cyan reference "
        "ghost at its saved pose; side-by-side (default): the original two-panel comparison",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="checkpoint iteration shown top-left on the video (45800 -> 'Iteration 45,800'); "
        "required for --layout overlay",
    )
    p.add_argument(
        "--reference-alpha",
        type=float,
        default=0.35,
        help="opacity of the cyan reference ghost in --layout overlay, in (0, 1)",
    )
    p.add_argument(
        "--backend",
        choices=("osmesa", "egl", "glfw"),
        default=None,
        help="MuJoCo GL backend -> MUJOCO_GL (default osmesa; an existing MUJOCO_GL wins without this flag)",
    )
    p.add_argument("--width", type=int, default=1280, help="total video width, must be <= 1280")
    p.add_argument("--height", type=int, default=720, help="total video height (includes the text footer)")
    p.add_argument(
        "--robot-variant",
        choices=("g1", "g1_no_plastic_hands"),
        default="g1",
        help="G1 visual/collision variant used for the replay model (default: g1)",
    )
    p.add_argument("--max-seconds", type=float, default=15.0, help="cap on the video duration in seconds")
    p.add_argument("--fps", type=float, default=25.0, help="output frames per second (reference plays in real time)")
    p.add_argument(
        "--failure-tail-seconds",
        type=float,
        default=0.5,
        help="for failed clips the video ends this long after the failure (both panels frozen at the "
        "terminal comparison); keeps a long static actual from looking like a stuck policy",
    )
    args = p.parse_args()
    if not (2 <= args.width <= 1280):
        p.error("--width must be in [2, 1280] (total width)")
    if args.width % 2 or args.height % 2:
        p.error("--width and --height must be even (yuv420p)")
    if args.height < 240:
        p.error("--height must be >= 240")
    if not (math.isfinite(args.max_seconds) and args.max_seconds > 0):
        p.error("--max-seconds must be > 0")
    if not (math.isfinite(args.fps) and args.fps > 0):
        p.error("--fps must be > 0")
    if not (math.isfinite(args.failure_tail_seconds) and args.failure_tail_seconds >= 0):
        p.error("--failure-tail-seconds must be >= 0")
    if not (math.isfinite(args.reference_alpha) and 0.0 < args.reference_alpha < 1.0):
        p.error("--reference-alpha must be in (0, 1)")
    if args.layout == "overlay" and args.iteration is None:
        p.error("--iteration is required for --layout overlay")
    return args


def main() -> None:
    args = parse_args()
    if args.backend:
        # Explicit flag wins over any inherited MUJOCO_GL; mujoco is imported later.
        os.environ["MUJOCO_GL"] = args.backend
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite video: {args.out}")
    traj = load_trajectory(args.trajectory)
    if args.label:
        traj.name = args.label
    summary = render_video(
        traj,
        args.out,
        width=args.width,
        height=args.height,
        max_seconds=args.max_seconds,
        fps=args.fps,
        failure_tail_seconds=args.failure_tail_seconds,
        layout=args.layout,
        iteration=args.iteration,
        reference_alpha=args.reference_alpha,
        robot_variant=args.robot_variant,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    frozen_phrase = "both panels frozen" if args.layout == "side-by-side" else "the overlay frozen"
    requested_tail = summary["requested_failure_tail_seconds"]
    actual_tail = summary["actual_failure_tail_seconds"]
    tail_note = f"{actual_tail:.2f}s actual tail"
    if requested_tail > actual_tail + 1e-9:
        tail_note += f" (requested {requested_tail:.2f}s; capped by reference/clip end)"
    if traj.failed:
        reasons = f" [{'+'.join(traj.termination_reasons)}]" if traj.termination_reasons else ""
        print(
            f"[status] clip FAILED at step {traj.failure_step} (t={traj.failure_time:.2f}s){reasons}; "
            f"video ends with {tail_note} after the failure with {frozen_phrase} at "
            "the terminal comparison; it does not pretend completion"
        )
    if not traj.terminal_state_recorded:
        print("[status] terminal state was not recorded in this NPZ (pre-terminal-recording schema); video labels it")
    if traj.sim_end_time < traj.ref_end_time - 1e-9:
        if traj.failed:
            print(
                f"[status] sim rows end at t={traj.sim_end_time:.2f}s while the reference runs to "
                f"{traj.ref_end_time:.2f}s: the actual failed; the video ends with {tail_note} "
                f"after the failure with {frozen_phrase} at the failure comparison "
                "(the reference is not played on as if the policy continued)"
            )
        else:
            print(
                f"[status] sim rows end at t={traj.sim_end_time:.2f}s while the reference runs to "
                f"{traj.ref_end_time:.2f}s: the actual freezes, the reference keeps playing"
            )


if __name__ == "__main__":
    main()
