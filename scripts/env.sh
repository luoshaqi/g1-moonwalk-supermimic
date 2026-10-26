# Source from Bash after setup; affects only the current shell.
MOONWALK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUPERMIMIC_ROOT="${SUPERMIMIC_ROOT:-$MOONWALK_ROOT/upstream}"
export MOONWALK_PYTHON="${MOONWALK_PYTHON:-$MOONWALK_ROOT/.venv/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SUPERMIMIC_ROOT/src:$SUPERMIMIC_ROOT/vendor/mjlab/src${PYTHONPATH:+:$PYTHONPATH}"
# Let Warp find the pip-installed CUDA runtime. Keep the host driver paths intact.
MOONWALK_CUDA_LIBS="$("$MOONWALK_PYTHON" -c 'import site,glob,os; print(os.pathsep.join(d for p in site.getsitepackages() for d in glob.glob(p+"/nvidia/*/lib")))')"
if [[ -n "$MOONWALK_CUDA_LIBS" ]]; then
  export LD_LIBRARY_PATH="$MOONWALK_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
MOONWALK_PY_INCLUDE="$("$MOONWALK_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
export CPATH="$MOONWALK_PY_INCLUDE${CPATH:+:$CPATH}"
unset MOONWALK_CUDA_LIBS MOONWALK_PY_INCLUDE
