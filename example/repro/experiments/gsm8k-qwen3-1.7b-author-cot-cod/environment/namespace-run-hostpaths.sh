#!/usr/bin/env bash
set -euo pipefail
RLVR_VLLM_RUNTIME_ROOT=/opt/experiment-runtime
RLVR_LAB_ROOT=/workspace/experiment
RLVR_BASE_PREFIX=/opt/base-runtime
[[ $# -gt 0 ]] || { echo "Supply a guest command." >&2; exit 2; }
RLVR_OPTIONAL_ENV=()
if [[ -n "${HF_ENDPOINT:-}" ]]; then RLVR_OPTIONAL_ENV+=("HF_ENDPOINT=$HF_ENDPOINT"); fi
if [[ -n "${VLLM_USE_V1:-}" ]]; then RLVR_OPTIONAL_ENV+=("VLLM_USE_V1=$VLLM_USE_V1"); fi
exec "$RLVR_VLLM_RUNTIME_ROOT/bin/proot" -0 -r "$RLVR_VLLM_RUNTIME_ROOT/rootfs" \
  -b "$RLVR_VLLM_RUNTIME_ROOT/rootfs:$RLVR_VLLM_RUNTIME_ROOT/rootfs" \
  -b /dev -b /proc -b /sys -b /etc/resolv.conf -b /etc/hosts \
  -b "$RLVR_LAB_ROOT:/workspace" -b "$RLVR_BASE_PREFIX:$RLVR_BASE_PREFIX" \
  -b "$RLVR_LAB_ROOT:$RLVR_LAB_ROOT" \
  -b "$RLVR_VLLM_RUNTIME_ROOT/evidence:/run-evidence" \
  -b "$RLVR_VLLM_RUNTIME_ROOT/cache:/run-cache" \
  -b "$RLVR_VLLM_RUNTIME_ROOT/wheelhouse:/wheelhouse" -w "$RLVR_LAB_ROOT" \
  /usr/bin/env -i \
  PATH=/opt/rlvr-vllm-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/opt/nvidia-driver \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  HTTP_PROXY="${HTTP_PROXY:-}" HTTPS_PROXY="${HTTPS_PROXY:-}" ALL_PROXY="${ALL_PROXY:-}" NO_PROXY="${NO_PROXY:-}" \
  http_proxy="${http_proxy:-}" https_proxy="${https_proxy:-}" all_proxy="${all_proxy:-}" no_proxy="${no_proxy:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  PYTHONPATH="$RLVR_LAB_ROOT/src" PYTHONDONTWRITEBYTECODE=1 \
  PIP_CONFIG_FILE=/dev/null PIP_REQUIRE_VIRTUALENV=1 PIP_NO_INPUT=1 \
  PIP_CACHE_DIR=/run-cache/pip HF_HOME=/run-cache/huggingface \
  CUDA_CACHE_PATH=/run-cache/cuda TRITON_CACHE_DIR=/run-cache/triton \
  TORCHINDUCTOR_CACHE_DIR=/run-cache/torchinductor VLLM_CACHE_ROOT=/run-cache/vllm \
  "${RLVR_OPTIONAL_ENV[@]}" \
  "$@"
