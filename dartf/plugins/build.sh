#!/usr/bin/env bash
# Build lq_plugins.so (TensorRT 10 IPluginV3 plugins) and lq_util.so (tracker helpers).
# usage: ./build.sh                         (GPU architecture detected with nvidia-smi; Jetson without it: sm_87)
#        ARCH=86 ./build.sh                 (80/86/87/89/90; override any of ARCH, CUTLASS_DIR, TRT_INC, TRT_LIB, NVCC)
# needs CUTLASS 3.5.1 (a full clone: headers and tools/util), TensorRT 10 headers and libraries, CUDA 12.4+.
set -euo pipefail
cd "$(dirname "$0")"
MACH=$(uname -m)-linux-gnu

if [ -z "${ARCH:-}" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)
  [ -n "$ARCH" ] || ARCH=87
fi
case "$ARCH" in 86|89) STAGES=${STAGES:-3};; *) STAGES=${STAGES:-5};; esac   # 99 KB smem per block on sm86/sm89

# CUTLASS: CUTLASS_DIR is the repository root; the older CUTLASS (= <root>/include) is still accepted
if [ -n "${CUTLASS:-}" ]; then CUTLASS_DIR=${CUTLASS_DIR:-$CUTLASS/..}; fi
CUTLASS_DIR=${CUTLASS_DIR:-../third_party/cutlass}
for f in include/cutlass/cutlass.h tools/util/include/cutlass/util/host_reorder.h; do
  [ -f "$CUTLASS_DIR/$f" ] || { echo "CUTLASS not found: $CUTLASS_DIR/$f is missing. Clone it with"; \
    echo "  git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git $(cd .. && pwd)/third_party/cutlass"; exit 1; }
done

first() { for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done; }
PYTRT_LIB=$(python3 -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null || true)
TRT_INC=${TRT_INC:-$(dirname "$(first ${TENSORRT_ROOT:+$TENSORRT_ROOT/include/NvInfer.h} /usr/include/$MACH/NvInfer.h /usr/include/NvInfer.h /usr/local/tensorrt/include/NvInfer.h)" 2>/dev/null || true)}
TRT_LIB=${TRT_LIB:-$(dirname "$(first ${TENSORRT_ROOT:+$TENSORRT_ROOT/lib/libnvinfer.so} /usr/lib/$MACH/libnvinfer.so /usr/local/tensorrt/lib/libnvinfer.so ${PYTRT_LIB:+$PYTRT_LIB/libnvinfer.so.10})" 2>/dev/null || true)}
[ -n "$TRT_INC" ] && [ -f "$TRT_INC/NvInfer.h" ] || { echo "TensorRT headers not found: set TRT_INC (directory with NvInfer.h) or TENSORRT_ROOT"; exit 1; }
[ -n "$TRT_LIB" ] && [ -d "$TRT_LIB" ] || { echo "TensorRT libraries not found: set TRT_LIB (directory with libnvinfer.so) or TENSORRT_ROOT"; exit 1; }
LNVINFER=-lnvinfer; [ -e "$TRT_LIB/libnvinfer.so" ] || LNVINFER=-l:libnvinfer.so.10        # pip wheels ship only the versioned name

NVCC=${NVCC:-$(command -v nvcc || first /usr/local/cuda/bin/nvcc $(ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1))}
[ -n "$NVCC" ] && [ -x "$NVCC" ] || { echo "nvcc not found: set NVCC or add CUDA to PATH"; exit 1; }
CUDA_LIB=$(dirname "$NVCC")/../lib64

echo "sm_$ARCH (stages $STAGES) | CUTLASS $CUTLASS_DIR | TensorRT $TRT_INC, $TRT_LIB | $NVCC"
cp ../kernels/lq_attn_kernel.cuh . 2>/dev/null || true
$NVCC -O3 -std=c++17 -arch=sm_$ARCH --use_fast_math --expt-relaxed-constexpr -DLQ_STAGES=$STAGES -Xcompiler -fPIC -shared \
  -I "$CUTLASS_DIR/include" -I "$CUTLASS_DIR/tools/util/include" -I "$TRT_INC" \
  lq_plugins.cu lq_attn_plugin.cu lq_block_plugins.cu lq_ground_plugins.cu lq_memattn_plugin.cu -o lq_plugins.so \
  -L "$TRT_LIB" -L "$CUDA_LIB" $LNVINFER -lcublasLt -lcublas -Xlinker -rpath -Xlinker "$TRT_LIB"
echo "built plugins/lq_plugins.so for sm_$ARCH"
python3 check_plugin.py "$(pwd)/lq_plugins.so" || true

# torch-free tracker helpers (row gather / transpose kernels), loaded through ctypes by demo/sam3_track_np.py
$NVCC -O3 -arch=sm_$ARCH -Xcompiler -fPIC -shared lq_util.cu -o lq_util.so && echo "built plugins/lq_util.so"
