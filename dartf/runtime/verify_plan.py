"""Compare a vision plan against the FP32 truth (fpn_2 rel-L2 and cosine) on one image, and print its latency.
The FP32 truth comes from onnxruntime on an FP32 ONNX graph (e.g. the pipeline's exact rewrite vision_1008_hf_hadnw.onnx) and is
cached next to it, or from a saved .npy.
usage: verify_plan.py <plan> --image IMG (--onnx FP32.onnx | --golden fpn2.npy) [--size 1008] [--max-rel 0.15]"""
import os, sys
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d) for d in ("export", "runtime")]
import argparse, hashlib, numpy as np
from trt_util import load_engine, Runner
from preprocess import load_image_tensor

ap = argparse.ArgumentParser()
ap.add_argument("plan"); ap.add_argument("--image", required=True); ap.add_argument("--onnx"); ap.add_argument("--golden")
ap.add_argument("--size", type=int, default=1008); ap.add_argument("--max-rel", type=float, default=0.15, help="exit with an error above this rel-L2")
a = ap.parse_args()
assert a.onnx or a.golden, "give --onnx (FP32 graph) or --golden (saved fpn_2 .npy)"
x, _ = load_image_tensor(a.image, a.size)
if a.golden:
    ref = np.load(a.golden)
else:
    key = hashlib.sha1(f"{os.path.abspath(a.image)}:{a.size}".encode()).hexdigest()[:10]
    cache = os.path.splitext(a.onnx)[0] + f".fpn2_{key}.npy"
    if os.path.exists(cache):
        ref = np.load(cache)
    else:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
        ref = sess.run(["fpn_2"], {sess.get_inputs()[0].name: x})[0].astype(np.float32); np.save(cache, ref)
r = Runner(load_engine(a.plan)); y = r({"images": x}, want=["fpn_2"])["fpn_2"].astype(np.float32)
rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
cos = float(y.ravel().astype(np.float64) @ ref.ravel().astype(np.float64) / (np.linalg.norm(y) * np.linalg.norm(ref)))
print(f"VERIFY {os.path.basename(a.plan)}: fpn_2 vs FP32 rel={rel:.4f} cos={cos:.5f} | {r.time({'images': x}, 10, 40):.1f} ms")
if rel > a.max_rel: sys.exit(f"verification failed: rel {rel:.4f} > {a.max_rel}")
