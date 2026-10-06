"""Energy per frame (Jetson): run a vision engine continuously for N seconds while sampling tegrastats; report mean VDD_GPU_SOC /
VDD_CPU_CV / VIN_SYS_5V0 power and J/frame.  usage: energy_bench.py <vision.plan> <image> [seconds] [--log energy.jsonl]"""
import os as _os, sys as _sys; _sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("export", "runtime")]   # repo-relative imports
import subprocess, time, re, sys, os, numpy as np, json
from trt_util import load_engine, Runner
from preprocess import load_image_tensor
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("plan"); ap.add_argument("image"); ap.add_argument("seconds", nargs="?", type=float, default=20); ap.add_argument("--log", help="append the JSON result to this file")
a = ap.parse_args(); plan, secs = a.plan, a.seconds
r=Runner(load_engine(plan))
assert "images" in r.inputs or "pixel_values" in r.inputs, "energy_bench measures vision engines (input images / pixel_values)"
x,_=load_image_tensor(a.image); feeds={r.inputs[0]:x}
for n in r.inputs:
    buf,shape,dt=r.bufs[n]; buf.upload(np.ascontiguousarray(feeds[n],dtype=dt))
ptrs=[r.bufs[n][0].ptr for n in r.names]
for _ in range(10): r.ctx.execute_v2(ptrs)
from cuda_alloc import DevBuf; DevBuf.sync()
p=subprocess.Popen(["tegrastats","--interval","200"],stdout=subprocess.PIPE,text=True); time.sleep(1.0)
t0=time.perf_counter(); n=0
while time.perf_counter()-t0<secs: r.ctx.execute_v2(ptrs); n+=1
DevBuf.sync(); t1=time.perf_counter(); p.terminate(); out=p.stdout.read()
def mean(key):
    v=[float(m.group(1)) for m in re.finditer(key+r" (\d+)mW",out)]; return np.mean(v)/1000 if v else float("nan")
gpu,cpu,sys5=mean("VDD_GPU_SOC"),mean("VDD_CPU_CV"),mean("VIN_SYS_5V0"); fps=n/(t1-t0)
res={"plan":os.path.basename(os.path.dirname(plan))+"/"+os.path.basename(plan),"frames":n,"ms_per_frame":1000/fps,"W_gpu_soc":gpu,"W_cpu_cv":cpu,"W_sys5v":sys5,"J_per_frame_gpu_soc":gpu/fps,"J_per_frame_sys":sys5/fps}
print(json.dumps(res))
if a.log: open(a.log,"a").write(json.dumps(res)+"\n")
