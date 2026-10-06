"""Structural calibration for HF-export graphs (any block count): marks the 6 GEMM inputs per block (q/k/v share one)
in a TensorRT FP32 engine, runs the PTQ images, writes calib_<tag>.json/npz keyed calib.block{i}.{fam} (i = graph
block index) with absmax/p99999/chmax — the format quantize_hf.py consumes; outputs go to the working directory.
Usage: CALIB_IMAGES=<dir of <id>.jpg> [CALIB_IDS=ids.json] collect_calib_hf.py <onnx> <tag> [size] [neck]   (ids default: calib/calib_ids.json)"""
import os as _os, sys as _sys; _sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("export", "runtime")]   # repo-relative imports
import tensorrt as trt, numpy as np, json, sys, os, time, onnx, collections
from trt_util import load_engine, Runner
from preprocess import load_image_tensor
IMAGES=os.environ.get("CALIB_IMAGES"); assert IMAGES, "set CALIB_IMAGES to the directory holding the calibration images (<id>.jpg; COCO train2017 for calib/calib_ids.json)"
IDS=os.environ.get("CALIB_IDS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib", "calib_ids.json")
onnx_path, tag = sys.argv[1], sys.argv[2]; size=int(sys.argv[3]) if len(sys.argv)>3 else 1008
m=onnx.load(onnx_path, load_external_data=False); g=m.graph; inits={t.name:t for t in g.initializer}; prod={o:n for n in g.node for o in n.output}
cons=collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
# structural sites -> map input tensor name -> (block, fam)
wm=[n for n in g.node if n.op_type=="MatMul" and n.input[1] in inits]; blk=-1; sites={}
for n in wm:
    K,N=list(inits[n.input[1]].dims)
    if (K,N)==(1024,4736): blk+=1; sites[n.input[0]]=(blk,"fc1")
    elif (K,N)==(4736,1024): sites[n.input[0]]=(blk,"fc2")
    elif (K,N)==(1024,1024):
        if sum(1 for c in cons[n.input[0]] if c.op_type=="MatMul")==3: sites[n.input[0]]=(blk+1,"qkv")
        elif (blk+1,"qkv") in sites.values(): sites[n.input[0]]=(blk+1,"proj")   # a block's output projection follows its q/k/v; earlier 1024x1024 products (the basis rotation) are not sites
NECK = len(sys.argv)>4 and sys.argv[4]=="neck"
if NECK:   # also record every Conv/ConvTranspose input except the patch-embedding conv on the image
    for n in g.node:
        if n.op_type in ("Conv","ConvTranspose") and n.input[0]!="pixel_values" and n.input[0]!="images" and n.input[0] not in sites: sites[n.input[0]]=("neck",n.input[0])
nblk=blk+1; print("blocks:",nblk,"site tensors:",len(sites)); sys.stdout.flush()
plan=f"calib_{tag}.plan"
logger=trt.Logger(trt.Logger.WARNING); b=trt.Builder(logger); net=b.create_network(0); p=trt.OnnxParser(net,logger); assert p.parse_from_file(onnx_path)
names={}
for i in range(net.num_layers):
    L=net.get_layer(i)
    for j in range(L.num_inputs):
        t=L.get_input(j)
        if t is not None and t.name in sites and t.name not in names:
            bi,fam=sites[t.name]; t.name=(f"calib.neck.{fam}" if bi=="neck" else f"calib.block{bi}.{fam}"); net.mark_output(t); names[t.name]=1
print("marked",len(names)); sys.stdout.flush()
cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8<<30); t=time.time(); ser=b.build_serialized_network(net,cfg); open(plan,"wb").write(memoryview(ser)); print("calib engine %.0fs"%(time.time()-t)); sys.stdout.flush()
run=Runner(load_engine(plan)); outs=[n for n in run.outputs if n.startswith("calib.")]
ids=json.load(open(IDS)); rng=np.random.default_rng(0)
if os.environ.get("CALIB_SUBSET"):   # "a" = first half, "b" = second half (noise-floor study)
    h=len(ids)//2; ids = ids[:h] if os.environ["CALIB_SUBSET"]=="a" else ids[h:]; print("calibration subset", os.environ["CALIB_SUBSET"], len(ids), "images")
st={s:{"absmax":0.0,"chmax":None,"samples":[]} for s in outs}
for k,iid in enumerate(ids):
    x,_=load_image_tensor(os.path.join(IMAGES, f"{iid}.jpg"), size); out=run({"images":x}, want=outs)
    for s in outs:
        a=np.abs(out[s].astype(np.float32)); flat=a.reshape(-1,a.shape[-1]); st[s]["absmax"]=max(st[s]["absmax"],float(flat.max())); cm=flat.max(axis=0); st[s]["chmax"]=cm if st[s]["chmax"] is None else np.maximum(st[s]["chmax"],cm); st[s]["samples"].append(flat.ravel()[rng.integers(0,flat.size,size=200000)])
    print(f"[{k+1}/{len(ids)}]"); sys.stdout.flush()
res={s:{"absmax":d["absmax"],"p99999":float(np.percentile(np.concatenate(d["samples"]),99.999)),"p9999":float(np.percentile(np.concatenate(d["samples"]),99.99)),"p999":float(np.percentile(np.concatenate(d["samples"]),99.9)),"max":d["absmax"]} for s,d in st.items()}
json.dump(res,open(f"calib_{tag}.json","w"),indent=1); np.savez_compressed(f"calib_{tag}.npz", **{s+"::chmax":d["chmax"] for s,d in st.items()}); os.remove(plan); print("saved calib_%s.json/npz"%tag)
