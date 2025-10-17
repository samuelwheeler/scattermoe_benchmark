#!/usr/bin/env python3
import argparse, json, time, os, torch
import intel_extension_for_pytorch as ipex
from scattermoe.mlp import MLP
import scattermoe.kernels.ops

scattermoe.kernels.ops.ALLOW_TF32 = False
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False

parser = argparse.ArgumentParser("Single-config, 16-step fwd+bwd timing")
parser.add_argument("--input_size", type=int, required=True)
parser.add_argument("--hidden_size", type=int, required=True)
parser.add_argument("--num_experts", type=int, required=True)
parser.add_argument("--top_k", type=int, required=True)
parser.add_argument("--num_tokens", type=int, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--warmup_steps", type=int, default=4)
parser.add_argument("--viztracer_outfile", type = str)
args = parser.parse_args()

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    reset_peak_mem = torch.cuda.reset_peak_memory_stats
    max_mem = torch.cuda.max_memory_allocated
    sync = torch.cuda.synchronize
    name = torch.cuda.get_device_properties(0).name
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
    reset_peak_mem = torch.xpu.reset_peak_memory_stats
    max_mem = torch.xpu.max_memory_allocated
    sync = torch.xpu.synchronize
    try:
        name = torch.xpu.get_device_properties(0).name
    except Exception:
        name = "Intel XPU"
else:
    raise NotImplementedError

if args.top_k > args.num_experts:
    print(json.dumps({"error": "InvalidConfig", "details": "top_k > num_experts"}))
    raise SystemExit(1)

from viztracer import VizTracer

torch.manual_seed(1234)
dtype = torch.bfloat16
steps = 16
tokens = args.num_tokens

model = (MLP(
    input_size=args.input_size,
    hidden_size=args.hidden_size,
    num_experts=args.num_experts,
    top_k=args.top_k,
    bias=False,
    activation=torch.nn.GELU(),
).to(device).to(dtype).train())

with torch.no_grad():
    x = torch.randn(tokens, args.input_size, dtype=dtype, device=device)
    logits = torch.randn(tokens, args.num_experts, dtype=dtype, device=device)
    weights = torch.softmax(logits.float(), dim=-1).to(dtype)
    k_w, k_idx = torch.topk(weights, args.top_k)
    k_w = k_w.contiguous()
    k_idx = k_idx.to(torch.int32).contiguous()

x.requires_grad_(True)
k_w.requires_grad_(True)
grad_out = torch.randn_like(x, dtype=dtype, device=device)

# Warmup
for _ in range(args.warmup_steps):
    y = model(x, k_w, k_idx)
    loss = (y * grad_out).sum()
    loss.backward()
    model.zero_grad(set_to_none=True)
sync()


with VizTracer(output_file=args.viztracer_outfile) as tracer:
    reset_peak_mem()
    sync()
    t0 = time.perf_counter_ns()
    for _ in range(steps):
        y = model(x, k_w, k_idx)
        loss = (y * grad_out).sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
    sync()
    t1 = time.perf_counter_ns()


elapsed_ns = t1 - t0
elapsed_s = elapsed_ns / 1e9
peak_mb = max_mem() / (1024**2)


result = {
    "device": {"name": name, "type": device.type},
    "config": vars(args),
    "metrics": {
        "elapsed_ns": elapsed_ns,
        "elapsed_s": elapsed_s,
        "avg_step_ns": elapsed_ns / steps,
        "avg_step_s": elapsed_s / steps,
        "peak_memory_mb": peak_mb,
        "steps": steps
    },
}

import io
import tempfile, shutil

def append_json_array(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cur = json.load(f)
                data = cur if isinstance(cur, list) else [cur]
        except json.JSONDecodeError:
            data = []
    data.append(obj)

    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        tmpname = tf.name
    os.replace(tmpname, path)

append_json_array(args.output, result)

print(f"{device.type} {name} | {elapsed_s:.4f}s total for 16 steps "
      f"(avg {elapsed_s/steps:.6f}s), peak {peak_mb:.1f} MB")
