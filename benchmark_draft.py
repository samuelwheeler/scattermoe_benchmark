#!/usr/bin/env python3
import argparse, json, time, os, torch
#import intel_extension_for_pytorch as ipex
from scattermoe.mlp import MLP
import scattermoe.kernels.ops

scattermoe.kernels.ops.ALLOW_TF32 = False
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False

parser = argparse.ArgumentParser("Single-config, 16-step fwd+bwd timing")
parser.add_argument("--input_size", type=int, default=1024)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--num_experts", type=int, default=16)
parser.add_argument("--top_k", type=int, default=8)
parser.add_argument("--num_tokens", type=int, default=1024)
parser.add_argument("--output", type=str, required=False)
parser.add_argument("--warmup_steps", type=int, default=4)
parser.add_argument("--sweep", action='store_true',)
args = parser.parse_args()


# Device selection

def device_stuff():
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

device = torch.accelerator.current_accelerator()
reset_peak_mem = torch.accelerator.reset_peak_memory_stats
max_mem = torch.accelerator.max_memory_allocated
sync = torch.accelerator.synchronize
name = "XXXX" #torch.accelerator.get_device_properties(0).name

if args.top_k > args.num_experts:
    print(json.dumps({"error": "InvalidConfig", "details": "top_k > num_experts"}))
    raise SystemExit(1)


torch.manual_seed(1234)
dtype = torch.bfloat16
steps = 16

def one_config():
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
    
    
    if True:
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
    

# ---- Sweep Properties ----
INPUT_SIZES=(1024, 2048)
HIDDEN_SIZES=(1024, 2048, 4096, 8192)
NUM_EXPERTS=(8, 16, 32, 64)
TOP_K=(1, 2, 8)
NUM_TOKENS=(512, 1024, 4096)

def sweep_parameters():
    """Sweep through all parameter combinations and run benchmarks"""
    results = []
    
    for input_size in INPUT_SIZES:
        for hidden_size in HIDDEN_SIZES:
            for num_experts in NUM_EXPERTS:
                for top_k in TOP_K:
                    for num_tokens in NUM_TOKENS:
                        # Skip invalid configurations
                        if top_k > num_experts:
                            continue
                        
                        # Update args for this configuration
                        args.input_size = input_size
                        args.hidden_size = hidden_size
                        args.num_experts = num_experts
                        args.top_k = top_k
                        args.num_tokens = num_tokens
                        
                        print(f"Running config: input_size={input_size}, hidden_size={hidden_size}, "
                              f"num_experts={num_experts}, top_k={top_k}, num_tokens={num_tokens}")
                        
                        try:
                            result = one_config()
                            if result:
                                result.update({
                                    'input_size': input_size,
                                    'hidden_size': hidden_size,
                                    'num_experts': num_experts,
                                    'top_k': top_k,
                                    'num_tokens': num_tokens
                                })
                                results.append(result)
                        except Exception as e:
                            print(f"Error in config: {e}")
                            results.append({
                                'input_size': input_size,
                                'hidden_size': hidden_size,
                                'num_experts': num_experts,
                                'top_k': top_k,
                                'num_tokens': num_tokens,
                                'error': str(e)
                            })
    
    return results

# Check if we should run sweep or single config
if not args.sweep: 
    # Single config mode - run as before
    print(f"Running config: input_size={args.input_size}, hidden_size={args.hidden_size}, "
          f"num_experts={args.num_experts}, top_k={args.top_k}, num_tokens={args.num_tokens}")
    one_config()
else:
    # Sweep mode - run all combinations
    sweep_results = sweep_parameters()
    
    # Save results to output file
    with open(args.output, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    print(f"Sweep completed. Results saved to {args.output}")

