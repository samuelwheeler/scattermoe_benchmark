#!/usr/bin/env bash
set -euo pipefail


# module load thapi
# module load frameworks
python3 -m pip install viztracer
export PATH=/home/sww/.local/aurora/frameworks/2025.2.0/bin:$PATH

# ---- Configuration grids ----
INPUT_SIZES=(1024 2048)
HIDDEN_SIZES=(1024 2048 4096 8192)
NUM_EXPERTS=(8 16 32 64)
TOP_K=(1 2 8)
NUM_TOKENS=(512 1024 4096)

# ---- Paths ----
SCRIPT="./new_benchmark.py"          
OUT_DIR="cleanup_test"
AGG_JSON="$OUT_DIR/thapi_all_results_with_ipex.json"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$OUT_DIR/pf_traces" "$OUT_DIR/viz_traces"

total=0
for input in "${INPUT_SIZES[@]}"; do
  for hidden in "${HIDDEN_SIZES[@]}"; do
    for experts in "${NUM_EXPERTS[@]}"; do
      for k in "${TOP_K[@]}"; do
        (( k > experts )) && continue
        for tokens in "${NUM_TOKENS[@]}"; do
          id="i${input}_h${hidden}_e${experts}_k${k}_t${tokens}"
          echo "Running $id"
          PFTRACE_NAME="$OUT_DIR/pf_traces/${id}.pftrace"
          VIZTRACER_OUT_NAME="$OUT_DIR/viz_traces/${id}.json"
        #   iprof sycl-ls
        if iprof -l $PFTRACE_NAME -- python3 -- "$SCRIPT" \
              --input_size "$input" \
              --hidden_size "$hidden" \
              --num_experts "$experts" \
              --top_k "$k" \
              --num_tokens "$tokens" \
              --output "$AGG_JSON" \
              --viztracer_outfile "$VIZTRACER_OUT_NAME" \
              >"$LOG_DIR/${id}.out" 2>"$LOG_DIR/${id}.err" ; then
            :
        else
            echo "ERROR: $id failed (see $LOG_DIR/${id}.err)"
        fi
          total=$((total + 1))
        done
      done
    done
  done
done

echo "Sweep complete. Attempted $total configurations."
echo "Results aggregated in: $AGG_JSON"
echo "Logs in: $LOG_DIR/"
