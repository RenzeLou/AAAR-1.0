gpu=$1  # e.g., 4,5,6
model=$2
max_context_len=$3  # 1000
max_model_len=$4
gpu_num=$(echo $gpu | tr -cd ',' | wc -c)
gpu_num=$((gpu_num + 1))

set -x
echo "export CUDA_VISIBLE_DEVICES=$gpu"
echo "Number of GPUs: $gpu_num"

export CUDA_VISIBLE_DEVICES=${gpu}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export HF_HOME=/data/rml6079/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python scripts/subtask1_equation_model_eval.open_source.py \
    --api_name ${model} \
    --template 1 \
    --root_dir './Equation_Inference' \
    --eval_data_file 'equation.1049.json' \
    --save_dir './Equation_Inference/eval_results' \
    --context_max_len ${max_context_len} \
    --temperature 0.8 \
    --top_p 0.95 \
    --seed 42 \
    --gpu_num ${gpu_num} \
    --max_model_len ${max_model_len} 