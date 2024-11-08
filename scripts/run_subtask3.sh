gpu=$1  # e.g., 4,5,6
model=$2
max_word_len=$3
max_model_len=$4
# pick_num=$5  # 1000
split=$5

gpu_num=$(echo $gpu | tr -cd ',' | wc -c)
gpu_num=$((gpu_num + 1))

set -x
echo "export CUDA_VISIBLE_DEVICES=$gpu"
echo "Number of GPUs: $gpu_num"

export CUDA_VISIBLE_DEVICES=${gpu}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/data/rml6079/.cache/huggingface
export HF_HOME=/data/rml6079/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# if split is 0, then there is no '--split' argument, else there is
if [ $split -eq 0 ]; then
    python scripts/subtask3_review_model_prediction.open_source.py \
        --api_name ${model} \
        --max_word_len ${max_word_len} \
        --temperature 0.8 \
        --top_p 0.95 \
        --seed 42 \
        --gpu_num ${gpu_num} \
        --max_model_len ${max_model_len} \
        --root_dir './Paper_Weakness' \
        --save_dir './Paper_Weakness/eval_results'
else
    python scripts/subtask3_review_model_prediction.open_source.py \
        --api_name ${model} \
        --max_word_len ${max_word_len} \
        --temperature 0.8 \
        --top_p 0.95 \
        --seed 42 \
        --gpu_num ${gpu_num} \
        --max_model_len ${max_model_len} \
        --root_dir './Paper_Weakness' \
        --save_dir './Paper_Weakness/eval_results' \
        --split
fi
    
