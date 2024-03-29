
eval_data_type="MSRVTT"
minor_data_type="msrvtt"
model_name="llava_vicuna_7b_v15_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_8frame"
root_path="/cpfs01/user/fangxinyu/LMUData"
# pred_path="${root_path}/${eval_data_type}-${model_name}.jsonl"
pred_path="/cpfs01/user/fangxinyu/xtuner/work_dirs/${model_name}/output_${eval_data_type}/${minor_data_type}_qa.json"
output_dir="${root_path}/video_zero_shot_result/${eval_data_type}-${model_name}"
output_json="${root_path}/video_zero_shot_result/final_result/${eval_data_type}-${model_name}_gpt35_turbo0125_result.json"
api_key="sk-UTzInM2T6ss8UML2E8874740Ae8e4874Ac416d8138379675"
api_base="https://api1.zhtec.xyz/v1"
num_tasks=32

echo "Start evaluating ${eval_data_type} with ${model_name}..."

python3 eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
