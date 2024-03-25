
eval_data_type="MSVD"
model_name="llava-internlm2-7b-chat-video-NEWVERSION-0324"
root_path="/cpfs01/user/fangxinyu/LMUData"
pred_path="${root_path}/${eval_data_type}-${model_name}.jsonl"
output_dir="${root_path}/video_zero_shot_result/${eval_data_type}-${model_name}"
output_json="${root_path}/video_zero_shot_result/final_result/${eval_data_type}-${model_name}_gpt35_turbo0125_result.json"
api_key="sk-UTzInM2T6ss8UML2E8874740Ae8e4874Ac416d8138379675"
api_base="https://api1.zhtec.xyz/v1"
num_tasks=32


python3 eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
