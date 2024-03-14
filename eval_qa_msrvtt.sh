


pred_path="/cpfs01/user/fangxinyu/LMUData/MSVD-internlm2-chat7b-video-0302.jsonl"
output_dir="/cpfs01/user/fangxinyu/VLMEvalKit/llava-internlm2-7b-chat-video-0302/msvd-gpt-video0302"
output_json="/cpfs01/user/fangxinyu/VLMEvalKit/llava-internlm2-7b-chat-video-0302/msvd_gpt35_turbo0125_result.json"
api_key="sk-UTzInM2T6ss8UML2E8874740Ae8e4874Ac416d8138379675"
api_base="https://uwjkzlwkokgy.cloud.sealos.io/v1"
num_tasks=32



python3 eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
