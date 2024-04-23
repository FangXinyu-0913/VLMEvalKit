from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

ungrouped = {
    'TransCore_M': partial(TransCoreM, root=TransCore_ROOT),
    'PandaGPT_13B': partial(PandaGPT, name='PandaGPT_13B', root=PandaGPT_ROOT),
    'flamingov2': partial(OpenFlamingo, name='v2', mpt_pth='anas-awadalla/mpt-7b', ckpt_pth='openflamingo/OpenFlamingo-9B-vitl-mpt7b'),
    'VisualGLM_6b': partial(VisualGLM, model_path='THUDM/visualglm-6b'),
    'mPLUG-Owl2': partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b'),
    'cogvlm-grounding-generalist':partial(CogVlm, name='cogvlm-grounding-generalist',tokenizer_name ='lmsys/vicuna-7b-v1.5'),
    'cogvlm-chat':partial(CogVlm, name='cogvlm-chat',tokenizer_name ='lmsys/vicuna-7b-v1.5'),
    'emu2_chat':partial(Emu, model_path='BAAI/Emu2-Chat',),
    'MMAlaya':partial(MMAlaya, model_path='DataCanvas/MMAlaya'),
    'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'OmniLMM_12B':partial(OmniLMM12B, model_path='openbmb/OmniLMM-12B', root=OmniLMM_ROOT),
}

api_models = {
    'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high', retry=10),
    'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high', retry=10),
    'GeminiProVision': partial(GeminiProVision, temperature=0, retry=10),
    'QwenVLPlus': partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),
    'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
    # Internal Only
    'GPT4V_INT': partial(GPT4V_Internal, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'Step1V': partial(Step1V, temperature=0, retry=10),
    'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10),
    'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10),
    'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10),
}

xtuner_series = {
    'llava-internlm2-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-7b', llava_path='xtuner/llava-internlm2-7b', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-internlm2-20b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-20b', llava_path='xtuner/llava-internlm2-20b', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-internlm-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm-chat-7b', llava_path='xtuner/llava-internlm-7b', visual_select_layer=-2, prompt_template='internlm_chat'),
    'llava-v1.5-7b-xtuner': partial(LLaVA_XTuner, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/shared/llmeval/dhd/hub/models--xtuner--llava-v1.5-7b-xtuner/snapshots/78784f77e0de4d5f401be5621c39414fc3b05f91', visual_select_layer=-2, prompt_template='vicuna'),
    'llava-v1.5-13b-xtuner': partial(LLaVA_XTuner, llm_path='lmsys/vicuna-13b-v1.5', llava_path='xtuner/llava-v1.5-13b-xtuner', visual_select_layer=-2, prompt_template='vicuna'),
    'llava-internlm2-7b-chat-video-0302': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner/0302-qlora-result', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-vicuna-1.5-7b-clip-vit-large-p14-336-0322': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_video_10frame-0322',visual_select_layer=-2, prompt_template='vicuna',num_frames=10,visual_encoder_path='/cpfs01/shared/llmeval/dhd/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'),
    'llava-vicuna-1.5-7b-laion-clip-vit-large-p14-224-0324': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_vicuna_7b_v15_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_10frame-0324',visual_select_layer=-2, prompt_template='vicuna',num_frames=10,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
    'llava-internlm2-7b-chat-video-NEWVERSION-0324': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_video_10frame_maxlen2048-0324',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=10,image_size=336,visual_encoder_path='/cpfs01/shared/llmeval/dhd/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'),
    'llava-vicuna-1.5-7b-laion-clip-vit-large-p14-224-16frame-0327': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_vicuna_7b_v15_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_16frame-0327',visual_select_layer=-2, prompt_template='vicuna',num_frames=16,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
    'llava-vicuna-1.5-7b-laion-clip-vit-large-p14-224-8frame-0328': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_vicuna_7b_v15_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_8frame-0328',visual_select_layer=-2, prompt_template='vicuna',num_frames=8,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'), 
    'llava-internlm2-7b-laion-clip-vit-large-p14-224-16frame-0403': partial(LLaVA_XTuner_VIDEO,llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_16f_maxlen12288-0403',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=16,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
    'llava-internlm2-7b-clip-vit-large-p14-336-16frame-0409': partial(LLaVA_XTuner_VIDEO,llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_video_16frame_maxlen12288-0409',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=16,image_size=336),
    'llava-internlm2-7b-laion-clip-vit-large-p14-224-16frame-0411-NEW': partial(LLaVA_XTuner_VIDEO,llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_16f_maxlen2048-0411',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=16,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
    'llava-internlm2-7b-laion-clip-vit-large-p14-224-128frame-TESTUNIVI':partial(LLaVA_XTuner_VIDEO,llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_128f1536l_testunivi-0412',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=128,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K',compress_video_tokens_with_chatunivi=True,mode='finetune'),
    'llava-internlm2-7b-laion-clip-vit-large-p14-224-128frame-TESTUNIVI-0416':partial(LLaVA_XTuner_VIDEO,llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_128f1792L_testunivi-0416',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=128,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K',compress_video_tokens_with_chatunivi=True,mode='finetune'),
    'llava-vicuna-1.5-7b-laion-clip-vit-large-p14-224-16frame-FULL-0420': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_vicuna_7b_v15_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_16frame-0420',visual_select_layer=-2, prompt_template='vicuna',num_frames=16,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
    'llava-vicuna-1.5-7b-laion-clip-vit-large-p14-224-16frame-DIRESFT-0420': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_vicuna_7b_v15_qlora_laion_clip_vit_large_p14_224_lora_e1_gpu8_finetune_video_16frame_diresft-0420',visual_select_layer=-2, prompt_template='vicuna',num_frames=16,image_size=224,visual_encoder_path='/cpfs01/shared/llmeval/fangxinyu/hub/models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
    'llava-internlm2-7b-clip-vit-large-p14-336-8frame-0421':partial(LLaVA_XTuner_VIDEO,llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_video_8f1472l-0421',visual_select_layer=-2, prompt_template='internlm2_chat',num_frames=8,image_size=336),
    # 'llava-internlm2-7b-chat-video-0322': partial(LLaVA_XTuner_VIDEO, llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner_result/temp', visual_select_layer=-2, prompt_template='internlm2_chat')
}

qwen_series = {
    'qwen_base': partial(QwenVL, model_path='Qwen/Qwen-VL'),
    'qwen_chat': partial(QwenVL, model_path='Qwen/Qwen-VL-Chat'),
    'monkey':partial(Monkey, model_path='echo840/Monkey'),
    'monkey-chat':partial(MonkeyChat, model_path='echo840/Monkey-Chat')
}

llava_series = {
    'llava_v1.5_7b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-13b'),
    'llava_v1_7b': partial(LLaVA, model_pth=LLAVA_V1_7B_MODEL_PTH),
    'sharegpt4v_7b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-7B'),
    'sharegpt4v_13b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-13B'),
    'llava_next_vicuna_7b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-vicuna-7b-hf'),
    'llava_next_vicuna_13b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-vicuna-13b-hf'),
    'llava_next_mistral_7b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-mistral-7b-hf'),
    'llava_next_yi_34b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-34b-hf'),
}

internvl_series = {
    'InternVL-Chat-V1-1':partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-Chinese-V1-1'),
    'InternVL-Chat-V1-2': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-Chinese-V1-2'),
    'InternVL-Chat-V1-2-Plus': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus'),
    'InternVL-Chat-V1-5': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-5-Preview'),
}

yivl_series = {
    'Yi_VL_6B':partial(Yi_VL, model_path='01-ai/Yi-VL-6B', root=Yi_ROOT),
    'Yi_VL_34B':partial(Yi_VL, model_path='01-ai/Yi-VL-34B', root=Yi_ROOT),
}

xcomposer_series = {
    'XComposer': partial(XComposer, model_path='internlm/internlm-xcomposer-vl-7b'),
    'sharecaptioner': partial(ShareCaptioner, model_path='Lin-Chen/ShareCaptioner'),
    'XComposer2': partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-7b'),
    'XComposer2_1.8b': partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-1_8b'),
    'XComposer2_4KHD': partial(XComposer2_4KHD, model_path='internlm/internlm-xcomposer2-4khd-7b'),
}

minigpt4_series = {
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b', root=MiniGPT4_ROOT),
}

idefics_series = {
    'idefics_9b_instruct': partial(IDEFICS, model_pth='HuggingFaceM4/idefics-9b-instruct'),
    'idefics_80b_instruct': partial(IDEFICS, model_pth='HuggingFaceM4/idefics-80b-instruct'),
    'idefics2_8b': partial(IDEFICS2, model_path='HuggingFaceM4/idefics2-8b'),
}

instructblip_series = {
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
}

deepseekvl_series = {
    'deepseek_vl_7b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-7b-chat'),
    'deepseek_vl_1.3b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-1.3b-chat'),
}

supported_VLM = {}

model_groups = [
    ungrouped, api_models, 
    xtuner_series, qwen_series, llava_series, internvl_series, yivl_series,
    xcomposer_series, minigpt4_series, idefics_series, instructblip_series,
    deepseekvl_series
]

for grp in model_groups:
    supported_VLM.update(grp)

transformer_ver = {}
transformer_ver['4.33.0'] = list(qwen_series) + list(internvl_series) + list(xcomposer_series) + [
    'mPLUG-Owl2', 'flamingov2', 'VisualGLM_6b', 'MMAlaya', 'PandaGPT_13B'
] + list(idefics_series) + list(minigpt4_series) + list(instructblip_series)
transformer_ver['4.37.0'] = [x for x in llava_series if 'next' not in x] + [
    'TransCore_M', 'cogvlm-chat', 'cogvlm-grounding-generalist', 'emu2_chat', 'MiniCPM-V', 'MiniCPM-V-2', 'OmniLMM_12B'
] + list(xtuner_series) + list(yivl_series) + list(deepseekvl_series)
transformer_ver['4.39.0'] = [x for x in llava_series if 'next' in x]

if __name__ == '__main__':
    import sys
    ver = sys.argv[1]
    if ver in transformer_ver:
        print(' '.join(transformer_ver[ver]))