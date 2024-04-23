import os
import os.path as osp
import string
import sys
import warnings

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel,
                          GenerationConfig, StoppingCriteriaList)

from ..smp import cn_string, get_cache_path
from ..utils import DATASET_TYPE, CustomPrompt
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import decord
from decord import VideoReader, cpu
import torch.distributed as dist
decord.bridge.set_bridge('torch')

import argparse
import math
from tqdm import tqdm
import numpy as np
import cv2
import random

from transformers import PreTrainedModel
from typing import List, Optional
from mmengine.model import BaseModel
from .chatuniviModel import ChatUniViMetaForCausalLM

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_INDEX = 0
IMAGE_TOKEN_INDEX = -200
VIDEO_TOKEN_INDEX = -201
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_VIDEO_TOKEN = '<video>'

def get_video_transform(video_decode_backend, num_frames, image_size=224):

    if video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=image_size),
                    CenterCropVideo(image_size),
                    # RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=image_size),
                CenterCropVideo(image_size),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=image_size),
                CenterCropVideo(image_size),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform

def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):
    
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LLaVA_XTuner_VIDEO(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self,
                 llava_path,
                 num_frames=10,
                 image_size=336,
                 llm_path=None,
                 visual_encoder_path='openai/clip-vit-large-patch14-336',
                 visual_select_layer=-2,
                 prompt_template=None,
                 stop_words=[],
                 torch_dtype=torch.float16,
                 compress_video_tokens_with_chatunivi=False,
                 mode='pretrain'):
        try:
            from peft import PeftModel
            from xtuner.utils import PROMPT_TEMPLATE, StopWordStoppingCriteria
        except Exception:
            warnings.warn(
                'Please install xtuner with `pip install -U xtuner` before '
                'using LLaVA_XTuner')
            sys.exit(-1)

        if not osp.isdir(llava_path):
            cache_path = get_cache_path(llava_path)
            if cache_path is not None:
                llava_path = cache_path
            else:
                llava_path = snapshot_download(repo_id=llava_path)
        assert osp.exists(llava_path) and osp.isdir(llava_path)

        # build visual_encoder
        if 'llm' in os.listdir(llava_path):
            assert llm_path is None, (
                "Please don't specify the `llm_path` since passed "
                '`llava_path` contains a LLM!')
            llm_path = osp.join(llava_path, 'llm')
        else:
            assert llm_path is not None, 'Please specify the `llm_path`!'

        llm = AutoModelForCausalLM.from_pretrained(llm_path,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch_dtype,
                                                   device_map='cpu')
        tokenizer = AutoTokenizer.from_pretrained(llm_path,
                                                  trust_remote_code=True,
                                                  encode_special_tokens=True)
        print(f'Load LLM from {llm_path}')

        # build visual_encoder
        if 'visual_encoder' in os.listdir(llava_path):
            assert visual_encoder_path is None, (
                "Please don't specify the `visual_encoder_path` since passed "
                '`llava_path` contains a visual encoder!')
            visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            assert visual_encoder_path is not None, (
                'Please specify the `visual_encoder_path`!')
        # print(visual_encoder_path)
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path, torch_dtype=torch_dtype, device_map='cpu')
        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)
        print(f'Load visual_encoder from {visual_encoder_path}')

        # load adapter
        if 'llm_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'llm_adapter')
            llm = PeftModel.from_pretrained(llm,
                                            adapter_path,
                                            trust_remote_code=True,
                                            device_map='cpu')
            print(f'Load LLM adapter from {llava_path}')
        if 'visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
            visual_encoder = PeftModel.from_pretrained(visual_encoder,
                                                       adapter_path,
                                                       trust_remote_code=True,
                                                       device_map='cpu')
            print(f'Load visual_encoder adapter from {llava_path}')

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        projector = AutoModel.from_pretrained(projector_path,
                                              trust_remote_code=True,
                                              torch_dtype=torch_dtype,
                                              device_map='cpu')
        print(f'Load projector from {llava_path}')

        llm.eval()
        visual_encoder.eval()
        projector.eval()

        self.llm = llm.cuda()
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder.cuda()
        self.image_processor = image_processor
        self.projector = projector.cuda()
        self.visual_select_layer = visual_select_layer
        self.num_frames = num_frames
        self.image_size = image_size
        if prompt_template is not None:
            self.prompt_template = PROMPT_TEMPLATE[prompt_template]
            stop_words += self.prompt_template.get('STOP_WORDS', [])
        else:
            self.prompt_template = None

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

        self.model_args = {
            'pretrain': {
                
                "use_cluster": True,
                "freeze": False,
                "vision_tune": False,

                "spatial_cluster_rate0": 64,  # 0.25
                "spatial_cluster_rate1": 32,  # 0.5
                "spatial_cluster_rate2": 16,  # 0.5

                "temporal_cluster_rate": 1/16,
            },

            'finetune':{

                "use_cluster": True,
                "freeze": False,
                "mm_tune": True,
                "vision_tune": False,

                "spatial_cluster_rate0": 64,  # 0.25
                "spatial_cluster_rate1": 32,  # 0.5
                "spatial_cluster_rate2": 16,  # 0.5

                "temporal_cluster_rate": 1/16,

            }
        }

        self.config = {'mm_hidden_size': 1024}
        self.enable_compress_tokens = compress_video_tokens_with_chatunivi
        if compress_video_tokens_with_chatunivi:
            self.chat_univi_model = ChatUniViMetaForCausalLM(model_args=self.model_args[mode], config=self.config)
        else:    
            self.chat_univi_model = None


    def build_gen_config(self, dataset):
        gen_kwargs = dict(max_new_tokens=1024,
                          do_sample=False,
                          temperature=1,
                          num_beams=1,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id
                          if self.tokenizer.pad_token_id is not None else
                          self.tokenizer.eos_token_id)
        # For single word generation
        # if (dataset is not None
        #         and DATASET_TYPE(dataset) in ['multi-choice', 'Y/N']):
        #     gen_kwargs.update(
        #         dict(max_new_tokens=5, do_sample=False, num_beams=1))
        return GenerationConfig(**gen_kwargs)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line
                                and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'

        if not cn_string(question):
            prompt = question + '\n' + ("Answer with the option's letter "
                                        'from the given choices directly.')
        else:
            prompt = question + '\n' + '请直接回答选项字母。'

        return {'image': tgt_path, 'text': prompt}

    def generate(self, video_path, prompt, dataset=None):
        from xtuner.dataset.utils import expand2square
        from xtuner.model.utils import prepare_inputs_labels_for_multimodal
        # from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        try:
            video_decode_backend = 'decord'
            video = load_and_transform_video(video_path, get_video_transform(video_decode_backend=video_decode_backend,num_frames=self.num_frames,image_size=self.image_size),
                                                video_decode_backend=video_decode_backend,
                                                num_frames=self.num_frames)
        except Exception as e:
            try:
                import cv2
                import numpy as np
                import imageio
                if os.path.isdir(video_path):
                    fname, fename = os.path.split(video_path)
                    target_file_name = os.path.join(fname, f'{fename}.mp4')
                    filelist = os.listdir(video_path)
                    fps = 3
                    with imageio.get_writer(target_file_name, fps=fps) as video:
                        for file_name in sorted(filelist):
                            if file_name.endswith('.jpg'):
                                frame = np.array(Image.open(os.path.join(video_path,file_name)).convert('RGB'))
                                # frame = cv2.imread(os.path.join(img_root,file_name)).convert('RGB')
                                video.append_data(frame)


                else:
                    from moviepy.editor import VideoFileClip
                    print(video_path)
                    # fname, fename = os.path.split(video_path)
                    fname = os.path.dirname(video_path)
                    fename = os.path.basename(video_path)
                    fename_name = fename.split('.')[0]
                    target_file_name = os.path.join(fname, f'{fename_name}.mp4')
                    video = VideoFileClip(video_path)
                    video.write_videofile(target_file_name)

                video_decode_backend = 'decord'
                video = load_and_transform_video(target_file_name, get_video_transform(video_decode_backend=video_decode_backend,num_frames=self.num_frames,image_size=self.image_size),
                                            video_decode_backend=video_decode_backend,
                                            num_frames=self.num_frames)

                os.remove(target_file_name)


            except Exception as ne:
                print(f'Error: {ne}, {video_path}, load video failed!')
                raise
        video = video.permute(1,0,2,3).cuda()
        # print(video.shape, self.visual_encoder)
        visual_outputs = self.visual_encoder(video, output_hidden_states=True)
        
        if self.enable_compress_tokens:
            pixel_values = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        else:
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        inputs = DEFAULT_VIDEO_TOKEN + '\n' + prompt

        if self.prompt_template:
            inputs = self.prompt_template['INSTRUCTION'].format(input=inputs)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_VIDEO_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer(chunk)
            else:
                cur_encode = self.tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
            if idx != len(chunk_encode) - 1:
                ids.append(VIDEO_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        if self.enable_compress_tokens:
            mm_inputs = self.prepare_inputs_labels_for_multimodal_VIDEO(
                llm=self.llm, input_ids=ids, pixel_values=pixel_values, instance_list=['video'], num_frames = self.num_frames, chatuniviModel= self.chat_univi_model)
        else:
            try:
                mm_inputs = self.prepare_inputs_labels_for_multimodal_VIDEO(
                    llm=self.llm, input_ids=ids, pixel_values=pixel_values, instance_list=['video'], num_frames = self.num_frames)
            except Exception as e:
                # print(video_path, prompt, ids, e)
                raise

        gen_config = self.build_gen_config(dataset)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)
        predict = self.tokenizer.decode(generate_output[0],
                                        skip_special_tokens=True).strip()
        return predict
    

    def prepare_inputs_labels_for_multimodal_VIDEO(
        self,
        llm: PreTrainedModel,
        instance_list: List[str] = ['image'],
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        num_frames: Optional[int] = 10,
        chatuniviModel: BaseModel = None):
        
        if pixel_values is None:
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'inputs_embeds': None,
                'labels': labels
            }

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_inputs_embeds = []
        new_labels = []
        cur_image_idx = 0

        split_sizes_overall = []
        vision_feature_overall = []
        overall_feat_after_proj_split_list = []
        if chatuniviModel is not None:
            for batch_idx, (cur_input_ids, instance) in enumerate(zip(input_ids, instance_list)):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
                split_sizes_perinstance = []
                vision_feat_perinstance = []
                if num_videos > 0:
                    for i in range(num_videos + 1):
                        if i < num_videos:
                            cur_pixel_values = pixel_values[cur_image_idx: cur_image_idx + num_frames]
                            cur_image_idx += num_frames
                            cur_image_features = chatuniviModel(cur_pixel_values, input_type="video").squeeze(0)
                            # cur_image_features = cur_image_features.to(cur_inputs_embeds.dtype)
                            # cur_image_features = cur_image_features.to(cur_inputs_embeds.device)
                            split_sizes_perinstance.append(cur_image_features.shape[0])
                            vision_feat_perinstance.append(cur_image_features)

                if num_images > 0:
                    for i in range(num_images + 1):
                        if i < num_images:
                            cur_pixel_values = pixel_values[cur_image_idx]
                            cur_image_idx += 1
                            cur_image_features = chatuniviModel(cur_pixel_values.unsqueeze(0), input_type="image").squeeze(0)
                            # cur_image_features = cur_image_features.to(cur_inputs_embeds.dtype)
                            # cur_image_features = cur_image_features.to(cur_inputs_embeds.device)
                            split_sizes_perinstance.append(cur_image_features.shape[0])
                            vision_feat_perinstance.append(cur_image_features)
                            # print('cur_image_features.shape:',cur_image_features.shape)

                if num_images == 0 and num_videos == 0:
                    cur_pixel_values = pixel_values[cur_image_idx]
                    cur_image_idx += 1
                    # print(f'empty image:{num_images} video:{num_videos} instance {instance} cur_input_ids {cur_input_ids} pixel')
                    ZERO_VISION_FEAT = torch.zeros(96, 1024).to(self.visual_encoder.device).to(self.visual_encoder.dtype)
                    split_sizes_perinstance.append(ZERO_VISION_FEAT.shape[0])
                    vision_feat_perinstance.append(ZERO_VISION_FEAT)
                               
                split_sizes_overall.append(split_sizes_perinstance)
                vision_feature_overall.append(torch.cat(vision_feat_perinstance))

            overall_feat_before_proj = torch.cat(vision_feature_overall)
            overall_feat_after_proj = self.projector(overall_feat_before_proj)
            overall_feat_after_proj_split = torch.split(overall_feat_after_proj, [lis for lists in split_sizes_overall for lis in lists], dim=0)
            i = 0
            
            for split_sizes_perinstance in split_sizes_overall:
                overall_feat_after_proj_split_list.append(overall_feat_after_proj_split[i:i+len(split_sizes_perinstance)])
                i = i + len(split_sizes_perinstance)

        cur_image_idx = 0
        if len(overall_feat_after_proj_split_list) == 0:
            overall_feat_after_proj_split_list.append(torch.zeros(96, 1024).to(self.visual_encoder.device).to(self.visual_encoder.dtype))
        for batch_idx, (cur_input_ids, instance, vision_feat_perinstance) in enumerate(zip(input_ids, instance_list, overall_feat_after_proj_split_list)):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()

            #num_videos = # (cur_input_ids == VIDEO_TOKEN_INDEX).sum() 
            # print(f'image num {num_images}, video num {num_videos}')
            if num_images == 0 and num_videos == 0:
                cur_pixel_values = pixel_values[cur_image_idx]
                cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
                cur_inputs_embeds = torch.cat(
                    [cur_inputs_embeds_1], dim=0)
                new_inputs_embeds.append(cur_inputs_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            
            if num_videos > 0:
                video_token_indices =  [-1] + torch.where(       #[-1, 4, cur_input_ids.shape[0]]
                    cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist() + [
                        cur_input_ids.shape[0]
                    ]
                
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(video_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[video_token_indices[i] +
                                                            1:video_token_indices[i +
                                                                                1]])
                    cur_labels_noim.append(cur_labels[video_token_indices[i] +
                                                    1:video_token_indices[i + 1]])
                # print(cur_input_ids_noim, cur_labels_noim)
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_inputs_embeds = llm.get_input_embeddings()(
                    torch.cat(cur_input_ids_noim))
                cur_inputs_embeds_no_im = torch.split(
                    cur_inputs_embeds, split_sizes, dim=0)
                # print(cur_inputs_embeds.shape, cur_inputs_embeds_no_im)
                cur_new_inputs_embeds = []
                cur_new_labels = []

                for i in range(num_videos + 1):
                    cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_videos:
                        cur_pixel_values = pixel_values[cur_image_idx: cur_image_idx + num_frames]
                        cur_image_idx += num_frames
                        if chatuniviModel is None:
                            for item in cur_pixel_values:
                                #VIDEO Squeeze to IMAGE
                                cur_new_inputs_embeds.append(item) #check here
                                cur_new_labels.append(
                                    torch.full((item.shape[0], ),
                                            IGNORE_INDEX,
                                            device=cur_labels.device,
                                            dtype=cur_labels.dtype))

                        else:
                            cur_new_inputs_embeds.append(vision_feat_perinstance[i])
                            cur_new_labels.append(
                                    torch.full((vision_feat_perinstance[i].shape[0], ),#report error here
                                            IGNORE_INDEX,
                                            device=cur_labels.device,
                                            dtype=cur_labels.dtype))

            
            if num_images > 0:
                image_token_indices = [-1] + torch.where(
                    cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                        cur_input_ids.shape[0]
                    ]
                # print(f'image token indices: {image_token_indices}, {cur_input_ids.shape[0]}')
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                            1:image_token_indices[i +
                                                                                1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i] +
                                                    1:image_token_indices[i + 1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_inputs_embeds = llm.get_input_embeddings()(
                    torch.cat(cur_input_ids_noim))
                cur_inputs_embeds_no_im = torch.split(
                    cur_inputs_embeds, split_sizes, dim=0)
                cur_new_inputs_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_pixel_values = pixel_values[cur_image_idx]
                        cur_image_idx += 1
                        if chatuniviModel is None:
                            cur_new_inputs_embeds.append(cur_pixel_values)
                            cur_new_labels.append(
                                torch.full((cur_pixel_values.shape[0], ),
                                        IGNORE_INDEX,
                                        device=cur_labels.device,
                                        dtype=cur_labels.dtype))
                        else:
                            # cur_image_features = chatuniviModel(cur_pixel_values.unsqueeze(0), input_type="image").squeeze(0)
                            # cur_image_features = cur_image_features.to(cur_inputs_embeds.dtype)
                            # cur_image_features = cur_image_features.to(cur_inputs_embeds.device)
                            cur_new_inputs_embeds.append(vision_feat_perinstance[i])
                            cur_new_labels.append(
                                torch.full((vision_feat_perinstance[i].shape[0], ),
                                        IGNORE_INDEX,
                                        device=cur_labels.device,
                                        dtype=cur_labels.dtype))

            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_inputs_embeds.append(cur_new_inputs_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_inputs_embeds)
        batch_size = len(new_inputs_embeds)

        new_inputs_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len),
                                    IGNORE_INDEX,
                                    dtype=new_labels[0].dtype,
                                    device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len),
                                dtype=position_ids.dtype,
                                device=position_ids.device)

        for i, (cur_new_embed,
                cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_inputs_embeds_padded.append(
                torch.cat((cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                    dtype=cur_new_embed.dtype,
                                    device=cur_new_embed.device)),
                        dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0,
                    cur_len,
                    dtype=position_ids.dtype,
                    device=position_ids.device)

        new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)
        # print(new_inputs_embeds.shape)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        # try:
        #     print(new_inputs_embeds.shape, new_labels.shape)
        # except:
        #     print('get failure')
        return {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_inputs_embeds,
            'labels': new_labels
        }
    


