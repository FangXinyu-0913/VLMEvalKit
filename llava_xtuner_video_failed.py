import os
import os.path as osp
import string
import sys
import warnings

import pandas as pd
import torch
from huggingface_hub import snapshot_download, scan_cache_dir
from PIL import Image
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel,
                          GenerationConfig, StoppingCriteriaList)

# from ..smp import cn_string, get_cache_path
from vlmeval.utils import DATASET_TYPE, CustomPrompt
import argparse
import math
from tqdm import tqdm
import numpy as np
import cv2
import random

from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import decord
from decord import VideoReader, cpu
import torch.distributed as dist
decord.bridge.set_bridge('torch')

def get_rank_and_world_size():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, world_size

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_cache_path(repo_id):
    hf_cache_info = scan_cache_dir()
    repos = list(hf_cache_info.repos)
    repo = None
    for r in repos:
        if r.repo_id == repo_id:
            repo = r
            break
    if repo is None:
        return None
    revs = list(repo.revisions)
    rev2keep, last_modified = None, 0
    for rev in revs:
        if rev.last_modified > last_modified:
            rev2keep, last_modified = rev, rev.last_modified 
    if rev2keep is None:
        return None
    return str(rev2keep.snapshot_path)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
def get_video_transform(video_decode_backend, num_frames):

    if video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=336),
                CenterCropVideo(336),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
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

class LLaVA_XTuner_video(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self,
                 llava_path,
                 llm_path=None,
                 visual_encoder_path='openai/clip-vit-large-patch14-336',
                 visual_select_layer=-2,
                 prompt_template=None,
                 stop_words=[],
                 torch_dtype=torch.float16):
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
        if prompt_template is not None:
            self.prompt_template = PROMPT_TEMPLATE[prompt_template]
            stop_words += self.prompt_template.get('STOP_WORDS', [])
        else:
            self.prompt_template = None

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

    def build_gen_config(self, dataset):
        gen_kwargs = dict(max_new_tokens=1024,
                          do_sample=True,
                          temperature=1,
                          num_beams=5,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id
                          if self.tokenizer.pad_token_id is not None else
                          self.tokenizer.eos_token_id)
        # For single word generation
        if (dataset is not None
                and 'VQA' in ['multi-choice', 'Y/N']):
            gen_kwargs.update(
                dict(max_new_tokens=5, do_sample=False, num_beams=1))
        return GenerationConfig(**gen_kwargs)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if 'VQA' == 'multi-choice':
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
        from xtuner.utils import DEFAULT_VIDEO_TOKEN, VIDEO_TOKEN_INDEX
        video_decode_backend = 'decord'
        num_frames = 10 #TODO
        print("why here!?")
        video = load_and_transform_video(video_path, get_video_transform(video_decode_backend=video_decode_backend, num_frames=num_frames),
                                            video_decode_backend=video_decode_backend,
                                            num_frames=num_frames) 
        # image = Image.open(image_path).convert('RGB')
        # image = expand2square(
        #     image,
        #     tuple(int(x * 255) for x in self.image_processor.image_mean))
        # image = self.image_processor.preprocess(
        #     image, return_tensors='pt')['pixel_values'][0]
        video = video.permute(1,0,2,3).cuda()
        visual_outputs = self.visual_encoder(video, output_hidden_states=True)
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
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

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
    
import json
if __name__ == '__main__':
    video_model = LLaVA_XTuner_video(llm_path='/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f', llava_path='/cpfs01/user/fangxinyu/xtuner/0302-qlora-result', visual_select_layer=-2, prompt_template='internlm2_chat')
    
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    args = parser.parse_args()

    # rank, world_size = get_rank_and_world_size()
    # import datetime
    # if world_size > 1:
    #     torch.cuda.set_device(rank)
    #     dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    
    gt_questions = json.load(open(args.gt_file_question, "r"))
    # gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))


    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    print(f'rank: {rank}, world_size: {world_size}')

    indices = list(range(rank, len(gt_questions), world_size))
    lt = len(indices)
    gt_questions = gt_questions[indices]

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        for fmt in tqdm(video_formats):  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = video_model.generate(video_path, question, args)
                sample_set['pred'] = output
                # print(sample_set)
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()