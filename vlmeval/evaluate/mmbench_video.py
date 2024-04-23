from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal
from functools import partial
from vlmeval.utils import track_progress_rich, can_infer_option
import argparse
from tqdm import tqdm
from vlmeval.smp import load, dump
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
from collections import OrderedDict, defaultdict
import pandas as pd


IMAGE9_PROMPT = (
    "The image provided is a compilation of 9 separate frames taken from a video. "
    "These frames were selected at regular intervals throughout the video's duration. "
    "They are arranged in a sequence from left to right, and top to bottom, reflecting the chronological order of the video. "
    "Please analyze this image and provide an answer to the following question about the video content.\n\n"
)

system_prompt = (
    "As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer. "
    "The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you. "
    "Your assessment should range from 0 to 3, based solely on the semantic similarity between the groundtruth and the candidate answer, disregarding any grammatical differences. "
    "A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect. "
    "A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect. "
    "A rating of 2 suggests high similarity, meaning the candidate answer is largely correct. "
    "Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct. "
    "Your response should be a single integer from 0, 1, 2, or 3."
)

MMV_DIMENSIONS = {
    "Coarse Perception": ["Video Topic", "Video Emotion", "Video Scene", "Video Style"],
    "Fine-grained Perception [Single-Instance]": [
        "OCR",
        "Object Recognition",
        "Attribute Recognition",
        "Event Recognition",
        "Human Motion",
        "Counting",
    ],
    "Fine-grained Perception [Cross-Instance]": [
        "Spatial Relationship",
        "Human-object Interaction",
        "Human Interaction",
    ],
    "Logic Reasoning": ["Structuralized Image-Text Understanding", "Mathematical Calculation"],
    "Attribute Reasoning": ["Physical Property", "Function Reasoning", "Identity Reasoning"],
    "Relation Reasoning": ["Natural Relation", "Physical Relation", "Social Relation"],
    "Common Sense Reasoning": ["Common Sense Reasoning"],
    "Temporal Reasoning": ["Counterfactual Reasoning", "Causal Reasoning", "Future Prediction"],
    "Hallucination": ["Hallucination"],
    "Audio": ["Audio"],
}
NoAH = []
for k, v in MMV_DIMENSIONS.items():
    if k not in ['Hallucination', 'Audio']:
        NoAH.extend(v)
MMV_DIMENSIONS['NoAH'] = NoAH
MMV_DIMENSIONS['Perception'] = MMV_DIMENSIONS['Coarse Perception'] + MMV_DIMENSIONS['Fine-grained Perception [Single-Instance]'] + MMV_DIMENSIONS['Fine-grained Perception [Cross-Instance]']
MMV_DIMENSIONS['Reasoning'] = MMV_DIMENSIONS['Logic Reasoning'] + MMV_DIMENSIONS['Attribute Reasoning'] + MMV_DIMENSIONS['Relation Reasoning'] + MMV_DIMENSIONS['Common Sense Reasoning'] + MMV_DIMENSIONS['Temporal Reasoning']

def build_prompt(item):
    tmpl = "Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: "
    return tmpl.format(item["question"], item["answer"], item["prediction"].replace('<|im_end|>',''))

model_map = dict(
    gpt35=partial(OpenAIWrapper, 'gpt-3.5-turbo-0613', verbose=True, retry=16),
    gpt35_alles=partial(OpenAIWrapperInternal, 'gpt-3.5-turbo-0613', retry=16, timeout=150, verbose=True, wait=15), 
    gpt4=partial(OpenAIWrapper, 'gpt-4-0613', verbose=True, retry=16),
    gpt4_alles=partial(OpenAIWrapperInternal, 'gpt-4-0613', retry=16, timeout=150, verbose=True, wait=15)
)



# 用pandas读取Excel文件
def read_excel_to_list_of_dicts(file_path):
    # 读取Excel文件，默认读取第一个工作表
    df = pd.read_excel(file_path, engine='openpyxl')

    # 将DataFrame转换为字典列表，每一行是一个字典
    list_of_dicts = df.to_dict(orient='records')

    # 遍历列表中的每一个字典
    for item in list_of_dicts:
        # 在这里处理每一个字典（即每一行）
        item['dimensions'] = eval(item['dimensions'])
    
    return list_of_dicts



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, default='gpt35')
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

def get_rating_distribution(data_path):
    distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    dataset = load(data_path)

    for item in tqdm(dataset):
        if "rating" not in item or item["rating"] not in ["0", "1", "2", "3"]:
            # print(item)
            # print(build_prompt(item))
            item["rating"] = can_infer_option(item["rating"], ["0", "1", "2", "3"]) 
            # item["rating"] = ('Failed to parse' + item['rating']) if 'rating' in item else ('Missing Field Rating')
            # input("人工标注得分：")
        try:
            rating = int(item['rating'])
            distribution[rating] += 1
        except Exception as e:
            print(item, item['rating'], 'Failed to parse')
            
    dump(dataset, data_path)
    return distribution

def mean_score(distribution):
    cnt, tot = 0, 0 
    for k, v in distribution.items():
        cnt += k * v
        tot += v
    return cnt / tot

def get_dimension_rating(data_path, dimensions=MMV_DIMENSIONS):
    data = load(data_path)
    sub_to_main = defaultdict(list)
    for main, subs in dimensions.items():
        for sub in subs:
            sub_to_main[sub].append(main)

    main_dimension_rating = {k: [] for k in dimensions.keys()}
    sub_dimension_rating = {k: [] for k in sub_to_main.keys()}
    for item in data:
        for d in item["dimensions"]:
            if type(item['rating']) == bool:
                item['rating'] = str(int(item['rating']))
            if len(item['rating']) == 1:
                sub_dimension_rating[d].append(int(item["rating"]))
                for sd in sub_to_main[d]:
                    main_dimension_rating[sd].append(int(item['rating']))

    main_dimension_rating = {k: np.mean(v) for k, v in main_dimension_rating.items()}
    sub_dimension_rating = {k: np.mean(v) for k, v in sub_dimension_rating.items()}
    return main_dimension_rating, sub_dimension_rating

def radar_graph(data):
    plt.figure(figsize=(10, 10))
    # radar chart
    labels = list(data.keys())
    data = list(data.values())
    angles = [n / float(len(labels)) * 2 * 3.1415926 for n in range(len(labels))]
    angles += angles[:1]
    data += data[:1]
    plt.polar(angles, data, "o-")
    plt.fill(angles, data, alpha=0.25)
    plt.ylim(0, 3)
    plt.xticks(angles[:-1], labels)

def report_rating(rating_file):
    distribution = get_rating_distribution(rating_file)
    ret = dict(distribution=distribution)
    main_rating, sub_rating = get_dimension_rating(rating_file)
    ret['main_rating'] = main_rating
    ret['sub_rating'] = sub_rating
    return ret

def visualize_rating(result, data_root):
    distribution = result['distribution']
    plt.bar(distribution.keys(), distribution.values())
    plt.title(f"Rating Distribution of {data_root}. \nAverage Rating: {mean_score(distribution):.2f}")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.savefig(osp.join(data_root, 'rating_dist.png'))

    main_rating, sub_rating = result['main_rating'], result['sub_rating']
    
    main_dim_csv = osp.join(data_root, 'main_dim.csv')
    sub_dim_csv = osp.join(data_root, 'sub_dim.csv')

    def d2df(D):
        return pd.DataFrame({x: [D[x]] for x in D})
    
    dump(d2df(main_rating), main_dim_csv)
    dump(d2df(sub_rating), sub_dim_csv)

    radar_graph(main_rating)
    plt.savefig(osp.join(data_root, 'main_rating.png'))

    radar_graph(sub_rating)
    plt.savefig(osp.join(data_root, 'sub_rating.png'))

# def main():
def MMBench_VIDEO_eval(data_file, model, nproc, verbose):
    
    data = read_excel_to_list_of_dicts(data_file)
    # model = model_map[args.model](system_prompt=system_prompt)
    print(model)
    model = OpenAIWrapper(model, retry=3, api_base='XIAOHAI',system_prompt=system_prompt, verbose=verbose)

    # for _, data_file in enumerate(data):
    data_root = data_file.replace('.xlsx', '')
    os.makedirs(data_root, exist_ok=True)

    tmp_file = osp.join(data_root, 'tmp.json')
    tgt_file = osp.join(data_root, 'rating.json')
    score_file = osp.join(data_root, 'score.json')

    while True:
        res = {} if not osp.exists(tmp_file) else load(tmp_file)
        res = {k: v for k, v in res.items() if model.fail_msg not in v}

        if len(res.keys()) == len(data):
            break
        
        if 'image.json' in data_file:
            for item in data:
                item['question'] = item['question'][len(IMAGE9_PROMPT): ] #重新截出来image

        data_un = [item for item in data if str(item['index']) not in res]
                
        prompts = [build_prompt(item) for item in data_un]
        
        indices = [item['index'] for item in data_un]

        if len(prompts):
            results = track_progress_rich(
                model.generate, 
                prompts, 
                keys=indices, 
                save=tmp_file,
                nproc=nproc,
                chunksize=nproc
            )
            for idx, r in zip(indices, results):
                res[idx] = r 
        dump(res, tmp_file)

    for item in data:
        if str(item['index']) in res.keys():
            item['rating'] = res[str(item['index'])]
    fail_cnt = 0
    for item in data:
        if 'rating' not in item.keys():
            fail_cnt += 1
        if model.fail_msg in item['rating']:
            item.pop('rating')
            fail_cnt += 1
    print(f'In {tgt_file}, {fail_cnt} rating records failed. ')
    dump(data, tgt_file)

    ret = report_rating(tgt_file)
    dump(ret, score_file)
    visualize_rating(ret, data_root)

# if __name__ == '__main__':
#     main()