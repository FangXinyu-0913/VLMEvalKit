from openai import OpenAI
import openai
# from openai import OpenAI
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
import time
from vlmeval.smp import *
from .misc import build_judge
from vlmeval.utils import track_progress_rich



# client = OpenAI(
#     api_key=api_key,
#     # api_base=api_base
# )
def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="", type=str, help="OpenAI API base.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args

def mvbench_check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def MVBench_eval(eval_file, result_save_path, result_leader_board_path):
    logger = get_logger('Evaluation')
    # print(eval_file)
    data = load(eval_file)
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    assert 'answer' in data and 'prediction' in data
    data = [data.iloc[i] for i in range(len(data))]
    for example in tqdm(data):
        print(example)
        task_type = example['split']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total        acc_dict[task_type][1] += 1
        total += 1
        acc_dict[task_type][1] += 1

        res_list.append({
            'pred': str(example['prediction']),
            'gt': str(example['answer'])
        })
        if mvbench_check_ans(pred=str(example['prediction']), gt=str(example['answer'])):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    with open(result_save_path, "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    correct = 0
    total = 0
    print(acc_dict)
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    print(final_res)

    with open(result_leader_board_path, "w") as f:
        json.dump(final_res, f)


def VIDEO_eval(model, key, question, answer, pred, output_dir):
    prompt = f"Please evaluate the following video-based question-answer pair:\nQuestion: {question}\nCorrect Answer: {answer}\nPredicted Answer: {pred}\n \
Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.\n \
Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.\n \
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. \n \
For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."
    retry = 5
    for i in range(retry):
        try:
            response_message = model.generate(prompt)
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, {'q': question, 'a': answer, 'pred': pred}]
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)
            break
        except Exception as e:
            pass

    return [{'pred': 'no', 'score': 0}, {'q': question, 'a': answer, 'pred': pred}]



def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """

    for file in caption_files:
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]
            # print(result_qa_pair)

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            time.sleep(10)

def Video_eval(pred_path, output_dir, output_json, score_result_file, model, nproc, verbose):
    """
    Main function to control the flow of the program.
    """
    if not os.path.exists(output_json):
        data = load(pred_path)
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        data['id'] = [str(x) for x in data['id']]
        # data['index'] = [str(x) for x in data['answer']]
        data['question'] = [str(x) for x in data['question']]
        new_pred_contents = []
        for id, q, a, idx, pred in zip(data['id'], data['question'], data['answer'], data['index'], data['prediction']):
            new_pred_contents.append({'id': id, 'question': q, 'answer': a, 'index': idx, 'pred': pred})
        
        # Generating list of id's and corresponding files
        id_list = [x['id'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        # Generate output directory if not exists.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Preparing dictionary of question-answer sets
        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['id']
            question = sample['question']
            answer = sample['answer']
            pred = sample['pred'].replace('<|im_end|>', '')
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set

        num_tasks = verbose
        model = build_judge(model, api_base='XIAOHAI', verbose=verbose, retry=2)
        retry_times = 0
        # While loop to ensure that all captions are processed.
        while True:
            # try:
                
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            # print(incomplete_files)
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1


            tups = []
            for file in incomplete_files:
                key = file[:-5] # Strip file extension
                qa_set = prediction_set[key]
                question = qa_set['q']
                answer = qa_set['a']
                pred = qa_set['pred']
                tups.append((model, key, question, answer, pred, output_dir))

            retry_times += 1
            if retry_times > 3:
                for file in incomplete_files:
                    key = file[:-5] # Strip file extension
                    qa_set = prediction_set[key]
                    question = qa_set['q']
                    answer = qa_set['a']
                    pred = qa_set['pred']
                    result_qa_pair = [{'pred': 'no', 'score': 0}, {'q': question, 'a': answer, 'pred': pred}]
                    with open(f"{output_dir}/{key}.json", "w") as f:
                        json.dump(result_qa_pair, f)
                break
                

            track_progress_rich(VIDEO_eval, tups, nproc=nproc, chunksize=nproc) 


            # except Exception as e:
            #     print(f"Error: {e}")

        # Combine all the processed files into one
        combined_contents = {}
        json_path = output_json

        # Iterate through json files
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    try:
                        content = json.load(json_file)
                    except:
                        print(file_name)
                        continue
                    combined_contents[file_name[:-5]] = content

        # Write combined content to a json file
        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)

        print(f"All evaluation completed! result saved in {json_path}")

        #remove all file in output_dir
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        #remove the output_dir
        os.rmdir(output_dir)
    else:
    #open existing json_file
        print(f"{output_json} already exists!")
        with open(output_json, "r") as json_file:
            combined_contents = json.load(json_file)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print('input:', pred_path)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)

    result = {'input': pred_path, 'yes_count': yes_count, 'no_count': no_count, 'accuracy': accuracy, 'average_score': average_score}
    
    with open(score_result_file, "w") as f:
        json.dump(result, f)

    print(f"score Result saved to {score_result_file}")

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    print(args.pred_path)

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    '''
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)
    '''
    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                try:
                    content = json.load(json_file)
                except:
                    print(file_name)
                    continue
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print('input:',args.pred_path)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()
