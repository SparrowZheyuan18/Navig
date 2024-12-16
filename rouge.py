from tqdm import tqdm
import json
from utils import load_data, dump_jsonl, parse_json, haversine_distance,search_place_with_retry, PatchImages
from utils import retrieve_similar_images
from llm import LLaVA, Qwen, LLaVA_sft, Qwen_sft, CPM, CPM_sft
import os
import argparse
import numpy as np
import sys
from PIL import Image
from rouge_score import rouge_scorer
from openai import OpenAI
import prompts
from configuration import Config
"""
Caculate the rouge scores between model gengerated and people response
"""
def Geoscore(distance):
    return 5000 * np.exp(-distance / 1492.7)

class Evaluator:

    def __init__(self, model, output_path):
        self.model_type = model
        self.output_path = output_path


    def getReasoning(self):
        print("///////////The 1st stage: Reasoning///////////")
        # load dataset
        data = load_data("/fs/clip-projects/geoguesser/test_set.jsonl") # it's dataset path because this is the first step
        # load model
        if self.model_type == "llava":
            reasoning_model = LLaVA()
        elif self.model_type == "qwen":
            reasoning_model = Qwen()
        elif self.model_type == "cpm":
            reasoning_model = CPM()
        elif self.model_type == "qwen_sft":
            reasoning_model = Qwen_sft()
        elif self.model_type == "cpm_sft":
            reasoning_model = CPM_sft()
        else:
            reasoning_model = LLaVA_sft()
        for row in tqdm(data):
            image = row["images"][0]
            query = prompts.reasoning_prompt
            response = reasoning_model.base_inference(query, image)
            row["model_response"] = response
            yield row


    def get_quality_eval(self):
        data = load_data("/fs/clip-projects/geoguesser/runze/data/results/Gws5k_GuesserPro(QwenVL2)/sft_W_transcript3/results_s1.jsonl")
        client = OpenAI(api_key = Config.OPENAI_API_KEY)
        prompt_template = '''Please generate a JSON object based on the given reasoning text, correct country name, city name, and coordinates. The format should be as follows:
        {
            "country": <0 or 1>,
            "country_correct": <0 or 1>,
            "others": <0 or 1>,
            "others_correct": <0 or 1>
        }
        Annotation Guidelines:
        country:
        If the reasoning includes a prediction about the country, set country = 1, otherwise set country = 0.
        If country = 1, determine if the prediction is correct based on the provided correct answer. Set country_correct = 1 if correct, otherwise set country_correct = 0.
        If country = 0, then country_correct should automatically be set to 0.
        others:
        If the reasoning includes predictions at a finer granularity (e.g., city, town, etc.), set others = 1, otherwise set others = 0.
        If others = 1, determine if the prediction is correct based on the provided correct answer. Set others_correct = 1 if correct, otherwise set others_correct = 0.
        Please make decisions based on the content of the reasoning and the correct answers provided.\n
        '''
        for row in tqdm(data):
            lat = row["LAT"]
            lon = row["LON"]
            country = row["country"]
            city = row["city"]
            text = row["image_reason"]
            special = f'''The reasoning text is: {text}
                correct country name:{country}
                correct city name: {city}
                coordinates: Latitude:{lat}, Longitude:{lon}'''
            prompt = prompt_template + special
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]
            )
            print(response.choices[0].message.content)
            row["quality"] = parse_json(response.choices[0].message.content)
            yield row

        
    def get_rouge(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        candidate_data = load_data(os.path.join(output_path,"tmp.jsonl"))
        for candidate_item in candidate_data:
            scores = scorer.score(candidate_item["response"], candidate_item["model_response"])
            candidate_item["score"] = scores
            yield candidate_item



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen',choices=["qwen", "llava", "cpm","qwen_sft", "llava_sft", "cpm_sft"])
    parser.add_argument('--reasoning_path',  type=str, default ='.')
    parser.add_argument('--results_file_Name', type=str, default = 'Final_results.jsonl')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    model = args.model
    output_path = args.reasoning_path
    results_fileName = args.results_file_Name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    evaluator = Evaluator(model, output_path)
    o_file = f"{output_path}/{results_fileName}"
    dump_jsonl(evaluator.get_quality_eval(),o_file)
    o_file = f"{output_path}/tmp.jsonl"
    dump_jsonl(evaluator.getReasoning(),o_file)
    
    o_file = f"{output_path}/{results_fileName}"
    dump_jsonl(evaluator.get_rouge(),o_file)
    o_file = f"{output_path}/{results_fileName}1"
    dump_jsonl(evaluator.get_fine_answer(),o_file)
    
    file_ls = ["cpm.jsonl", "cpm_sft.jsonl", "qwen.jsonl", "qwen_sft.jsonl", "llava.jsonl", "llava_sft.jsonl"]
    for file in file_ls:
        data = load_data(os.path.join(output_path,file))
        rouge1 = [0,0,0]
        rouge2 = [0,0,0]
        rougeL = [0,0,0]
        for item in data:
            rouge1 = [a + b for a, b in zip(item["score"]["rouge1"],rouge1)]
            rouge2 = [a + b for a, b in zip(item["score"]["rouge2"],rouge2)]
            rougeL = [a + b for a, b in zip(item["score"]["rougeL"],rougeL)]
        rouge1 = [item/50 for item in rouge1]
        rouge2 = [item/50 for item in rouge2]
        rougeL = [item/50 for item in rougeL]
        print(f"{file} avg.rouge results:")
        print(f"rouge1 is {rouge1}; rouge2 is {rouge2}; rougeL is {rougeL}")




