from tqdm import tqdm
import json
from utils import load_data, dump_jsonl, parse_json, haversine_distance
from llm import LLaVA, Qwen, LLaVA_sft, Qwen_sft, CPM, CPM_sft
import os
import argparse
import numpy as np
import sys
import glob
import prompts
"""
Define the following parameters for evaluation:
    dataset: name of the dataset, could be Gws15k, im2gps3k, OpenWorld, and yfcc4k.
    model: the model used in the pipeline, could be llava or qwen.
    output_path: the output path.
"""

def Geoscore(distance):
    return 5000 * np.exp(-distance / 1492.7)

def find_file(pattern, folder_path):
    results = glob.glob(f'{folder_path}/{pattern}') 
    if len(results) > 0: 
        return results[0]
    else:
        return None  

class Ablation:

    def __init__(self, dataset_path, model, output_path,model_path,ckpt_dir):
        self.dataset_path = dataset_path
        self.model_type = model
        self.output_path = output_path
        if model == 'qwen':
            self.base_model = Qwen(model_path = model_path)  
        elif model == 'llava':
            self.base_model =LLaVA(model_path = model_path)
        else:
            self.base_model = CPM(model_path = model_path)
        self.ckpt_dir = ckpt_dir
        self.model_path = model_path



    def getReasoning(self,base = False):
        print("///////////The 1st stage: Reasoning///////////")
        # load dataset
        data = load_data(dataset_path+"/meta.jsonl") # it's dataset path because this is the first step
        # load model
        if not base:
            if self.model_type == "llava":
                reasoning_model = LLaVA_sft(model_path = self.model_path, ckpt_dir = self.ckpt_dir)
            elif self.model_type == "qwen":
                reasoning_model = Qwen_sft(model_path = self.model_path, ckpt_dir = self.ckpt_dir)
            else:
                reasoning_model = CPM_sft(model_path = self.model_path, ckpt_dir = self.ckpt_dir)
        else:
            reasoning_model = self.base_model
        for row in tqdm(data):
            image = f"{dataset_path}/images/{row['ID']}.jpg"
            query = prompts.reasoning_prompt
            response = reasoning_model.base_inference(query, image)
            row["image_reason"] = response
            yield row

    

    def guessCoordinates(self,load_file,choice): 
        print("///////////The 6th stage: Guessing the Coordinates///////////")
        data = load_data(load_file)
        '''
        create a query. this query forms like:
            base_query: a base query for models, which can be used for directly test model performance.
            intro_query: an intro query to introduce models other information that it can refer to.
            reason_query: the reasoning provided by our model.
            osm_query: searching results of the osm.
            rag_query: form rag results as query, with a distance threshold.
            outro_query: outro. if not including the extra information, the outro is not needed.
        '''
        base_query = prompts.base_query
        for row in tqdm(data):
            image = f"{dataset_path}/images/{row['ID']}.jpg"
            reason = row.get("image_reason","")
            osm_results = row.get("osm",None)
            comment = row.get('comment',{})
            rag = row.get("retrieved_content",{})
            rag_formed = ""
            rag_threshold = 30
            rag_keys = rag.keys()
            for rag_key in rag_keys:
                if not rag[rag_key]:
                    continue
                valid_results = [item for item in rag[rag_key] if item["distance"] <= rag_threshold]
                if not valid_results:
                    continue
                unique_clues = list(set(item["relevant_clue"] for item in valid_results))
                clues = ' '.join([item for item in unique_clues])
                rag_formed += f"the relevant clues of {rag_key} in this image are: {clues}"
            
            comment_formed=""
            for category in comment.keys():
                if not comment[category]:
                    continue
                comment_formed += f"{category}: {comment[category]} \n"

            filtered_Query = {key: [v for v in value if v != 'None'] for key, value in row.get('genQuery',{}).items()}
            filtered_Query = {key: value for key, value in filtered_Query.items() if value}

            # define the queries
            intro_query = prompts.intro_query
            reason_query = prompts.reason_query_template.format(reason = reason)
            
            rag_query = prompts.rag_query_template.format(rag_formed = rag_formed)
            comment_query = prompts.comment_query_template.format(comment_formed = comment_formed)
            osm_query = prompts.osm_query_template.format(filtered_Query = filtered_Query, osm_results = osm_results)
            outro_query = prompts.outro_query
            # decide whether to append the query or not, using keys
            k_intro = 0
            if choice == 1:
                k_reason = 0 
                k_osm = 1 if osm_results else 0
                k_rag = 1 if rag_formed else 0
                k_comment = 1 if comment_formed else 0
            elif choice == 2 or choice == 3 :
                k_reason = 1 
                k_osm = 0 if osm_results else 0
                k_rag = 0 if rag_formed else 0
                k_comment = 0 if comment_formed else 0
            elif choice == 4:
                k_intro = 0
                k_reason = 0 
                k_osm = 0 if osm_results else 0
                k_rag = 0 if rag_formed else 0
                k_comment = 0 if comment_formed else 0
            k_outro = 0 
            # form the queries

            usage = {"reasoning": k_reason, "osm": k_osm, "rag": k_rag, "comment": k_comment}
            row["usage"] = usage
            query = base_query + intro_query * k_intro + reason_query * k_reason + osm_query * k_osm + comment_query * k_comment + rag_query * k_rag + outro_query * k_outro
            print(query)
            answer = self.base_model.base_inference(query, image) 
            print(f"model response {answer}")
            answer = parse_json(answer)
            print(f"parser response {answer}")
            print("correct answer:", row["LAT"], row["LON"])
            sys.stdout.flush()
            row["answer"] = answer
            yield row



    def guess_forward(self,load_file,choice):
        # Guess Stage:
        o_file = f"{output_path}/{results_fileName}"
        dump_jsonl(self.guessCoordinates(load_file,choice), o_file)


    def calculate_score(self):
        data = load_data(f"{output_path}/{results_fileName}")
        counts = [0, 0, 0, 0, 0]
        total_points = 0
        Distance = 0
        thresholds = [1, 25, 200, 750, 2500]
        for row in data:
            correct_answer = [float(row["LAT"]), float(row["LON"])]
            try:
                guessed_answer = [float(row["answer"]["latitude"]), float(row["answer"]["longitude"])]
                distance = haversine_distance(guessed_answer, correct_answer)
            except:
                guessed_answer = [0, 0]
                distance = 10000
            points = Geoscore(distance)
            total_points += points
            Distance += distance
            # row["distance"] = distance
            # print(guessed_answer, correct_answer, distance)
            for i, t in enumerate(thresholds):
                if distance <= t:
                    counts[i] += 1
        total_num = len(data)
        score = [count / total_num for count in counts]
        print(f"Five LeveL: Street-> Continent{score}")
        print(f"Avg.Geoscore is {total_points/total_num}")
        print(f"Avg.distance is {Distance/total_num}")

    def calculate_score_cc(self):
        data = load_data(f"{output_path}/{results_fileName}")
        counts = [0, 0]
        for row in data:
            try:
                correct_country = row["country"]
                correct_city = row["city"]
                predicted_country = row["answer"]["country"]
                predicted_city = row["answer"]["city"]
                if correct_country in predicted_country:
                    counts[0] += 1
                if correct_city in predicted_city:
                    counts[1] += 1
            except:
                continue
        total_num = len(data)
        score = [count / total_num for count in counts]
        print(score)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/fs/clip-projects/geoguesser/runze/data/Gws15k",help="Please input valid dataset path,only for Gws5k")
    parser.add_argument('--model', type=str, default='qwen',choices=["qwen", "llava", "cpm"])
    parser.add_argument('--reasoning_path',  type=str, default ='.')
    parser.add_argument('--results_file_Name', type=str, default = 'Final_results.jsonl')
    parser.add_argument('--model_path', type=str, default = 'vlms/qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--ckpt_dir', type=str, default = 'vlms/qwen/checkpoint-534')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--WO_reasoning', action='store_true', default=False, help='Option for without reasoning')
    group.add_argument('--WO_tools', action='store_true', default=False, help='Option for without tools')
    group.add_argument('--direct_guess', action='store_true', default=False, help='Option for direct guess')
    group.add_argument('--base_guess', action='store_true', default=False, help='Option for guess using base reasoning')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    dataset_path = args.dataset_path
    model = args.model
    output_path = args.reasoning_path
    results_fileName = args.results_file_Name
    model_path = args.model_path
    ckpt_dir = args.ckpt_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    

    ablation = Ablation(dataset_path, model, output_path, model_path, ckpt_dir)
    #choice 1:
    if args.WO_reasoning:
        results_s5 = find_file("results_s5.jsonl",output_path)
        if results_s5:
            ablation.guess_forward(results_s5,1)
            ablation.calculate_score()
            ablation.calculate_score_cc()
        else:
            print("Can't find corresponding file. Please run evalution.py first")


    #choice 2:
    elif args.WO_tools:
        results_s1 = find_file("results_s1.jsonl",output_path)
        if results_s1:
            ablation.guess_forward(results_s1,2)
            ablation.calculate_score()
            ablation.calculate_score_cc()
        else:
            print("Can't find corresponding file. Please run evalution.py first")

    #choice 3:
    elif args.base_guess:
        results_s1_base = find_file("results_s1_base.jsonl",output_path)
        if results_s1_base:
            ablation.guess_forward(results_s1_base,3)
        else:
            o_file = f"{output_path}/results_s1_base.jsonl"
            dump_jsonl(ablation.getReasoning(base=True),o_file)
            results_s1_base = find_file("results_s1_base.jsonl",output_path)
            ablation.guess_forward(results_s1_base,3)
        ablation.calculate_score()
        ablation.calculate_score_cc()
    
    #choice 4:
    elif args.direct_guess:
        ablation.guess_forward(dataset_path+"/meta.jsonl",4)
        ablation.calculate_score()
        ablation.calculate_score_cc()