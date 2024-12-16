import json
import torch
from utils import load_data, dump_jsonl, parse_json, haversine_distance,search_place_with_retry, PatchImages
from utils import retrieve_similar_images
from llm import LLaVA, Qwen, LLaVA_sft, Qwen_sft, CPM, CPM_sft
import os
import argparse
import gc
import sys
from PIL import Image
import prompts
"""
Define the following parameters for evaluation:
    dataset: name of the dataset, could be Gws15k, im2gps3k, OpenWorld, and yfcc4k.
    model: the model used in the pipeline, could be llava or qwen.
"""


class Inference:

    def __init__(self, model, crop_box_treshold, crop_text_treshold, model_path, ckpt_dir):
        self.model_type = model
        if model == 'qwen':
            self.base_model = Qwen(model_path = model_path)
            self.reasoning_model =  Qwen_sft(model_path = model_path, ckpt_dir = ckpt_dir) 
        elif model == 'llava':
            self.base_model =LLaVA(model_path = model_path)
            self.reasoning_model = LLaVA_sft(model_path = model_path, ckpt_dir = ckpt_dir)
        else:
            self.base_model = CPM(model_path = model_path)
            self.reasoning_model = CPM_sft(model_path = model_path, ckpt_dir = ckpt_dir)
        
        self.crop_box_treshold = crop_box_treshold
        self.crop_text_treshold = crop_text_treshold
        



    def getReasoning(self,image_path):
        print("///////////The 1st stage: Reasoning///////////")
        self.results = {}
        query = prompts.reasoning_prompt
        response = self.reasoning_model.base_inference(query, image_path)
        print(f"MACRONAV : {response}")
        self.results["image_reason"] = response



        
    def getGrounding(self,image_path):
        print("///////////The 2th stage: Grounding///////////")
        ground = PatchImages(['road sign', 'house', 'building sign'])
        patch_dir = 'output/patches'
        
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        
        basename = os.path.basename(image_path)
        filename, _ = os.path.splitext(basename) 
        patchResult = ground(image_path,BOX_TRESHOLD=self.crop_box_treshold,TEXT_TRESHOLD=self.crop_text_treshold)
        tmp = {}
        for type, cropped_ls in patchResult.items() :
            tmp[type] = []
            for i, cropped_img in enumerate(cropped_ls):
                try:
                    cropped_img = Image.fromarray(cropped_img)
                    save_path = os.path.join(patch_dir, f"{filename}_{type}_{i}.jpg")
                    cropped_img.save(save_path)
                    tmp[type].append(save_path)
                except:
                    continue
        self.results["crop"] = tmp
        print(f"MICRONAV :Grounding the Image Details results in {patch_dir}")

        

    

    def getRAG(self):
        print("///////////The 3th stage: Retriving with Grounding Images///////////")
        retrieved_dict = {}
        crop_dict = self.results["crop"]
        items = crop_dict.keys()
        for item in items:
            images = crop_dict[item]
            if images:
                sim_images, sim_texts, distances = retrieve_similar_images(images[0], threshold=40)
                if sim_images:
                    retrieved_dict[item] = [
                        {
                            "similar_image": sim_images[i],
                            "relevant_clue": sim_texts[i],
                            "distance": distances[i]
                        } for i in range(len(sim_images))
                    ]
                else:
                    retrieved_dict[item] = []
            else:
                retrieved_dict[item] = []
        
        self.results["retrieved_content"] = retrieved_dict
        print(f"MICRONAV Tools using: Guidebook {retrieved_dict}")



    def getComment(self):
        print("///////////The 4th stage: Commenting with Grounding Images///////////")
        commented_dict = {}
        crop_dict = self.results["crop"]
        items = crop_dict.keys()
        for item in items:
            if item == 'house':
                k = 3
            else:
                k = len(crop_dict[item])
            commented_dict[item] = ""
            query = prompts.comment_gen_template.format(item = item)
            images = crop_dict[item][:k]
            if images:
                for image in images:
                    response = self.base_model.base_inference(query,image)
                    commented_dict[item] += response + "\t"
            else:
                commented_dict[item] = ""
        
        self.results["comment"] = commented_dict
        print(f"MICRONAV Tools using: VLM {commented_dict}")

    def getOSM(self):       
        print("///////////The 5rd stage: Search OCR///////////")

        prompt =prompts.osm_gen
        self.results['genQuery'] = {}
        self.results['osm'] = None
        for category in self.results["crop"].keys() :
            self.results['genQuery'][category] = []
            images = self.results["crop"][category]
            for image in images:
                query = self.base_model.base_inference(prompt, image)
                print(query)
                self.results['genQuery'][category].append(query)
                if 'None' in query:
                    self.results['osm'] = None
                else:
                    response = search_place_with_retry(query, top_k=3)
                    if self.results['osm'] is None:
                        self.results['osm'] = response
                    elif response is not None:
                        self.results['osm'].extend(response)
        print(f"MICRONAV Tools using: Map {self.results['osm']}")
    

    def guessCoordinates(self,image_path): 
        print("///////////The 6th stage: Guessing the Coordinates///////////")
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
        image = image_path
        reason = self.results.get("image_reason","")
        osm_results = self.results.get("osm",None)
        comment = self.results.get('comment',{})
        rag = self.results.get("retrieved_content",{})
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

        filtered_Query = {key: [v for v in value if v != 'None'] for key, value in self.results.get('genQuery',{}).items()}
        filtered_Query = {key: value for key, value in filtered_Query.items() if value}

        # define the queries
        intro_query = prompts.intro_query
        reason_query = prompts.reason_query_template.format(reason = reason)
        
        rag_query = prompts.rag_query_template.format(rag_formed = rag_formed)
        comment_query = prompts.comment_query_template.format(comment_formed = comment_formed)
        osm_query = prompts.osm_query_template.format(filtered_Query = filtered_Query, osm_results = osm_results)
        outro_query = prompts.outro_query
        # decide whether to append the query or not, using keys
        k_intro = 1
        k_reason = 1
        k_osm = 1 if osm_results else 0
        k_rag = 1 if rag_formed else 0
        k_comment = 1 if comment_formed else 0
        k_outro = 1 if k_reason else 0
        # form the queries
        usage = {"reasoning": k_reason, "osm": k_osm, "rag": k_rag, "comment": k_comment}
        self.results["usage"] = usage
        query = base_query + intro_query * k_intro + reason_query * k_reason + osm_query * k_osm + comment_query * k_comment + rag_query * k_rag + outro_query * k_outro
        print(query)
        answer = self.base_model.base_inference(query, image) 
        answer = parse_json(answer)
        sys.stdout.flush()
        self.results["answer"] = answer

        print(f"Guess answer {answer}")


    def forward(self, image_path):
        self.getReasoning(image_path)
        self.getGrounding(image_path)
        self.getRAG()
        self.getComment()
        self.getOSM()
        self.guessCoordinates(image_path)
        
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="",help="input a valid image path")
    parser.add_argument('--model', type=str, default='qwen',choices=["qwen", "llava", "cpm"])
    parser.add_argument('--crop_box_treshold', type=float, default = 0.65)
    parser.add_argument('--crop_text_treshold', type=float, default = 0.55)
    parser.add_argument('--model_path', type=str, default = 'vlms/qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--ckpt_dir', type=str, default = 'vlms/qwen/checkpoint-534')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args) 
    image_path = args.image_path
    model = args.model
    crop_box_treshold = args.crop_box_treshold
    crop_text_treshold = args.crop_text_treshold
    model_path = args.model_path
    ckpt_dir = args.ckpt_dir
    inference = Inference(model,crop_box_treshold, crop_text_treshold, model_path, ckpt_dir)
    inference.forward(image_path)