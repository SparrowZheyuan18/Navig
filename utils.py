import json
import requests
import torch
import clip
from PIL import Image
import numpy as np
import re
import time
import math
from retry import retry
import faiss

import os
from groundingdino.util.inference import load_model, load_image, predict,annotate
import torch
from torchvision.ops import box_convert
from collections import defaultdict
import cv2


def load_data(path):
    with open(path, 'r',encoding = 'utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def dump_jsonl(objects, file_name):
    out_file = open(file_name, "w",encoding='utf-8')
    for obj in objects:
        tmp = out_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out_file.flush()


@retry(tries=3, delay=2)
def search_place_nominatim(query: str, top_k=3):
    query = query.replace("#","")
    url = f"XXXX={query}&format=json"
    headers = {
        "User-Agent": "Geoguess (change404.forever@gmail.com)"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  
    time.sleep(1)  # Respect server rate limit
    try:
        content = response.json()
    except Exception as e:
        print("Error decoding JSON:", response.text)
        return None
    if not content:
        return None

    content = [
        {
            "place_name": item.get("name", "N/A"),
            "location": item["display_name"],
            "lat": item["lat"],
            "lon": item["lon"]
        } for item in content[:top_k]
    ]
    return content

def search_place_with_retry(query: str, top_k=3):
    try:
        return search_place_nominatim(query, top_k)
    except Exception as e:
        print(f"All retries failed: {e}")
        return None
    

def retrieve_similar_images(input_image_path, k=5, threshold=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_features_array = np.load("guidebook/image_features.npy")  
    index = faiss.read_index("guidebook/faiss_index.index")  

    with open("guidebook/text_descriptions.txt", "r", encoding='utf-8') as f:
        text_descriptions = [line.strip() for line in f.readlines()]

    with open("guidebook/image_paths.txt", "r", encoding='utf-8') as f:
        image_paths = [line.strip() for line in f.readlines()]
    input_image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        input_image_features = model.encode_image(input_image).cpu().numpy()
    input_image_features = input_image_features.astype('float32')
    distances, indices = index.search(input_image_features, k)

    filtered_similar_texts = []
    filtered_similar_images = []
    filtered_distances = []

    for i, distance in enumerate(distances[0]):
        if threshold is None or distance < threshold:
            filtered_similar_texts.append(text_descriptions[indices[0][i]])
            filtered_similar_images.append(image_paths[indices[0][i]])
            filtered_distances.append(float(distance))
    
    if not filtered_similar_images:
        return [], [], []
    # print(filtered_similar_images, filtered_similar_texts, filtered_distances)
    return filtered_similar_images, filtered_similar_texts, filtered_distances


def parse_guess(guess):
    pattern = r'\(([^)]+),\s*([^)]+)\)'
    match = re.search(pattern, guess)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    else:
        raise ValueError("The answer is not in the correct format.")


def parse_json_part(guess):
    if guess is None:
        return None
    if guess.startswith("```json"):
        guess = guess[7:].strip()  
    if guess.endswith("```"):
        guess = guess[:-3].strip()
    try:
        parsed_dict = json.loads(guess)
        return parsed_dict
    except json.JSONDecodeError as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        fixed_json_str = re.sub(r'(":\s*)(-?\d+\.?\d*)"', r'\1\2', guess)
        try:
            parsed_dict = json.loads(fixed_json_str)
            return parsed_dict
        except json.JSONDecodeError as e:
            error_message = str(e)
            print(f"Again Error: {error_message}")
            fixed_json_str = re.sub(r',(\s*})', r'\1', guess)
            try:
                parsed_dict = json.loads(fixed_json_str)
                return parsed_dict
            except json.JSONDecodeError as e:
                return None

def parse_json(guess):
    matches = re.findall(r'\{.*?\}', guess, re.DOTALL)
    for match in matches:
        return parse_json_part(match)


def haversine_distance(guess, answer):
    '''
    guess and answer are both:
    [lat, lng]
    '''
    lat1, lon1 = guess
    lat2, lon2 = answer
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance



class PatchImages :
    def __init__(self, geo_objects: list, groundingDinoConfigPath = None, WeightsPath = None):
        self.geo_objects = geo_objects
        if groundingDinoConfigPath is None:
            CONFIG_PATH = os.path.join(os.getcwd(), "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        else:
            CONFIG_PATH = os.path.join(os.getcwd(),groundingDinoConfigPath)

        if WeightsPath is None:
            WEIGHTS_PATH = os.path.join(os.getcwd(), "GroundingDINO", "weights", "groundingdino_swint_ogc.pth")
        else:
            WEIGHTS_PATH = os.path.join(os.getcwd(), WeightsPath)

        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        self.saveImgs = defaultdict(list)
    def __call__(self, image_path: str, BOX_TRESHOLD: float = 0.3, TEXT_TRESHOLD: float = 0.25)->dict:
        '''
        Input: Picture, list of geo-objects(string)
        Output: list of image patches
        '''
        image_patches = {}
        for geo_object in self.geo_objects:
            TEXT_PROMPT = geo_object
            image_patches[geo_object] = []
            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                                model=self.model, 
                                image=image, 
                                caption=TEXT_PROMPT, 
                                box_threshold=BOX_TRESHOLD, 
                                text_threshold=TEXT_TRESHOLD
                            )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            self.saveImgs[geo_object].append(annotated_frame)
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                if abs(x2-x1) < 224 and abs(y2-y1) < 224:
                    continue
                image_patches[geo_object].append(image_source[int(y1):int(y2), int(x1):int(x2)])

        return image_patches

    def save_annotation(self,output_path:str):
        '''
        Input: list of image patches: category->list of image patches
        Output: save the patches
        '''
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        for geo_object in self.geo_objects:
            for i, img in enumerate(self.saveImgs[geo_object]):
                cv2.imwrite(os.path.join(output_path, f"{geo_object}_{i}.jpg"), img)
