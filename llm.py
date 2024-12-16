# -*- coding: utf-8 -*-Z
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
from swift.tuners import Swift
import torch
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import base64

from openai import OpenAI
from configuration import Config
    

class LLaVA:

    # inference code of LLaVA model
    def __init__(self, model_path='vlms/llava/llava-v1.6-vicuna-7b-hf'):
        self.model_type = 'llava1_6-vicuna-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)


    def base_inference(self, query, image_path = None):
        query = '<image>' + query
        if image_path and not isinstance(image_path,list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class LLaVA_sft:

    # inference code of LLaVA model
    def __init__(self, model_path='vlms/llava/llava-v1.6-vicuna-7b-hf', ckpt_dir="vlms/llava/checkpoint-534"):
        self.ckpt_dir = ckpt_dir
        self.model_type = 'llava1_6-vicuna-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)
        self.model = Swift.from_pretrained(self.model, self.ckpt_dir, inference_mode=True)


    def base_inference(self, query, image_path):
        query = '<image>' + query
        image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class Qwen:

    # inference code of Qwen model
    def __init__(self, model_path='vlms/qwen/Qwen2-VL-7B-Instruct'):
        self.model_type = 'qwen2-vl-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)


    def base_inference(self, query, image_path = None):
        query = '<image>' + query
        if image_path and not isinstance(image_path,list):
            image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response
    

class Qwen_sft:

    # inference code of Qwen model
    def __init__(self, model_path='vlms/qwen/Qwen2-VL-7B-Instruct', ckpt_dir="vlms/qwen/checkpoint-534"):
        self.ckpt_dir = ckpt_dir
        self.model_type = 'qwen2-vl-7b-instruct'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)
        self.model = Swift.from_pretrained(self.model, self.ckpt_dir, inference_mode=True)

    
    def base_inference(self, query, image_path):
        query = '<image>' + query
        image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response


class CPM:
    def __init__(self, model_path='vlms/cpm/MiniCPM-V-2_6'):
        self.model_type = 'minicpm-v-v2_6-chat'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)


    def base_inference(self, query, image_path = None):
        query = '<image>' + query
        if image_path and not isinstance(image_path,list):
            image_path = [image_path]
        
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response



class CPM_sft:

    def __init__(self, model_path='vlms/cpm/MiniCPM-V-2_6', ckpt_dir="vlms/cpm/checkpoint-534"):
        self.ckpt_dir = ckpt_dir
        self.model_type = 'minicpm-v-v2_6-chat'
        self.template_type = get_default_template_type(self.model_type)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.float16, model_id_or_path=self.model_path,
                                        model_kwargs={'device_map': 'auto'})
        self.model.generation_config.max_new_tokens = 256
        self.template = get_template(self.template_type, self.tokenizer)
        seed_everything(42)
        self.model = Swift.from_pretrained(self.model, self.ckpt_dir, inference_mode=True)

    
    def base_inference(self, query, image_path):
        query = '<image>' + query
        image_path = [image_path]
        response, _ = inference(self.model, self.template, query, images=image_path)
        return response
    

