# Navig_2025

## preliminary
- torch: 2.2.2, CUDA: 11.8, python: 3.10.14
- Creat the environment according to environment.txt
- Download [GroundingDINO](XXXX) to the project root directory. 
- VLMs weight: The weight files are in [link](XXXX), download them and replace 'vlms/*/checkpoint-534' folder. The model file can download from huggingface soon.



## Evalute
```
python evaluation.py --model $model --model_path $model_path --ckpt_dir $ckpt_dir --dataset_path $dataset_path --reasoning_path $reasoning_path --results_file_Name $results_file_Name  --crop_box_treshold $crop_box_treshold --crop_text_treshold $crop_box_treshold
```

Explaination:

--model: which model you want to use, ["qwen", "llava", "cpm"]

--dataset_path: The dataset path which should include images folder and meta.jsonl

--reasoning_path: Output_path

--results_file_Name: The name of final results file.

--crop_box_treshold: hyperparameters of GroundingDINO's, (0,1)

--crop_text_treshold: hyperparameters of GroundingDINO's, (0,1)


Example:
```
python evaluation.py --model "llava" --dataset_path dataset/Gws15k --reasoning_path output/Gws15k --results_file_Name "results_s6_llava.jsonl"  --crop_box_treshold 0.3 --crop_text_treshold 0.25 --model_path vlms/llava/llava-v1.6-vicuna-7b-hf --ckpt_dir vlms/llava/checkpoint-534
```

## Inference

```
python inference.py --model $model --model_path $model_path --ckpt_dir $ckpt_dir --image_path $image_path  --crop_box_treshold $crop_box_treshold --crop_text_treshold $crop_box_treshold
```

Explanation:

--model: which model you want to use, ["qwen", "llava", "cpm"]

--image_path: The image you want to infer

--crop_box_treshold: hyperparameters of GroundingDINO's, (0,1)

--crop_text_treshold: hyperparameters of GroundingDINO's, (0,1)


Example:
```
python inference.py --model "qwen" --image_path "examples/-33.3741232_-70.661976.jpg"  --crop_box_treshold 0.3 --crop_text_treshold 0.25 --model_path vlms/qwen/Qwen2-VL-7B-Instruct --ckpt_dir vlms/qwen/checkpoint-534
```
