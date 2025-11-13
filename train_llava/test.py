from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoProcessor,LlavaProcessor
from typing import List

class LlavaDataset(Dataset):

    def __init__(self, data_dir) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(data_dir=data_dir)

    def build_dataset(self, data_dir) -> tuple[List, Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        images_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(path_or_buf=chat_file).to_dict("records")
        return chat_data, images_dir

    def __len__(self)-> int:
        return len(self.chat_data)

    def __getitem__(self, index) -> tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        humen_input = cur_data["conversations"][0]["value"]
        gpt_output = cur_data["conversations"][1]["value"]
        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return (humen_input, gpt_output, image_path)


def build_qaimage(processor, q_text: str, a_text: str, image_path: Path):

    # 千问的对话模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 是否直接返回token ID（默认False，返回字符串）
        add_generation_prompt=True,  # 是否在末尾添加生成提示（如"Assistant:"）
    )
    image_file = image_path
    raw_image = Image.open(image_file)

    inputs = processor(prompt, raw_image, return_tensors="pt")

    return prompt  # ,inputs


if __name__ == "__main__":
    data_dir = "datasets/LLaVA-CC3M-Pretrain-595K"
    test_llavadataset = LlavaDataset(data_dir)
    llava_model_path = "models/llava_clip-L-14-336_Qwen1.5-1.8B"
    llava_process = LlavaProcessor.from_pretrained(llava_model_path, use_fast=True)

    test_data = test_llavadataset[123]
    llava_process()


    # 千问的对话模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "q_text"},
    ]
    prompt = llava_process.tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 是否直接返回token ID（默认False，返回字符串）
        add_generation_prompt=True,  # 是否在末尾添加生成提示（如"Assistant:"）
    )
    image_file = test_data[2]
    raw_image = Image.open(image_file)

    inputs = llava_process(text=prompt, images=raw_image, return_tensors="pt")


    # build_qaimage(llava_process, test_data[0], test_data[1], test_data[2])
