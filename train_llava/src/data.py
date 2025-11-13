from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch
from typing import List
from transformers import AutoProcessor, LlavaProcessor


# 单个数据经过处理之后的数据结果类
@dataclass
class QaImagaOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor


# 基于Dataset构建LlavaDataset，用于处理单个训练数据
class LlavaDataset(Dataset):

    def __init__(self, data_dir) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(data_dir=data_dir)

    def build_dataset(self, data_dir) -> tuple[list, Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        images_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(path_or_buf=chat_file).to_dict("records")
        return chat_data, images_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        humen_input = cur_data["conversations"][0]["value"]
        gpt_output = cur_data["conversations"][1]["value"]
        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return (humen_input, gpt_output, image_path)


# 构建成QaImagaOutput输出结果
def build_qaimage(
    processor: LlavaProcessor, q_text: str, a_text: str, image_path: Path
) -> QaImagaOutput:

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

    inputs = processor(text=prompt, images=raw_image, return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text, return_tensors="pt", padding="longest", truncation=True
    )["input_ids"]

    return QaImagaOutput(
        q_input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        a_input_ids=a_input_ids,
    )


# 训练前的多batch整理
class TrainLlavaModelCollator:
    def __init__(self, processor: LlavaProcessor, IGNORE_INDEX: int = -100) -> dict:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX
        self.eos_token_ids = self.processor.tokenizer.eos_token_ids
        self.pad_token_ids = self.processor.tokenizer.pad_token_ids

    def convert_one_piece(self, q_input_ids: torch.Tensor, a_input_ids: torch.Tensor):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.eos_token_ids).reshape(1, -1),
            ],
            axis=1,
        )
        # labels 用来控制是否计算loss
        labels = torch.concat(
            [
                torch.full_like(q_input_ids, fill_value=self.ignore_index),
                a_input_ids,
                torch.tensor(self.eos_token_ids).reshape(1, -1),
            ],
            axis=1,
        )
        return input_ids, labels

    def __call__(self, features: List):
        """
        feature 为单个处理LlavaDataset处理后的结果
        """
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_input_len_list = []
        for feature in features:
            qaimage_output: QaImagaOutput = build_qaimage(
                self.processor, feature[0], feature[1], feature[2]
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])

            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values_list.append(qaimage_output.pixel_values)
        max_input_len = max(max_input_len_list)
        # 对齐token的长度
        input_ids = []
        for index, value in enumerate(input_ids_list):
            new_value = torch.concat(
                [
                    torch.full(
                        size=(1, max_input_len - max_input_len_list[index]),
                        fill_value=self.pad_token_ids,
                    ),
                    value,
                ],
                axis=1,
            )
            input_ids.append(new_value)
        labels = []
        for index, value in enumerate(labels_list):
            labels.append(
                torch.concat(
                    [
                        torch.full(
                            size=(1, max_input_len - max_input_len_list[index]),
                            fill_value=self.ignore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
            )
        input_ids = torch.concat(input_ids, axis=0)
        labels = torch.concat(labels, axis=0)
        pixel_values = torch.concat(pixel_values_list, axis=0)

        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.pad_token_ids] = 0  # 将填充的置为0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }
