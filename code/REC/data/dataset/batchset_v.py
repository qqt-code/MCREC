# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from torch.utils.data import Dataset

import torch
import pandas as pd
from transformers import AutoTokenizer
import logging
from asyncio.log import logger
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import numpy as np
from qwen_vl_utils import process_vision_info
import os



class BatchTextDataset(Dataset):
    def __init__(self, config, dataload):
        self.item_num = dataload.item_num
        self.item_list = dataload.id2token['item_id']
        self.config = config
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        # self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(config['item_pretrain_dir'])
        # self.pad_id = self.tokenizer.pad_token_id
        # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.logger = logging.getLogger()
        self.load_content()

    def __len__(self):
        return self.item_num

    def load_content(self):
        self.env = pd.read_csv(self.text_path, delimiter=',', dtype={'item_id': str})
        self.env = self.env[self.text_keys + ['item_id']]
        self.env = self.env.set_index('item_id').T.to_dict()
        self.logger.info(f"Text Item num: {len(self.env)}")

    def __getitem__(self, index):
        def process_item(item):
            chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{% endif %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            
            if item != self.item_list[0] and item not in self.env:
                self.logger.info(f"{item} not in self.env")
            item_i = self.env.get(item, {})
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"\n{key}: {value}"

                # 使用dataset/cover/目录下的图片
                if len(item_i):
                    item_i['img'] = os.path.join("/home/yanruiran/workspace/lyc/HLLM/information", self.config['dataset'], f"{item}.jpg")
                    # 如果图片不存在, 则不使用图片
                    if not os.path.exists(item_i['img']):
                        item_i['img'] = None
                        #logger.info(f"{item_i['img']} not exists")
                if item_i['img']:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": item_i['img'],
                                    #"resized_height": 224, "resized_width": 224
                                    "resized_height": 28, "resized_width": 28
                                    #"resized_height": 56, "resized_width": 56
                                    #"resized_height": 84, "resized_width": 84
                                    #"resized_height": 112, "resized_width": 112
                                },
                                {"type": "text", "text": text_str},
                            ],
                        }
                    ]
                    image_inputs, video_inputs = process_vision_info(messages)
                    text_prompt = self.processor.apply_chat_template(messages, chat_template=chat_template, add_generation_prompt=False)
                    # print(text_prompt)

                    inputs = self.processor(
                        text=[text_prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        # padding=True,
                        # return_tensors="pt",
                        truncation=True,
                        max_length=self.max_text_length,
                    )
                    ids = inputs['input_ids'][0]
                    mask = inputs['attention_mask'][0]
                    pixel_values = inputs['pixel_values']
                    image_grid_thw = inputs['image_grid_thw']

                    return ids, mask, pixel_values, image_grid_thw #文本输入token id / 注意力掩码 / 图像的预处理张量 /图像的网格信息
                
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_str},
                            ],
                        }
                    ]

                    text_prompt = self.processor.apply_chat_template(messages, chat_template=chat_template, add_generation_prompt=False)
                    # print(text_prompt)

                    inputs = self.processor(
                        text=[text_prompt],
                        padding=True,
                        # return_tensors="pt",
                        truncation=True,
                        max_length=self.max_text_length,
                    )

                    ids = inputs['input_ids'][0]
                    mask = inputs['attention_mask'][0]

                    return ids, mask, None, None
                
            # 如果没有文本信息, 则直接返回空
            ids = self.processor.tokenizer.encode(text_str)
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)
            return ids, mask, None, None

        if index == 0 or index == self.item_num:
            item_token_i = ""
        else:
            item_token_i = self.item_list[index]
        pos_input_ids, pos_cu_input_lens, pos_position_ids, pos_pixel_values, pos_image_grid_thw = [], [], [], [], []
        ids, mask, pixel_values, image_grid_thw = process_item(item_token_i)
        pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
        pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
        pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
        if pixel_values is not None:
                pos_pixel_values.extend(pixel_values)
        if image_grid_thw is not None:
            pos_image_grid_thw.extend(image_grid_thw)
        outputs = {
            "pos_item_ids": torch.as_tensor(index, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
            "pos_pixel_values": torch.as_tensor(np.array(pos_pixel_values), dtype=torch.float32),
            "pos_image_grid_thw": torch.as_tensor(np.array(pos_image_grid_thw), dtype=torch.int64),
        }
        return outputs
