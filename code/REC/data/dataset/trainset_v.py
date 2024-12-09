# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from asyncio.log import logger
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import random
import datetime
import pytz
import math
import torch.distributed as dist
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,AutoImageProcessor
from qwen_vl_utils import process_vision_info
import os


# 数据形式为 [[user_seq], [neg_item_seq]] , [mask]


class SEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']

        self.length = len(self.train_seq)

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        self.num_negatives = config['num_negatives']
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        logger.info(f"Use random sample {self.random_sample} for mask id")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample(item_seq))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        return torch.as_tensor(item_seq, dtype=torch.int64), torch.as_tensor(neg_item, dtype=torch.int64), torch.as_tensor(masked_index, dtype=torch.int64)

    def __getitem__(self, index):
        # 最长长度为maxlen+1, 及若max_len是5
        # 则存在    1,2,3,4,5,6序列,
        # pos       2,3,4,5,6
        # neg       0,8,9,7,9,8
        # mask_index 1,1,1,1,1
        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)

        return item_seq, neg_item, masked_index


class TextSEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']
        self.length = len(self.train_seq)
        self.train_time_seq = dataload.train_feat['time_seq']
        self.id2token = dataload.id2token['item_id']

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        # self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(config['item_pretrain_dir'])
        #self.processor = AutoImageProcessor.from_pretrained(config['item_pretrain_dir'])
        # self.pad_id = self.tokenizer.pad_token_id
        # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.num_negatives = config['num_negatives']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])  # for llm only
        logger.info(f"Use random sample {self.random_sample} for mask id")
        logger.info(f"Text path: {self.text_path}")
        logger.info(f"Text keys: {self.text_keys}")
        logger.info(f"Item prompt: {self.item_prompt}")
        self.load_content()

    def __len__(self):
        return self.length

    def load_content(self):
        self.env = pd.read_csv(self.text_path, delimiter=',', dtype={'item_id': str})
        self.env = self.env[self.text_keys + ['item_id']]
        self.env = self.env.set_index('item_id').T.to_dict()
        logger.info(f"Text Item num: {len(self.env)}")

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample([]))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        return item_seq, neg_item, masked_index

    def _padding_time_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return torch.tensor(vq_time, dtype=torch.long)

    def __getitem__(self, index):

        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)
        time_seq = self.train_time_seq[index]
        time_seq = self._padding_time_sequence(list(time_seq), self.max_seq_length)
        item_seq_token = self.id2token[item_seq]
        neg_items_token = self.id2token[neg_item]
        pos_input_ids, pos_cu_input_lens, pos_position_ids, pos_pixel_values, pos_image_grid_thw = [], [], [], [], []
        neg_input_ids, neg_cu_input_lens, neg_position_ids, neg_pixel_values, neg_image_grid_thw = [], [], [], [], []

        
        def process_item(item):
            chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{% endif %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            
            if item != self.id2token[0] and item not in self.env:
                # assert item in self.env, f"{item}"
                logger.info(f"{item} not in self.env")
            item_i = self.env.get(item, {})
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"\n{key}: {value}"

            if len(item_i):#暂时不用图片
                item_i['img'] = os.path.join("/home/yanruiran/workspace/lyc/HLLM/information", self.config['dataset'], f"{item}.jpg")
                # 如果图片不存在, 则不使用图片
                # print("hahaha",item_i['img'])
                if not os.path.exists(item_i['img']):
                    item_i['img'] = None
                    # logger.info(f"{item_i['img']} not exists")
            text_str = ""
            if len(item_i):
                #text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"\n{key}: {value}"

                # 判断是否有图片
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
                    pixel_values = inputs['pixel_values'] #每一个都是[256,1176]
                    image_grid_thw = inputs['image_grid_thw']

                    return ids, mask, pixel_values, image_grid_thw
                
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
        

        for item in item_seq_token:
            ids, mask, pixel_values, image_grid_thw = process_item(item)
            pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
            pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
            if pixel_values is not None:
                pos_pixel_values.extend(pixel_values)
            if image_grid_thw is not None:
                pos_image_grid_thw.extend(image_grid_thw)  


        for neg in neg_items_token:
            ids, mask, pixel_values, image_grid_thw = process_item(neg)
            neg_input_ids.extend(ids + [0] * self.item_emb_token_n)
            neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            neg_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
            if pixel_values is not None:
                neg_pixel_values.extend(pixel_values)
            if image_grid_thw is not None:
                neg_image_grid_thw.extend(image_grid_thw)
        
        outputs = {
            "pos_item_ids": torch.as_tensor(item_seq, dtype=torch.int64),
            "neg_item_ids": torch.as_tensor(neg_item, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
            "pos_pixel_values": torch.as_tensor(np.array(pos_pixel_values), dtype=torch.float32),
            "pos_image_grid_thw": torch.as_tensor(np.array(pos_image_grid_thw), dtype=torch.int64),
            "neg_input_ids": torch.as_tensor(neg_input_ids, dtype=torch.int64),
            "neg_cu_input_lens": torch.as_tensor(neg_cu_input_lens, dtype=torch.int64),
            "neg_position_ids": torch.as_tensor(neg_position_ids, dtype=torch.int64),
            "neg_pixel_values": torch.as_tensor(np.array(neg_pixel_values), dtype=torch.float32),
            "neg_image_grid_thw": torch.as_tensor(np.array(neg_image_grid_thw), dtype=torch.int64),
            "attention_mask": torch.as_tensor(masked_index, dtype=torch.int64),
            "time_ids": torch.as_tensor(time_seq, dtype=torch.int64),
        }

        return outputs
