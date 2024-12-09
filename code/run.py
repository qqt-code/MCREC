# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9510))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Failed to attach debugger: {e}")
#     pass

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from cProfile import run
from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist

import os
import numpy as np
import argparse
import torch.distributed as dist
import torch
from peft import LoraConfig, TaskType, get_peft_model


def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[], gradient_checkpointing=False, stage=None):

    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        logger.info(f"Update text_path to {config['text_path']}")

    # get model and data
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"{len(train_loader) = }")

    print(config['model'])
    model = get_model(config['model'])(config, dataload)

    peft_config = LoraConfig(# task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                             )
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # # 打印当前显存占用
    # print(f"Current CUDA memory usage: {torch.cuda.memory_summary(device=None, abbreviated=False)}")

    world_size = torch.distributed.get_world_size()
    # 确保 DeepSpeed 配置被正确加载
    if 'deepspeed_config' in config:
        config['deepspeed_config']['train_batch_size'] = config['train_batch_size'] * world_size

    # 在创建模型之后，设置梯度检查点
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 如果指定了 stage，更新 DeepSpeed 配置
    if stage is not None:
        config['deepspeed_config']['zero_optimization']['stage'] = stage

    trainer = Trainer(config, model)

    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(dataload)
    logger.info(model)

    if config['val_only']:
        ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        logger.info(f'Eval only model load from {ckpt_path}')
        msg = trainer.model.load_state_dict(ckpt, False)
        logger.info(f'{msg.unexpected_keys = }')
        logger.info(f'{msg.missing_keys = }')
        test_result = trainer.evaluate(test_loader, load_best_model=False, show_progress=config['show_progress'], init_model=True)
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    else: # run this
        # training process
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')
        #wandb
        # model evaluation
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_metric_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Set ZeRO stage")
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK']) 
    config_file = args.config_file # LLM_deepspeed.yaml and HLLM_V.yaml

    # 设置CUDA设备
    torch.cuda.set_device(local_rank) 

    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://')

    # 打印调试信息
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 运行主循环 args.gradient_checkpointing = False,args.stage = None
    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args, 
             gradient_checkpointing=args.gradient_checkpointing, stage=args.stage) 

    # 清理进程组
    dist.destroy_process_group()
