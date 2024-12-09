# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import sys
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

from REC.data.dataset import BatchTextDataset
from REC.data.dataset.collate_fn import customize_rmpad_collate
from torch.utils.data import DataLoader
from REC.evaluator import Evaluator, Collector
from REC.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    get_tensorboard, set_color, get_gpu_usage, WandbLogger
from REC.utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy
import wandb


class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()

        self.wandblogger = WandbLogger(config)

        self.optim_args = config['optim_args']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']

        self.rank = torch.distributed.get_rank()

        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger)

        self.checkpoint_dir = config['checkpoint_dir']
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)

        self.saved_model_name = '{}-{}.pth'.format(self.config['model'], 0)
        self.saved_model_file = os.path.join(self.checkpoint_dir, self.saved_model_name)

        self.use_text = config['use_text']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.update_interval = config['update_interval'] if config['update_interval'] else 20
        self.scheduler_config = config['scheduler_args']
        if config['freeze_prefix'] or config['freeze_ad']:
            freeze_prefix = config['freeze_prefix'] if config['freeze_prefix'] else []
            if config['freeze_ad']:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config['ft_item']:
                freeze_prefix.extend(['item_embedding'])

            self._freeze_params(freeze_prefix)

        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None

        # 只在主进程中初始化 wandb
        if self.rank == 0 and config.get('log_wandb', False):
            wandb.init(project=config.get('wandb_project', 'HLLM'), config=config)
            self.wandblogger = WandbLogger(config)

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None): #调整学习率，warm up作用是逐步增加学习率
        if self.scheduler_config['type'] == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config['type'] == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)

    def _build_optimizer(self):
        params = self.model.parameters()
        return optim.AdamW(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            pbar = tqdm(
                total=len(train_data),
                miniters=self.update_interval,
                desc=set_color(f"Train [{epoch_idx:>3}/{self.epochs:>3}]", 'pink'),
                file=sys.stdout
            )
        bwd_time = t.time()
        for batch_idx, data in enumerate(train_data):
            start_time = bwd_time
            self.optimizer.zero_grad()
            data = self.to_device(data)
            # for key in data:
            #     print(f"{key}: {data[key].shape}")
            # print("-"*50)
            data_time = t.time()
            losses = self.model(data)
            fwd_time = t.time()
            if self.config['loss'] and self.config['loss'] == 'nce':
                model_out = losses
                losses = model_out.pop('loss')
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            self.lite.backward(losses)
            grad_norm = self.optimizer.step()
            bwd_time = t.time()
            if self.scheduler_config:
                self.lr_scheduler.step()
            if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                msg = f"loss: {losses:.4f} data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f}s bwd: {bwd_time-fwd_time:.3f}s"
                if self.scheduler_config:
                    msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                if self.config['loss'] and self.config['loss'] == 'nce':
                    for k, v in model_out.items():
                        msg += f" {k}: {v:.3f}"
                if grad_norm:
                    msg = msg + f" grad_norm: {grad_norm.sum():.4f}"
                pbar.set_postfix_str(msg, refresh=False)
                pbar.update(self.update_interval)
                self.logger.info("\n" + "-"*50)
            if self.config['debug'] and batch_idx >= 10:
                break
            #break

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }

        self.lite.save(os.path.join(self.checkpoint_dir, self.saved_model_name), state=state)
        if self.rank == 0 and verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {self.saved_model_file}')

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learning_rate': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace('@', '_')
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def to_device(self, data):
        device = self.device
        if isinstance(data, tuple) or isinstance(data, list):
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device)
            return data
        else:
            return data.to(device)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.scheduler_config:
            warmup_rate = self.scheduler_config.get('warmup', 0.001)
            tot_steps = len(train_data) * self.epochs # len(train_data) = 4733
            warmup_steps = tot_steps * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)

        self.logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        self.logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
        self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        self.logger.info(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        world_size = dist.get_world_size()
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
        nnodes = world_size // local_world_size
        nproc_per_node = local_world_size

        self.logger.info(f"World size: {world_size}")
        self.logger.info(f"Local world size: {local_world_size}")
        self.logger.info(f"Number of nodes: {nnodes}")
        self.logger.info(f"Processes per node: {nproc_per_node}")

        precision = self.config['precision'] if self.config['precision'] else '32' #bf16-mixed
        if self.config['strategy'] == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy with {world_size=}, {local_world_size=}, {nnodes=}, {nproc_per_node=}")
            ds_config = self.config.get('deepspeed_config', {})
            ds_config['train_micro_batch_size_per_gpu'] = self.config['train_batch_size']
            strategy = DeepSpeedStrategy(config=ds_config)
            self.lite = L.Fabric(
                accelerator='gpu',
                strategy=strategy,
                devices=nproc_per_node,
                num_nodes=nnodes,
                precision=precision
            )
        else:
            self.logger.info(f"Use DDP strategy with {world_size=}, {local_world_size=}, {nnodes=}, {nproc_per_node=}")
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(
                accelerator='gpu',
                strategy=strategy,
                devices=nproc_per_node,
                num_nodes=nnodes,
                precision=precision
            )
        self.lite.launch()
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
        # self.model = HLLM_V self.optimizer = Adamw
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)

        if self.config['auto_resume']:
            raise NotImplementedError

        valid_step = 0
        #self.start_epoch = 0, self.epochs = 5
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            if self.config['need_training'] == None or self.config['need_training']:
                train_data.sampler.set_epoch(epoch_idx)
                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0: #self.eval_step = 1
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                    for name, value in valid_result.items():
                        self.tensorboard.add_scalar(name.replace('@', '_'), value, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                # 只在主进程中记录 wandb
                if self.rank == 0:
                    wandb.log({
                        "epoch": epoch_idx,
                        "valid_score": valid_score,
                        **valid_result
                    })

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                        (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        # 在 fit 方法的末尾
        # 记录最佳验证分数和结果, 用 best_valid关键字
        wandb.log({
            "best_valid_score": self.best_valid_score,
            "best_valid": self.best_valid_result
        })

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        user, time_seq, history_index, positive_u, positive_i = batched_data #user.shape = (256,10) time_seq.shape = (256,10,6) 
        interaction = self.to_device(user)
        time_seq = self.to_device(time_seq)
        if self.config['model'] == 'HLLM' or self.config['model'] == 'HLLM_V':
            if self.config['stage'] == 3:
                scores = self.model.module.predict(interaction, time_seq, self.item_feature)
            else:
                scores = self.model((interaction, time_seq, self.item_feature), mode='predict')
        else:
            scores = self.model.module.predict(interaction, time_seq, self.item_feature)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return scores, positive_u, positive_i

    @torch.no_grad()
    def compute_item_feature(self, config, data):
        if self.use_text: #run this
            item_data = BatchTextDataset(config, data)
            item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
            item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=14, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
            self.logger.info(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
            self.item_feature = []
            with torch.no_grad():
                for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):
                    items = self.to_device(items)
                    items = self.model(items, mode='compute_item')
                    self.item_feature.append(items)
                if isinstance(items, tuple):
                    self.item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
                else:
                    self.item_feature = torch.cat(self.item_feature)
                if self.config['stage'] == 3:
                    self.item_feature = self.item_feature.bfloat16()
        else:
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all()

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, init_model=False):
        if not eval_data:
            return
        if init_model:
            world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
            nnodes = world_size // local_world_size
            if self.config['strategy'] == 'deepspeed':
                self.logger.info(f"Use deepspeed strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DeepSpeedStrategy(stage=self.config['stage'], precision=precision)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
            else:
                self.logger.info(f"Use DDP strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DDPStrategy(find_unused_parameters=True)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model = self.lite.setup(self.model)

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            state = {"model": self.model}
            self.lite.load(checkpoint_file, state)
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        with torch.no_grad():
            self.model.eval()
            eval_func = self._full_sort_batch_eval

            self.tot_item_num = eval_data.dataset.dataload.item_num
            self.compute_item_feature(self.config, eval_data.dataset.dataload)
            iter_data = (
                tqdm(
                    eval_data,
                    total=len(eval_data),
                    ncols=150,
                    desc=set_color(f"Evaluate   ", 'pink'),
                    file=sys.stdout
                ) if show_progress and self.rank == 0 else eval_data
            )
            fwd_time = t.time()
            for batch_idx, batched_data in enumerate(iter_data):
                start_time = fwd_time
                data_time = t.time()
                scores, positive_u, positive_i = eval_func(batched_data)
                fwd_time = t.time()

                if show_progress and self.rank == 0:
                    iter_data.set_postfix_str(f"data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f}", refresh=False)
                self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)
            num_total_examples = len(eval_data.sampler.dataset)
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)

            metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
            for k, v in result.items():
                result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device), num_total_examples).cpu()
                result[k] = round(result_cpu.item(), metric_decimal_place)
            self.wandblogger.log_eval_metrics(result, head='eval')

            return result













