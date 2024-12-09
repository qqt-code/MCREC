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
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["OMP_NUM_THREADS"] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+')
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Set ZeRO stage")

    args, unknown_args = parser.parse_known_args()
    config_file = args.config_file

    if len(config_file) == 2:
        run_yaml = f"../TORCHRUN run.py --config_file {config_file[0]} {config_file[1]}"
    elif len(config_file) == 1:
        run_yaml = f"../TORCHRUN run.py --config_file {config_file[0]}"

    if args.gradient_checkpointing:
        run_yaml += " --gradient_checkpointing"
    
    if args.stage is not None:
        run_yaml += f" --stage {args.stage}"

    run_yaml += f" {' '.join(unknown_args)}"

    # print(run_yaml)
    os.system(run_yaml)
