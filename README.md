# [MCRec: Multimodal LLMs with Collaborative Filtering Alignments Enhanced Sequential Recommendation]
<!-- (https://arxiv.org/abs/2409.12740) -->

<div align="center">

<!-- [![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.12740-da282a.svg)](https://arxiv.org/abs/2409.12740)
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-ByteDance/HLLM-yellow)](https://huggingface.co/ByteDance/HLLM) -->
[![Recommendation](https://img.shields.io/badge/Task-Recommendation-blue)]()

</div>

<!-- ## 🔥 Update
- [2024.09.20] Codes and Weights are released ! -->


## Installation

1. Install packages via `pip3 install -r requirements.txt`. 
Some basic packages are shown below :
```
pytorch==2.1.0
deepspeed==0.14.2
transformers==4.41.1
lightning==2.4.0
flash-attn==2.5.9post1
fbgemm-gpu==0.5.0 [optional for HSTU]
sentencepiece==0.2.0 [optional for Baichuan2]
```
<!-- 2. Prepare `PixelRec` and `Amazon Book Reviews` Datasets:
    1. Download `PixelRec` Interactions and Item Information from [PixelRec](https://github.com/westlake-repl/PixelRec) and put into the dataset and information folder.
    2. Download `Amazon Book Reviews` [Interactions](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv) and [Item Information](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz), process them by `process_books.py`, and put into the dataset and information folder. We also provide [Interactions](https://huggingface.co/ByteDance/HLLM/resolve/main/Interactions/amazon_books.csv) and [Item Information](https://huggingface.co/ByteDance/HLLM/resolve/main/ItemInformation/amazon_books.csv) of Books after processing.
    3. Please note that Interactions and Item Information should be put into two folders like:
        ```bash
        ├── dataset # Store Interactions
        │   ├── amazon_books.csv
        │   ├── Pixel1M.csv
        │   ├── Pixel200K.csv
        │   └── Pixel8M.csv
        └── information # Store Item Information
            ├── amazon_books.csv
            ├── Pixel1M.csv
            ├── Pixel200K.csv
            └── Pixel8M.csv
        ``` 
        Here dataset represents **data_path**, and infomation represents **text_path**. -->
2. Prepare `PixelRec` dataset
3. Prepare pre-trained LLM models, such as [Qwen2VL](https://huggingface.co/Qwen/Qwen2-VL-2B),[TinyLlama](https://github.com/jzhang38/TinyLlama), 

## Training
To train MCRec on PixelRec Reviews, you can run the following command.

> Set `master_addr`, `master_port`, `nproc_per_node`, `nnodes` and `node_rank` in environment variables for multinodes training.

<!-- > All hyper-parameters (except model's config) can be found in code/REC/utils/argument_list.py and passed through CLI. More model's hyper-parameters are in `IDNet/*` or `HLLM/*`.  -->

```python
# Item and User LLM are initialized by specific pretrain_dir.
# train image + text
sh sh/user_2b_vl_item_2b_vl_user_visual.sh
# train text 
sh sh/user_2b_vl_item_2b.sh
# # Please remove tag in books dataset.
```
> You can use `--gradient_checkpointing True` and `--stage 3` with deepspeed to save memory.

You can also train ID-based models by the following command.
```python
python3 main.py \
--config_file overall/ID.yaml IDNet/{hstu / sasrec / llama_id}.yaml \
--loss nce \
--epochs 201 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 64 \
--MAX_ITEM_LIST_LENGTH 10 \
--optim_args.learning_rate 1e-4
```