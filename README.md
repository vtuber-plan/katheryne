# katheryne
Easy Language Model Trainer


## Let's benchmark GPUs LLM training with katheryne

### Training settings for 7B Llama (100 steps)
```
Stage: Pretrain
Model: meta-llama/Llama-2-7b-hf
Dataset: bigscience-data/roots_zh-cn_wikipedia
per_device_train_batch_size: 2
accumulate_grad_batches: 64
max_seq_len: 512
max_steps: 100
gradient_checkpointing: true
dtype: bf16
lora: {"r": 16, "target_modules": ["q_proj", "v_proj"]}
```

|              GPU             |  Time   |     Memory   |
|------------------------------|---------|--------------|
|    NVIDIA A800 80GB PCIe     |  00:41  |   14,850MiB  |
|   NVIDIA GeForce RTX 4090    |  01:02  |   15,078MiB  |
|      Iluvatar BI-V150        |  01:09  |   22,798MiB  |
|   NVIDIA GeForce RTX 3090    |  01:36  |   14,928MiB  |

