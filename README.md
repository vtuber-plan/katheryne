# katheryne
Easy Language Model Trainer


## Let's benchmark GPUs LLM training with katheryne

### Training settings - 7B Llama (100 steps)
```
Stage: Pretrain
Model: meta-llama/Llama-2-7b-hf
Dataset: bigscience-data/roots_zh-cn_wikipedia
per_device_train_batch_size: 2
accumulate_grad_batches: 1
max_seq_len: 512
max_steps: 100
gradient_checkpointing: true
dtype: bf16
lora: {"r": 16, "target_modules": ["q_proj", "v_proj"]}
```

|              GPU             |  Time   |   GPU Memory   |  Memory Usage  |
|------------------------------|---------|----------------|----------------|
|          NVIDIA L40S         |  00:38  |      48G       |    15,134MiB   |
|NVIDIA RTX 6000 ada Generation|  00:38  |      48G       |    15,134MiB   |
|    NVIDIA A800 80GB PCIe     |  00:41  |      80G       |    14,850MiB   |
|    NVIDIA A100 40GB PCIe     |  00:44  |      40G       |    14,863MiB   |
|   NVIDIA GeForce RTX 4090    |  01:02  |      24G       |    15,078MiB   |
|      Iluvatar BI-V150        |  01:09  |      32G       |    22,798MiB   |
|       NVIDIA RTX A6000       |  01:13  |      48G       |    14,944MiB   |
|         NVIDIA A40           |  01:16  |      48G       |    15,809MiB   |
|   NVIDIA GeForce RTX 3090    |  01:36  |      24G       |    14,928MiB   |


