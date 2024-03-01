# katheryne
Easy Language Model Trainer


## Let's benchmark GPUs LLM training with katheryne

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

|     GPU     |  Time |     Memory   |
|-------------|-------|--------------|
|  RTX 3090   |       |   14,928MiB  |
|  Tesla A800 |       |              |