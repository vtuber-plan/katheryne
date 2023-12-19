

import platform


def setup_strategy_fsdp(hparams, world_size, rank, devices):
    from lightning.pytorch.strategies import FSDPStrategy
    strategy_params = hparams.get("strategy_params", {})
    if platform.system().lower() == 'windows' and \
            "process_group_backend" in strategy_params and \
            strategy_params["process_group_backend"] == "nccl":
        raise ValueError("Windows does not support nccl")
    fsdp = FSDPStrategy(**strategy_params)
    return fsdp

def setup_strategy_deepspeed(hparams, world_size, rank, devices):
    if world_size is None:
        ds_world_size = len(devices)
    else:
        ds_world_size = int(world_size)
    
    if "fp16" in hparams and hparams.fp16:
        ds_precision = "fp16"
    elif "bf16" in hparams and hparams.bf16:
        ds_precision = "bf16"
    else:
        ds_precision = "fp32"

    from lightning.pytorch.strategies import DeepSpeedStrategy
    from katheryne.utils.ds_utils import get_train_ds_config
    ds_config = get_train_ds_config(
        offload=hparams.offload,
        stage=hparams.zero_stage,
        precision=ds_precision
    )
    ds_config['train_micro_batch_size_per_gpu'] = hparams.per_device_train_batch_size
    ds_config['train_batch_size'] = hparams.per_device_train_batch_size * ds_world_size * hparams.accumulate_grad_batches
    ds = DeepSpeedStrategy(
        zero_optimization=True,
        stage=hparams.zero_stage,
        remote_device = hparams.get("remote_device", "cpu"),
        offload_optimizer = hparams.offload,
        offload_optimizer_device = 'cpu',
        offload_parameters = hparams.offload,
        cpu_checkpointing = hparams.offload,
        offload_params_device = "cpu",
        nvme_path=hparams.get("nvme_path", "./nvme_offload"),
        contiguous_memory_optimization=True,
        config=ds_config,
    )
    return ds

def setup_strategy_ddp(hparams, world_size, rank, devices):
    from lightning.pytorch.strategies import DDPStrategy
    strategy_params = hparams.get("strategy_params", {})
    if platform.system().lower() == 'windows' and \
            "process_group_backend" in strategy_params and \
            strategy_params["process_group_backend"] == "nccl":
        raise ValueError("Windows does not support nccl")
    ddp = DDPStrategy(**strategy_params)
    return ddp
