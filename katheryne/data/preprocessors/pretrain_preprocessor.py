import datasets
from katheryne.utils.datasets_info import DatasetInfo
from katheryne.utils.hparams import HParams


def pretrain_dataset_preprocess(
    hparams: HParams, dataset_info: DatasetInfo, dataset: datasets.IterableDataset
) -> datasets.IterableDataset:
    pass
