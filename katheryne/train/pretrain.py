from katheryne.light_modules.models.pretrain_model import PretrainLanguageModel
from katheryne.data.loader.pretrain import create_pretrain_dataset
from katheryne.train.train import train

def pretrain():
    train(create_pretrain_dataset, PretrainLanguageModel)

if __name__ == "__main__":
    pretrain()