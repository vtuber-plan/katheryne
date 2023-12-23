from katheryne.light_modules.models.reward_model import RewardLanguageModel
from katheryne.data.loader.reward import create_reward_dataset
from katheryne.train.train import train

def reward():
    train(create_reward_dataset, RewardLanguageModel)

if __name__ == "__main__":
    reward()