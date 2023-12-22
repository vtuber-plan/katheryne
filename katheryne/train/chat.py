from katheryne.light_modules.models.chat_model import ChatLanguageModel
from katheryne.data.loader.chat import create_chat_dataset
from katheryne.train.train import train

def chat():
    train(create_chat_dataset, ChatLanguageModel)

if __name__ == "__main__":
    chat()