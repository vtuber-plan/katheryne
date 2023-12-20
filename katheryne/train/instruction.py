from katheryne.light_modules.models.instruction_model import InstructionLanguageModel
from katheryne.data.loader.instruction import create_instruction_dataset
from katheryne.train.train import train

def instruction():
    train(create_instruction_dataset, InstructionLanguageModel)

if __name__ == "__main__":
    instruction()