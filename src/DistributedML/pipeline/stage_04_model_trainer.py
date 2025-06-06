from src.DistributedML.config.configuration import ConfigurationManager
from src.DistributedML.components.model_training import ModelTrainer
from src.DistributedML.logging import logger


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()