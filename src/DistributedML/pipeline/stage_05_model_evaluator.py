from src.DistributedML.config.configuration import ConfigurationManager
from src.DistributedML.components.model_evaluation import ModelEvaluation
from src.DistributedML.logging import logger




class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()