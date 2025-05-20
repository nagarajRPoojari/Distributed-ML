from src.DistributedML.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.DistributedML.pipeline.stage_02_data_validation import DataValidationPipeline
from src.DistributedML.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from src.DistributedML.logging import logger 


STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e



# STAGE_NAME = "Prepare base model"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    prepare_base_model = PrepareBaseModelPipeline()
#    prepare_base_model.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\")
# except Exception as e:
#         logger.exception(e)
#         raise e
    
    
STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainerPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e