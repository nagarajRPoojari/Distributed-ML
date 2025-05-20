from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse

from src.DistributedML.entity import ModelEvaluationConfig
from src.DistributedML.components.model.expert import UNet3DExpert
from src.DistributedML.components.model.router import MoELayer
from src.DistributedML.components.model_training import get_dataloaders
import matplotlib.pyplot as plt




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    
    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    
    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer, 
                               batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                               column_text="article", 
                               column_summary="highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
                            padding="max_length", return_tensors="pt")
            
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device), 
                            length_penalty=0.8, num_beams=8, max_length=128)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
            
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True) 
                for s in summaries]      
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
        score = metric.compute()
        return score


    def evaluate(self):
        _, val_loader = get_dataloaders()
        model = MoELayer(num_experts=4, input_dim=128*128*64, expert_class=UNet3DExpert, in_channels=1, out_channels=2)
        model.eval()

        with torch.no_grad():
            for images, _ in val_loader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                self.visualize_inference(images[0, 0].numpy(), preds[0].numpy(), slice_idx=32)
                break



    def visualize_inference(self, image, prediction, slice_idx):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image[slice_idx], cmap='gray')
        axes[0].set_title("Original Slice")
        axes[1].imshow(prediction[slice_idx], cmap='jet')
        axes[1].set_title("Predicted Segmentation")
        plt.show()
