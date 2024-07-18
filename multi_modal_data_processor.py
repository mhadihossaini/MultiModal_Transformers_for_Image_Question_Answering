import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import AutoTokenizer, AutoFeatureExtractor
from PIL import Image
import os
@dataclass
class MultimodalDataProcessor:
    text_tokenizer: AutoTokenizer
    image_preprocessor: AutoFeatureExtractor

    def process_text(self, text_data: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes the input text data.

        Args:
            text_data (List[str]): List of text strings to tokenize.

        Returns:
            Dict[str, torch.Tensor]: Tokenized text data including input IDs, token type IDs, and attention masks.
        """
        tokenized = self.text_tokenizer(
            text=text_data,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True
        )
        return {
            "input_ids": tokenized['input_ids'].squeeze(),
            "token_type_ids": tokenized['token_type_ids'].squeeze(),
            "attention_mask": tokenized['attention_mask'].squeeze()
        }

    def process_images(self, image_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocesses the input image data.

        Args:
            image_ids (List[str]): List of image file names to preprocess.

        Returns:
            Dict[str, torch.Tensor]: Preprocessed image data including pixel values.
        """
        images = [Image.open(os.path.join("dataset", "images", img_id + ".png")).convert('RGB') for img_id in image_ids]
        processed = self.image_preprocessor(images=images, return_tensors="pt")
        return {
            "pixel_values": processed['pixel_values'].squeeze()
        }

    def __call__(self, batch_data: Union[Dict, List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of data by tokenizing text and preprocessing images.

        Args:
            batch_data (Union[Dict, List[Dict]]): A batch of raw data containing text and image IDs.

        Returns:
            Dict[str, torch.Tensor]: Processed batch including tokenized text, preprocessed images, and labels.
        """
        if isinstance(batch_data, dict):
            text_data = batch_data['question']
            image_data = batch_data['image_id']
            labels = batch_data['label']
        else:
            text_data = [item['question'] for item in batch_data]
            image_data = [item['image_id'] for item in batch_data]
            labels = [item['label'] for item in batch_data]
        
        return {
            **self.process_text(text_data),
            **self.process_images(image_data),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }