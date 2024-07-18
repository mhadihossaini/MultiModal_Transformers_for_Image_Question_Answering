import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import AutoTokenizer, AutoFeatureExtractor
from PIL import Image
import os

# Define the MultimodalDataProcessor class
class data_process_for_test:
    def __init__(self, text_tokenizer, image_preprocessor):
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = image_preprocessor

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

    def process_images(self, image_paths: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocesses the input image data.

        Args:
            image_paths (List[str]): List of image file paths to preprocess.

        Returns:
            Dict[str, torch.Tensor]: Preprocessed image data including pixel values.
        """
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        processed = self.image_preprocessor(images=images, return_tensors="pt")
        return {
            "pixel_values": processed['pixel_values'].squeeze()
        }

    def __call__(self, text_data: List[str], image_paths: List[str]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of data by tokenizing text and preprocessing images.

        Args:
            text_data (List[str]): A list of text strings.
            image_paths (List[str]): A list of image file paths.

        Returns:
            Dict[str, torch.Tensor]: Processed batch including tokenized text and preprocessed images.
        """
        return {
            **self.process_text(text_data),
            **self.process_images(image_paths)
        }