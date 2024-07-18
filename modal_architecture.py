import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Dict, List
from PIL import Image
import os
from transformers import AutoModel, AutoTokenizer
from multi_modal_data_processor import  MultimodalDataProcessor
class MultimodalVQA(nn.Module):
    def __init__(
        self, 
        answer_space: List[str],  # List of possible answers for classification.
        intermediate_dim: int = 512,  # Dimension of the intermediate fusion layer.
        text_model_name: str = 'roberta-base', 
        image_model_name: str = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
    ):
        super(MultimodalVQA, self).__init__()
        self.num_classes = len(answer_space)
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name

        # Load pretrained text and image models
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)
        self.image_encoder = AutoModel.from_pretrained(self.image_model_name)

        # Define multi-head self-attention for text featuress
        self.text_attention = nn.MultiheadAttention(embed_dim=self.text_encoder.config.hidden_size, num_heads=8)
        self.text_linear = nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)

        # Define multi-head self-attention for image features
        self.image_attention = nn.MultiheadAttention(embed_dim=self.image_encoder.config.hidden_size, num_heads=8)
        self.image_linear = nn.Linear(self.image_encoder.config.hidden_size, self.image_encoder.config.hidden_size)

        # Define fusion layer to combine text and image features
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Define classification head
        self.classification_head = nn.Linear(intermediate_dim, self.num_classes)
        
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        pixel_values: torch.FloatTensor, 
        attention_mask: Optional[torch.LongTensor] = None, 
        token_type_ids: Optional[torch.LongTensor] = None, 
        labels: Optional[torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode text data
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        text_features = text_outputs.last_hidden_state.permute(1, 0, 2)  # Prepare for multi-head attention
        text_attn_output, _ = self.text_attention(text_features, text_features, text_features)
        text_attn_output = self.text_linear(text_attn_output.permute(1, 0, 2).mean(dim=1))  # Mean pooling and linear layer

        # Encode image data
        image_outputs = self.image_encoder(pixel_values=pixel_values, return_dict=True)
        image_features = image_outputs.last_hidden_state.permute(1, 0, 2)  # Prepare for multi-head attention
        image_attn_output, _ = self.image_attention(image_features, image_features, image_features)
        image_attn_output = self.image_linear(image_attn_output.permute(1, 0, 2).mean(dim=1))  # Mean pooling and linear layer

        # Concatenate text and image features and pass through the fusion layer
        combined_representation = self.fusion_layer(torch.cat((text_attn_output, image_attn_output), dim=1))
        
        # Get logits from the classification head
        logits = self.classification_head(combined_representation)
        
        output = {"logits": logits}
        if labels is not None:
            # Compute loss if labels are provided
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss
        
        return output