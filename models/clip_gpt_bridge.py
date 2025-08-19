from transformers import GPT2LMHeadModel
from torch import nn as nn
import torch

class ImageConditionedGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.image_proj = nn.Linear(512, config.n_embd)  # CLIP feature dim (512) to GPT embedding dim

    def forward(self, input_ids, image_features, labels=None, **kwargs):
        # Get token embeddings
        token_embeddings = self.transformer.wte(input_ids)
        # Project image features
        image_embedding = self.image_proj(image_features).unsqueeze(1)  # [batch, 1, n_embd]
        # Identify <image> token positions
        image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embeddings)
        # Replace <image> token embeddings with projected image features
        embeddings = torch.where(image_mask, image_embedding, token_embeddings)
        # Transformer forward pass
        transformer_outputs = self.transformer(inputs_embeds=embeddings, **kwargs)
        logits = self.lm_head(transformer_outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {'loss': loss, 'logits': logits}
    
