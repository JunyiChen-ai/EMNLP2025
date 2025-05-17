import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class AttentionPooling(nn.Module):
    """
    Attention pooling layer for weighted aggregation of token representations
    """
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, token_embeddings, attention_mask):
        # token_embeddings: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        
        # Compute attention weights
        attention_weights = self.attention(token_embeddings)
        
        # Expand attention_mask to match attention_weights dimensions
        extended_attention_mask = attention_mask.unsqueeze(-1)
        
        # Apply attention_mask to zero out weights at padding positions
        attention_weights = attention_weights * extended_attention_mask
        
        # Re-normalize weights
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum of token representations
        weighted_sum = torch.bmm(
            attention_weights.transpose(1, 2),  # [batch_size, 1, seq_len]
            token_embeddings                    # [batch_size, seq_len, hidden_size]
        )  # [batch_size, 1, hidden_size]
        
        return weighted_sum.squeeze(1)  # [batch_size, hidden_size]

class SelfGatingLayer(nn.Module):
    def __init__(self, input_features):
        super(SelfGatingLayer, self).__init__()
        # Create weights for the gating mechanism
        self.gate_weights = nn.Linear(input_features, input_features)
        # Initialize with small values to start with minimal gating
        nn.init.xavier_uniform_(self.gate_weights.weight, gain=0.1)
        nn.init.zeros_(self.gate_weights.bias)
        
    def forward(self, x):
        # Compute gating coefficients
        gates = torch.sigmoid(self.gate_weights(x))
        # Apply the gates to the input (element-wise multiplication)
        gated_output = gates * x
        return gated_output

class BertForFakeNewsDetection(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        dropout=0.1,
        num_features=768,
        num_classes=2,
        fp16=False,
        freeze_bert_layers=0,    # number of bottom BERT layers to freeze
        use_last_n_layers=4,     # number of last BERT layers to use for feature extraction
        use_attention_pooling=True  # whether to use attention pooling
    ):
        """
        BERT-based fake news detection model

        Args:
            model_name (str): BERT model name
            dropout (float): dropout probability
            num_features (int): feature dimension
            num_classes (int): number of classes
            fp16 (bool): whether to use half-precision floating point
            freeze_bert_layers (int): number of bottom BERT layers to freeze
            use_last_n_layers (int): number of last BERT layers to use for feature extraction
            use_attention_pooling (bool): whether to use attention pooling
        """
        super(BertForFakeNewsDetection, self).__init__()
        
        # Load pretrained BERT model
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(model_name, config=config)
        
        # Freeze the bottom N layers of BERT
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_bert_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        # Dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 1.5)  # higher dropout rate
        
        # Record the number of BERT layers to use
        self.use_last_n_layers = min(use_last_n_layers, 12)  # BERT-base has 12 layers
        bert_output_dim = 768 * self.use_last_n_layers  # dimension increases if using multiple layers
        
        # Whether to use attention pooling
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_pooling = AttentionPooling(768)
        
        # Feature aggregation and projection layer
        self.feature_aggregation = nn.Sequential(
            nn.Linear(bert_output_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_features)
        )
        self.self_gating = SelfGatingLayer(num_features)
        
        # Feature normalization layer
        self.layernorm = nn.LayerNorm(num_features)
        
        # Record parameters
        self.in_features = num_features
        self.fp16 = fp16
        
    def forward(self, input_ids, attention_mask=None, domain_ids=None):
        """
        Forward pass

        Args:
            input_ids (torch.Tensor): input sequence token ids
            attention_mask (torch.Tensor): attention mask

        Returns:
            torch.Tensor: output feature vector
        """
        # Use half-precision computation (if enabled) - compatible with older PyTorch versions
        with torch.cuda.amp.autocast(enabled=self.fp16):
            # Obtain outputs from all layers of the BERT model
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Extract outputs of the last N layers
            hidden_states = outputs.hidden_states
            
            # Determine feature extraction method based on settings
            if self.use_attention_pooling:
                # Apply attention pooling to the last layer
                last_hidden_state = outputs.last_hidden_state
                pooled_output = self.attention_pooling(last_hidden_state, attention_mask)
            else:
                # Use the [CLS] token's hidden state as text representation
                pooled_output = outputs.pooler_output
            
            # Apply dropout to prevent overfitting
            pooled_output = self.dropout1(pooled_output)
            
            # If using multiple layers, aggregate them
            if self.use_last_n_layers > 1:
                # Extract the [CLS] features from the last N layers
                layers_output = []
                for i in range(-self.use_last_n_layers, 0):
                    layer_output = hidden_states[i][:, 0]  # select [CLS] token
                    layers_output.append(layer_output)
                
                # Concatenate multi-layer features
                concat_features = torch.cat(layers_output, dim=1)
                
                # Feature aggregation and dimensionality reduction
                features = self.feature_aggregation(concat_features)
                features = self.self_gating(features)
            else:
                # Use only the last layer
                features = pooled_output
                features = self.self_gating(features)
            
            # Apply the second dropout
            features = self.dropout2(features)
            
            # Normalize features
            features = self.layernorm(features)
            
        return features
    
    def extract_features(self, input_ids, attention_mask=None, domain_ids=None):
        """
        Extract features without classification

        Args:
            input_ids (torch.Tensor): input sequence token ids
            attention_mask (torch.Tensor): attention mask

        Returns:
            torch.Tensor: feature vector
        """
        with torch.no_grad():
            features = self.forward(input_ids, attention_mask)
        return features
