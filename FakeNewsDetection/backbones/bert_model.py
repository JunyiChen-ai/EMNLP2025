# import torch
# import torch.nn as nn
# from transformers import BertModel, BertConfig

# class AttentionPooling(nn.Module):
#     """
#     注意力池化层，用于加权聚合token表示
#     """
#     def __init__(self, hidden_size):
#         super(AttentionPooling, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1),
#             nn.Softmax(dim=1)
#         )
        
#     def forward(self, token_embeddings, attention_mask):
#         # token_embeddings: [batch_size, seq_len, hidden_size]
#         # attention_mask: [batch_size, seq_len]
        
#         # 计算注意力权重
#         attention_weights = self.attention(token_embeddings)
        
#         # 将attention_mask扩展为与attention_weights相同的维度
#         extended_attention_mask = attention_mask.unsqueeze(-1)
        
#         # 应用attention_mask，将padding位置的权重设为0
#         attention_weights = attention_weights * extended_attention_mask
        
#         # 重新归一化权重
#         attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
#         # 加权聚合token表示
#         weighted_sum = torch.bmm(
#             attention_weights.transpose(1, 2),  # [batch_size, 1, seq_len]
#             token_embeddings  # [batch_size, seq_len, hidden_size]
#         )  # [batch_size, 1, hidden_size]
        
#         return weighted_sum.squeeze(1)  # [batch_size, hidden_size]

# class SelfGatingLayer(nn.Module):
#     def __init__(self, input_features):
#         super(SelfGatingLayer, self).__init__()
#         # Create weights for the gating mechanism
#         self.gate_weights = nn.Linear(input_features, input_features)
#         # Initialize with small values to start with minimal gating
#         nn.init.xavier_uniform_(self.gate_weights.weight, gain=0.1)
#         nn.init.zeros_(self.gate_weights.bias)
        
#     def forward(self, x):
#         # Compute gating coefficients
#         gates = torch.sigmoid(self.gate_weights(x))
#         # Apply the gates to the input (element-wise multiplication)
#         gated_output = gates * x
#         return gated_output

# class BertForFakeNewsDetection(nn.Module):
#     def __init__(
#         self,
#         model_name="bert-base-uncased",
#         dropout=0.1,
#         num_features=768,
#         num_classes=2,
#         fp16=False,
#         freeze_bert_layers=0,  # 冻结BERT底部的N层
#         use_last_n_layers=4,   # 使用BERT的最后N层进行特征提取
#         use_attention_pooling=True  # 使用注意力池化
#     ):
#         """
#         基于BERT的假新闻检测模型
        
#         参数:
#             model_name (str): BERT模型名称
#             dropout (float): dropout概率
#             num_features (int): 特征维度
#             num_classes (int): 分类数量
#             fp16 (bool): 是否使用半精度浮点数
#             freeze_bert_layers (int): 冻结BERT底部的N层
#             use_last_n_layers (int): 使用BERT的最后N层进行特征提取
#             use_attention_pooling (bool): 是否使用注意力池化
#         """
#         super(BertForFakeNewsDetection, self).__init__()
        
#         # 加载预训练的BERT模型
#         config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
#         self.bert = BertModel.from_pretrained(model_name, config=config)
        
#         # 冻结BERT底部的N层
#         if freeze_bert_layers > 0:
#             for param in self.bert.embeddings.parameters():
#                 param.requires_grad = False
                
#             for i in range(freeze_bert_layers):
#                 for param in self.bert.encoder.layer[i].parameters():
#                     param.requires_grad = False
        
#         # 防止过拟合的dropout层
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout * 1.5)  # 更高的dropout率
        
#         # 记录要使用的BERT层数
#         self.use_last_n_layers = min(use_last_n_layers, 12)  # BERT base有12层
#         bert_output_dim = 768 * self.use_last_n_layers  # 如果使用多层，维度会增加
        
#         # 是否使用注意力池化
#         self.use_attention_pooling = use_attention_pooling
#         if use_attention_pooling:
#             self.attention_pooling = AttentionPooling(768)
        
#         # 特征聚合和投影层
#         self.feature_aggregation = nn.Sequential(
#             nn.Linear(bert_output_dim, 1024),
#             nn.LayerNorm(1024),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, num_features)
#         )
#         self.self_gating = SelfGatingLayer(num_features)
        
#         # 特征归一化层
#         self.layernorm = nn.LayerNorm(num_features)
        
#         # 记录参数
#         self.in_features = num_features
#         self.fp16 = fp16
        
#     def forward(self, input_ids, attention_mask=None,domain_ids=None):
#         """
#         前向传播
        
#         参数:
#             input_ids (torch.Tensor): 输入序列的token ids
#             attention_mask (torch.Tensor): 注意力掩码
            
#         返回:
#             torch.Tensor: 输出特征向量
#         """
#         # 使用半精度计算（如果启用）- 适配较旧版本的PyTorch
#         with torch.cuda.amp.autocast(enabled=self.fp16):
#             # 获取BERT模型的所有层输出
#             outputs = self.bert(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#             )
            
#             # 提取最后N层的输出
#             hidden_states = outputs.hidden_states
            
#             # 根据设置决定特征提取方式
#             if self.use_attention_pooling:
#                 # 对最后一层使用注意力池化
#                 last_hidden_state = outputs.last_hidden_state
#                 pooled_output = self.attention_pooling(last_hidden_state, attention_mask)
#             else:
#                 # 使用[CLS]标记的隐藏状态作为文本表示
#                 pooled_output = outputs.pooler_output
            
#             # 应用dropout防止过拟合
#             pooled_output = self.dropout1(pooled_output)
            
#             # 如果使用多层，需要将它们聚合
#             if self.use_last_n_layers > 1:
#                 # 提取最后N层的[CLS]特征
#                 layers_output = []
#                 for i in range(-self.use_last_n_layers, 0):
#                     layer_output = hidden_states[i][:, 0]  # 取[CLS]标记
#                     layers_output.append(layer_output)
                
#                 # 拼接多层特征
#                 concat_features = torch.cat(layers_output, dim=1)
                
#                 # 特征聚合与降维
#                 features = self.feature_aggregation(concat_features)
#                 features = self.self_gating(features)
#             else:
#                 # 仅使用最后一层
#                 features = pooled_output
#                 features = self.self_gating(features)
#             # 应用第二个dropout
#             features = self.dropout2(features)
            
#             # 特征归一化
#             features = self.layernorm(features)
            
#         return features
    
#     def extract_features(self, input_ids, attention_mask=None,domain_ids=None):
#         """
#         提取特征，不进行分类
        
#         参数:
#             input_ids (torch.Tensor): 输入序列的token ids
#             attention_mask (torch.Tensor): 注意力掩码
            
#         返回:
#             torch.Tensor: 特征向量
#         """
#         with torch.no_grad():
#             features = self.forward(input_ids, attention_mask)
#         return features 
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
