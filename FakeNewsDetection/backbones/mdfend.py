# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel

# class DomainFeatureExtractor(nn.Module):
#     """
#     领域特征提取器
#     """
#     def __init__(self, d_model, dropout=0.1):
#         super(DomainFeatureExtractor, self).__init__()
        
#         self.FFN_1 = nn.Linear(d_model, d_model)
#         self.FFN_2 = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text_feature):
#         """
#         参数:
#         - text_feature: [batch_size, d_model]
#         返回:
#         - domain_feature: [batch_size, d_model]
#         """
#         domain_feature = self.FFN_1(text_feature)
#         domain_feature = F.relu(domain_feature)
#         domain_feature = self.dropout(domain_feature)
#         domain_feature = self.FFN_2(domain_feature)
        
#         return domain_feature

# class AdaptiveGate(nn.Module):
#     """
#     自适应门控机制，根据文本特征和领域特征生成门控值
#     """
#     def __init__(self, d_model, dropout=0.1):
#         super(AdaptiveGate, self).__init__()
        
#         self.linear_1 = nn.Linear(d_model, d_model)
#         self.linear_2 = nn.Linear(d_model, d_model)
#         self.gate_linear = nn.Linear(d_model * 2, 1)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text_feature, domain_feature):
#         """
#         参数:
#         - text_feature: [batch_size, d_model]
#         - domain_feature: [batch_size, d_model]
#         返回:
#         - gate: [batch_size, 1]
#         """
#         text_feature = self.linear_1(text_feature)
#         text_feature = F.relu(text_feature)
#         text_feature = self.dropout(text_feature)
        
#         domain_feature = self.linear_2(domain_feature)
#         domain_feature = F.relu(domain_feature)
#         domain_feature = self.dropout(domain_feature)
        
#         # 拼接特征
#         concat_feature = torch.cat([text_feature, domain_feature], dim=1)
#         gate = torch.sigmoid(self.gate_linear(concat_feature))
        
#         return gate

# class FakeNewsClassifier(nn.Module):
#     """
#     虚假新闻分类器
#     """
#     def __init__(self, d_model, num_domains=1, dropout=0.1):
#         super(FakeNewsClassifier, self).__init__()
        
#         # 为每个领域创建一个分类器
#         self.classifiers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(d_model, 2)
#             ) for _ in range(num_domains)
#         ])
        
#     def forward(self, feature, domain_ids):
#         """
#         参数:
#         - feature: [batch_size, d_model]
#         - domain_ids: [batch_size]
#         返回:
#         - logits: [batch_size, 2]
#         """
#         batch_size = feature.size(0)
#         logits = torch.zeros(batch_size, 2).to(feature.device)
        
#         for i in range(batch_size):
#             domain_id = domain_ids[i].item()
#             logits[i] = self.classifiers[domain_id](feature[i].unsqueeze(0)).squeeze(0)
            
#         return logits

# class MDFEND(nn.Module):
#     """
#     MDFEND: 多领域虚假新闻检测模型
#     """
#     def __init__(self, bert_model, num_domains, d_model=768, dropout=0.1, freeze_bert_layers=0):
#         super(MDFEND, self).__init__()
        
#         # BERT用于文本特征提取
#         # self.bert = bert_model
#         self.bert = BertModel.from_pretrained(bert_model)
#         if freeze_bert_layers > 0:
#             for param in self.bert.embeddings.parameters():
#                 param.requires_grad = False
                
#             # 冻结所有BERT层
#             # for i in range(freeze_bert_layers):
#             #     for param in self.bert.encoder.layer[i].parameters():
#             #         param.requires_grad = False
#             for param in self.bert.encoder.parameters():
#                 param.requires_grad = False
#         self.d_model = d_model
#         self.num_domains = num_domains
        
#         # 文本特征提取
#         self.text_feature_extractor = nn.Sequential(
#             nn.Linear(768, d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # 领域特征提取器
#         self.domain_feature_extractors = nn.ModuleList([
#             DomainFeatureExtractor(d_model, dropout) for _ in range(num_domains)
#         ])
        
#         # 自适应门控
#         self.adaptive_gates = nn.ModuleList([
#             AdaptiveGate(d_model, dropout) for _ in range(num_domains)
#         ])
        
#         # 虚假新闻分类器
#         self.classifier = FakeNewsClassifier(d_model, num_domains, dropout)
        
#     def forward(self, input_ids, attention_mask, domain_ids):
#         """
#         前向传播
#         参数:
#         - input_ids: [batch_size, seq_len]
#         - attention_mask: [batch_size, seq_len]
#         - token_type_ids: [batch_size, seq_len]
#         - domain_ids: [batch_size]
#         返回:
#         - final_feature: [batch_size, d_model]
#         """
#         # BERT编码
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
        
#         # 获取[CLS]标记的表示作为文本特征
#         pooled_output = outputs.pooler_output  # [batch_size, d_model]
        
#         # 提取文本特征
#         text_feature = self.text_feature_extractor(pooled_output)  # [batch_size, d_model]
        
#         batch_size = input_ids.size(0)
#         final_feature = torch.zeros_like(text_feature)
        
#         for i in range(batch_size):
#             domain_id = domain_ids[i].item()
            
#             # 提取领域特征
#             domain_feature = self.domain_feature_extractors[domain_id](text_feature[i].unsqueeze(0))
            
#             # 计算自适应门控值
#             gate = self.adaptive_gates[domain_id](text_feature[i].unsqueeze(0), domain_feature)
            
#             # 融合特征
#             final_feature[i] = gate * domain_feature + (1 - gate) * text_feature[i].unsqueeze(0)
        
#         return final_feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class DomainFeatureExtractor(nn.Module):
    """
    Domain feature extractor
    """
    def __init__(self, d_model, dropout=0.1):
        super(DomainFeatureExtractor, self).__init__()
        
        self.FFN_1 = nn.Linear(d_model, d_model)
        self.FFN_2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_feature):
        """
        Args:
        - text_feature: [batch_size, d_model]
        Returns:
        - domain_feature: [batch_size, d_model]
        """
        domain_feature = self.FFN_1(text_feature)
        domain_feature = F.relu(domain_feature)
        domain_feature = self.dropout(domain_feature)
        domain_feature = self.FFN_2(domain_feature)
        
        return domain_feature

class AdaptiveGate(nn.Module):
    """
    Adaptive gating mechanism that generates gate values based on text and domain features
    """
    def __init__(self, d_model, dropout=0.1):
        super(AdaptiveGate, self).__init__()
        
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.gate_linear = nn.Linear(d_model * 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_feature, domain_feature):
        """
        Args:
        - text_feature: [batch_size, d_model]
        - domain_feature: [batch_size, d_model]
        Returns:
        - gate: [batch_size, 1]
        """
        text_feature = self.linear_1(text_feature)
        text_feature = F.relu(text_feature)
        text_feature = self.dropout(text_feature)
        
        domain_feature = self.linear_2(domain_feature)
        domain_feature = F.relu(domain_feature)
        domain_feature = self.dropout(domain_feature)
        
        # Concatenate features
        concat_feature = torch.cat([text_feature, domain_feature], dim=1)
        gate = torch.sigmoid(self.gate_linear(concat_feature))
        
        return gate

class FakeNewsClassifier(nn.Module):
    """
    Fake news classifier
    """
    def __init__(self, d_model, num_domains=1, dropout=0.1):
        super(FakeNewsClassifier, self).__init__()
        
        # Create a classifier for each domain
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2)
            ) for _ in range(num_domains)
        ])
        
    def forward(self, feature, domain_ids):
        """
        Args:
        - feature: [batch_size, d_model]
        - domain_ids: [batch_size]
        Returns:
        - logits: [batch_size, 2]
        """
        batch_size = feature.size(0)
        logits = torch.zeros(batch_size, 2).to(feature.device)
        
        for i in range(batch_size):
            domain_id = domain_ids[i].item()
            logits[i] = self.classifiers[domain_id](feature[i].unsqueeze(0)).squeeze(0)
            
        return logits

class MDFEND(nn.Module):
    """
    MDFEND: Multi-domain fake news detection model
    """
    def __init__(self, bert_model, num_domains, d_model=768, dropout=0.1, freeze_bert_layers=0):
        super(MDFEND, self).__init__()
        
        # BERT for text feature extraction
        # self.bert = bert_model
        self.bert = BertModel.from_pretrained(bert_model)
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
            # Freeze all BERT layers
            for param in self.bert.encoder.parameters():
                param.requires_grad = False
        
        self.d_model = d_model
        self.num_domains = num_domains
        
        # Text feature extraction
        self.text_feature_extractor = nn.Sequential(
            nn.Linear(768, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Domain feature extractors
        self.domain_feature_extractors = nn.ModuleList([
            DomainFeatureExtractor(d_model, dropout) for _ in range(num_domains)
        ])
        
        # Adaptive gates
        self.adaptive_gates = nn.ModuleList([
            AdaptiveGate(d_model, dropout) for _ in range(num_domains)
        ])
        
        # Fake news classifier
        self.classifier = FakeNewsClassifier(d_model, num_domains, dropout)
        
    def forward(self, input_ids, attention_mask, domain_ids):
        """
        Forward pass
        Args:
        - input_ids: [batch_size, seq_len]
        - attention_mask: [batch_size, seq_len]
        - domain_ids: [batch_size]
        Returns:
        - final_feature: [batch_size, d_model]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get [CLS] token representation as text feature
        pooled_output = outputs.pooler_output  # [batch_size, d_model]
        
        # Extract text features
        text_feature = self.text_feature_extractor(pooled_output)  # [batch_size, d_model]
        
        batch_size = input_ids.size(0)
        final_feature = torch.zeros_like(text_feature)
        
        for i in range(batch_size):
            domain_id = domain_ids[i].item()
            
            # Extract domain feature
            domain_feature = self.domain_feature_extractors[domain_id](text_feature[i].unsqueeze(0))
            
            # Compute adaptive gate value
            gate = self.adaptive_gates[domain_id](text_feature[i].unsqueeze(0), domain_feature)
            
            # Fuse features
            final_feature[i] = gate * domain_feature + (1 - gate) * text_feature[i].unsqueeze(0)
        
        return final_feature
