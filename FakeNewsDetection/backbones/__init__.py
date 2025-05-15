# from .bert_model import BertForFakeNewsDetection
# from .mdfend import MDFEND
# def get_model(name='bert-base-uncased',  dropout=0.1, fp16=False, num_features=768, 
#               freeze_bert_layers=0, use_last_n_layers=1, use_attention_pooling=False,num_domains=1):
#     """
#     获取模型实例
    
#     参数:
#         name (str): 模型名称或预训练模型路径
#         dropout (float): Dropout率
#         fp16 (bool): 是否使用FP16
#         num_features (int): 特征维度
#         freeze_bert_layers (int): 冻结BERT底部的N层
#         use_last_n_layers (int): 使用BERT最后N层的特征
#         use_attention_pooling (bool): 是否使用注意力池化
        
#     返回:
#         torch.nn.Module: 模型实例
#     """
#     # 支持BERT系列模型
#     if 'bert' in name.lower():
#         return BertForFakeNewsDetection(
#             model_name=name,
#             dropout=dropout,
#             num_features=num_features,
#             num_classes=2,
#             fp16=fp16,
#             freeze_bert_layers=freeze_bert_layers,
#             use_last_n_layers=use_last_n_layers,
#             use_attention_pooling=use_attention_pooling
#         )
#     elif 'mdfend' in name.lower():
#         return MDFEND(
#             bert_model="bert-base-chinese",
#             num_domains=num_domains,
#             d_model=num_features,
#             dropout=dropout,
#             freeze_bert_layers=freeze_bert_layers
#         )
#     else:
#         raise ValueError(f"不支持的模型: {name}，目前仅支持BERT系列模型") 
from .bert_model import BertForFakeNewsDetection
from .mdfend import MDFEND

def get_model(name='bert-base-uncased', dropout=0.1, fp16=False, num_features=768, 
              freeze_bert_layers=0, use_last_n_layers=1, use_attention_pooling=False, num_domains=1):
    """
    Get model instance

    Args:
        name (str): model name or path to pretrained model
        dropout (float): dropout rate
        fp16 (bool): whether to use FP16
        num_features (int): feature dimension
        freeze_bert_layers (int): number of bottom BERT layers to freeze
        use_last_n_layers (int): number of last BERT layers to use for features
        use_attention_pooling (bool): whether to use attention pooling

    Returns:
        torch.nn.Module: model instance
    """
    # Support BERT-series models
    if 'bert' in name.lower():
        return BertForFakeNewsDetection(
            model_name=name,
            dropout=dropout,
            num_features=num_features,
            num_classes=2,
            fp16=fp16,
            freeze_bert_layers=freeze_bert_layers,
            use_last_n_layers=use_last_n_layers,
            use_attention_pooling=use_attention_pooling
        )
    elif 'mdfend' in name.lower():
        return MDFEND(
            bert_model="bert-base-chinese",
            num_domains=num_domains,
            d_model=num_features,
            dropout=dropout,
            freeze_bert_layers=freeze_bert_layers
        )
    else:
        raise ValueError(f"Unsupported model: {name}. Currently only BERT-series models are supported")
