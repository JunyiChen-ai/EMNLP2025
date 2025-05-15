# from easydict import EasyDict as edict
# import os

# def get_config():
#     """
#     获取配置信息
    
#     返回:
#         edict: 配置字典
#     """
#     cfg = edict()
    
#     # 基础配置
#     cfg.output = "./FakeNewsDetection/output"  # 输出路径
#     cfg.seed = 2023  # 随机种子，调整为更优值
#     cfg.fp16 = False   # 不使用FP16
    
#     # 数据相关配置
#     # cfg.data_path = "./processed_data/FineFake_preprocessed_endef_modified.pkl"  # 数据路径
#     cfg.data_path = "./processed_data/FineFake_preprocessed_endef.pkl"  # 数据路径
#     # cfg.data_path = "./processed_data/Chinese_preprocessed_endef.pkl"  # 数据路径
#     # 确保数据路径存在，如果不存在，尝试其他可能的路径
#     if not os.path.exists(cfg.data_path):
#         alternative_paths = [
#             "/data/jehc223/EMNLP/processed_data/FineFake_preprocessed_endef.pkl",
#             "/data/jehc223/processed_data/FineFake_preprocessed_endef.pkl",
#             "./FineFake_preprocessed_endef.pkl"
#         ]
#         for path in alternative_paths:
#             if os.path.exists(path):
#                 cfg.data_path = path
#                 break
    
#     # 数据集划分配置
#     cfg.data_split = edict()
#     cfg.data_split.val_ratio = 0.2  # 验证集比例
#     cfg.data_split.test_ratio = 0.2  # 测试集比例
    
#     cfg.batch_size = 16  # 批量大小
#     cfg.max_length = 256  # 统一最大序列长度为256
#     cfg.num_workers = 1  # 减少数据加载线程数以避免问题
    
#     # 数据增强配置
#     cfg.augmentation = edict()
#     cfg.augmentation.enabled = False  # 是否启用数据增强
#     cfg.augmentation.p = 0.3  # 执行增强的概率
#     cfg.augmentation.augment_size = 0.3  # 增强样本比例
    
#     # 模型配置
#     # cfg.network = "bhadresh-savani/bert-base-go-emotion"  # 骨架网络
#     cfg.network = "bert-base-uncased"  # 骨架网络
#     # cfg.network = "mdfend"  # 骨架网络
#     cfg.num_classes = 3  # 类别数量
#     cfg.dropout = 0.2  # 增大Dropout率，提高正则化效果
#     cfg.embedding_size = 512  # 嵌入维度
    
#     # BERT特定配置
#     cfg.bert = edict()
#     cfg.bert.freeze_layers = 2  # 冻结BERT底部的2层
#     cfg.bert.use_last_n_layers = 3  # 使用BERT最后3层的特征
#     cfg.bert.use_attention_pooling = True  # 使用注意力池化
    
#     # 训练配置
#     cfg.num_epoch = 15  # 增加训练轮次
#     cfg.warmup_epoch = 2  # 预热轮次
#     cfg.frequent = 10  # 日志记录频率
#     cfg.lr = 2e-5  # 学习率
#     cfg.min_lr = 1e-6  # 最小学习率
#     cfg.weight_decay = 2e-4  # 增加权重衰减
#     cfg.momentum = 0.9  # 动量
#     cfg.scheduler = "cosine"  # 使用简单的余弦退火
#     cfg.scheduler_restarts = 2  # 重启次数
    
#     # 混合精度训练配置
#     cfg.mixed_precision = False  # 不使用混合精度训练
    
#     # 分类器配置
#     cfg.classifier = edict()
#     cfg.classifier.dropout_rate = 0.25  # 分类器的Dropout率
#     cfg.classifier.temperature = 0.1  # Softmax温度系数
    
#     # 早停技术配置
#     cfg.early_stopping = edict()
#     cfg.early_stopping.enabled = True  # 启用早停
#     cfg.early_stopping.patience = 3  # 容忍的轮次
#     cfg.early_stopping.min_delta = 0.001  # 最小改进阈值
    
#     # 样本权重配置
#     cfg.class_weights = edict()
#     cfg.class_weights.enabled = True  # 启用类别权重
#     cfg.class_weights.values = [1.0, 1.2, 1.0]  # 真新闻/假新闻/不确定的权重
    
#     # 标签平滑配置
#     cfg.label_smoothing = edict()
#     cfg.label_smoothing.enabled = True  # 是否启用标签平滑
#     cfg.label_smoothing.pos_value = 0.90  # 正样本的目标值
#     cfg.label_smoothing.neg_value = 0.10  # 负样本的目标值
    
#     # InvReg相关配置
#     cfg.invreg = edict()
#     cfg.invreg.env_num = 2  # 使用2个环境（简化模型）
#     cfg.invreg.env_num_lst = [2,2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # 环境数量列表
#     cfg.invreg.loss_weight_irm = 0.2  # InvReg损失权重
#     cfg.invreg.loss_weight_irm_anneal = False  # 使用退火策略
#     cfg.invreg.irm_train = "var"  # IRM训练模式
#     cfg.invreg.stage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 调整阶段转换轮次
#     cfg.invreg.use_mlp = False  # 是否使用MLP进行特征重映射
#     cfg.invreg.mlp_hidden_dim = 512  # MLP隐藏层维度
#     cfg.invreg.mlp_output_dim = 128  # MLP输出层维度
#     # 新增：是否累积历史分区
#     cfg.invreg.use_historical_partitions = True  # 是否使用并优化历史分区
#     # 新增：是否使用Elbow方法选择最优分区集
#     cfg.invreg.use_elbow_method = True  # 是否使用Elbow方法选择最优分区集
#     # 新增：是否保存所有分区信息
#     cfg.save_all_partitions = True  # 是否保存所有分区的元数据
    
#     # 断点续训配置
#     cfg.resume = False  # 是否从checkpoint恢复
#     cfg.pretrained = None  # 预训练模型路径
#     cfg.pretrained_ep = 0  # 预训练模型轮次
    
#     return cfg 
from easydict import EasyDict as edict
import os

def get_config():
    """
    Get configuration information
    
    Returns:
        edict: configuration dictionary
    """
    cfg = edict()
    
    # Base configuration
    cfg.output = "./FakeNewsDetection/output"  # output path
    cfg.seed = 2023  # random seed, adjust for optimal value
    cfg.fp16 = False   # do not use FP16
    
    # Data-related configuration
    # cfg.data_path = "./processed_data/FineFake_preprocessed_endef_modified.pkl"  # data path
    cfg.data_path = "./processed_data/FineFake_preprocessed_endef.pkl"  # data path
    # cfg.data_path = "./processed_data/Chinese_preprocessed_endef.pkl"  # data path
    # Ensure data path exists; if not, try other possible paths
    if not os.path.exists(cfg.data_path):
        alternative_paths = [
            "/data/jehc223/EMNLP/processed_data/FineFake_preprocessed_endef.pkl",
            "/data/jehc223/processed_data/FineFake_preprocessed_endef.pkl",
            "./FineFake_preprocessed_endef.pkl"
        ]
        for path in alternative_paths:
            if os.path.exists(path):
                cfg.data_path = path
                break
    
    # Dataset split configuration
    cfg.data_split = edict()
    cfg.data_split.val_ratio = 0.2  # validation set ratio
    cfg.data_split.test_ratio = 0.2  # test set ratio
    
    cfg.batch_size = 16  # batch size
    cfg.max_length = 256  # maximum sequence length
    cfg.num_workers = 1  # number of data loading workers
    
    # Data augmentation configuration
    cfg.augmentation = edict()
    cfg.augmentation.enabled = False  # enable data augmentation
    cfg.augmentation.p = 0.3  # probability of applying augmentation
    cfg.augmentation.augment_size = 0.3  # proportion of augmented samples
    
    # Model configuration
    # cfg.network = "bhadresh-savani/bert-base-go-emotion"  # backbone network
    cfg.network = "bert-base-uncased"  # backbone network
    # cfg.network = "mdfend"  # backbone network
    cfg.num_classes = 3  # number of classes
    cfg.dropout = 0.2  # dropout rate for regularization
    cfg.embedding_size = 512  # embedding dimension
    
    # BERT-specific configuration
    cfg.bert = edict()
    cfg.bert.freeze_layers = 2  # freeze bottom 2 layers of BERT
    cfg.bert.use_last_n_layers = 3  # use features from last 3 layers of BERT
    cfg.bert.use_attention_pooling = True  # use attention pooling
    
    # Training configuration
    cfg.num_epoch = 15  # number of training epochs
    cfg.warmup_epoch = 2  # number of warmup epochs
    cfg.frequent = 10  # logging frequency
    cfg.lr = 2e-5  # learning rate
    cfg.min_lr = 1e-6  # minimum learning rate
    cfg.weight_decay = 2e-4  # weight decay factor
    cfg.momentum = 0.9  # momentum for optimizer
    cfg.scheduler = "cosine"  # use cosine annealing scheduler
    cfg.scheduler_restarts = 2  # number of scheduler restarts
    
    # Mixed precision training configuration
    cfg.mixed_precision = False  # enable mixed precision training
    
    # Classifier configuration
    cfg.classifier = edict()
    cfg.classifier.dropout_rate = 0.25  # dropout rate for classifier
    cfg.classifier.temperature = 0.1  # softmax temperature coefficient
    
    # Early stopping configuration
    cfg.early_stopping = edict()
    cfg.early_stopping.enabled = True  # enable early stopping
    cfg.early_stopping.patience = 3  # number of tolerated epochs without improvement
    cfg.early_stopping.min_delta = 0.001  # minimum improvement threshold
    
    # Class weight configuration
    cfg.class_weights = edict()
    cfg.class_weights.enabled = True  # enable class weighting
    cfg.class_weights.values = [1.0, 1.2, 1.0]  # weights for real/fake/uncertain
    
    # Label smoothing configuration
    cfg.label_smoothing = edict()
    cfg.label_smoothing.enabled = True  # enable label smoothing
    cfg.label_smoothing.pos_value = 0.90  # target value for positive samples
    cfg.label_smoothing.neg_value = 0.10  # target value for negative samples
    
    # InvReg-related configuration
    cfg.invreg = edict()
    cfg.invreg.env_num = 2  # number of environments to use
    cfg.invreg.env_num_lst = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # list of environment counts
    cfg.invreg.loss_weight_irm = 0.2  # IRM loss weight
    cfg.invreg.loss_weight_irm_anneal = False  # use annealing strategy
    cfg.invreg.irm_train = "var"  # IRM training mode
    cfg.invreg.stage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # stages for phase transitions
    cfg.invreg.use_mlp = False  # use MLP for feature remapping
    cfg.invreg.mlp_hidden_dim = 512  # hidden dimension for MLP
    cfg.invreg.mlp_output_dim = 128  # output dimension for MLP
    # New: whether to accumulate historical partitions
    cfg.invreg.use_historical_partitions = True  # enable accumulation of historical partitions
    # New: whether to use the Elbow method to select optimal partition set
    cfg.invreg.use_elbow_method = True  # enable Elbow method for optimal partition selection
    # New: whether to save all partition metadata
    cfg.save_all_partitions = True  # save metadata for all partitions
    
    # Checkpoint resume configuration
    cfg.resume = False  # resume from checkpoint
    cfg.pretrained = None  # path to pretrained model
    cfg.pretrained_ep = 0  # epoch number of pretrained model
    
    return cfg
