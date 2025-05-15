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
