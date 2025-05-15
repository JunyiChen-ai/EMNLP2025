import os
import sys

import argparse
import logging
import numpy as np
import time
import warnings

# Suppress all warning messages
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from easydict import EasyDict as edict
import json

from backbones import get_model
from data_utils.tensor_dataset import get_tensor_dataloader, PreprocessedTensorDataset, remove_empty_tensors
from data_utils.data_augmentation import AugmentedDataset
from fake_news_fc import FakeNewsFC, InvRegLoss
from utils.utils_logging import AverageMeter, init_logging, CallBackLogging, CallBackModelCheckpoint
from utils.partition_utils import update_partition, load_past_partition
from configs.invreg_config import get_config

# Set environment variables

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Arguments object
    """
    parser = argparse.ArgumentParser(description='InvReg for Fake News Detection')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--language', type=str, default='zh', help='Dataset language, en or zh')
    parser.add_argument('--local_rank', type=int, default=0, help='Local device rank')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate model, do not train')
    parser.add_argument('--checkpoint', type=str, default='FakeNewsDetection/output/best_model.pt', help='Checkpoint path to use for evaluation')
    parser.add_argument('--hyperparams', type=str, default=None, help='Path to hyperparameters JSON file to load')
    parser.add_argument('--no_mlp', action='store_true', help='Do not use MLP for feature remapping, use original features for environment partitioning')
    parser.add_argument('--data_path', type=str, default=None, help='Dataset path')
    
    args = parser.parse_args()
    return args

def set_seed(seed=2023):
    """
    Set random seed to ensure experiment reproducibility
    
    Parameters:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """Implementation of early stopping mechanism, monitoring validation performance"""
    
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        """
        Initialize early stopping
        
        Parameters:
            patience (int): Number of epochs to tolerate
            min_delta (float): Minimum improvement threshold
            mode (str): 'min' or 'max', depending on whether the monitored metric is better when smaller or larger
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        """
        Update early stopping status
        
        Parameters:
            score (float): Current score
            
        Returns:
            bool: Whether training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            delta = score - self.best_score
            if delta > self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            delta = self.best_score - score
            if delta > self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False

def evaluate(model, classifier, data_loader, device):
    """
    Evaluate model performance
    
    Parameters:
        model: Feature extraction model
        classifier: Classifier
        data_loader: Data loader
        device: Computing device
        
    Returns:
        dict: Evaluation results dictionary, including accuracy, precision, recall and F1 score (binary and macro average)
              as well as demographic parity metrics
    """
    model.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    # Lists for storing group information
    all_topics = []
    all_platforms = []
    all_authors = []
    
    # Check if dataset contains group information
    has_group_info = False
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            domain_ids = batch['domain_id'].to(device)
            
            # Check if group information is included
            if 'topic' in batch and 'platform' in batch and 'author' in batch:
                has_group_info = True
                all_topics.extend(batch['topic'])
                all_platforms.extend(batch['platform'])
                all_authors.extend(batch['author'])
            
            # Get features
            features = model(input_ids, attention_mask,domain_ids)
            
            # Use classifier to get logits
            _, _, logits = classifier(features, labels, return_logits=True)
            
            # Get prediction results
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # Binary metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    # Macro average F1
    _, _, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # Initialize return results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1
    }
    
    # If group information exists, calculate demographic parity metrics
    if has_group_info:
        # Convert to numpy arrays for better processing
        all_preds = np.array(all_preds)
        all_topics = np.array(all_topics)
        all_platforms = np.array(all_platforms)
        all_authors = np.array(all_authors)
        
        # Calculate the proportion of predicted positives for each group (for binary classification, the proportion predicted as 1)
        # Calculate maximum demographic disparity for Topics groups
        topic_demographic_parity = calculate_max_demographic_parity(all_preds, all_topics)
        
        # Calculate maximum demographic disparity for Platforms groups
        platform_demographic_parity = calculate_max_demographic_parity(all_preds, all_platforms)
        
        # Calculate maximum demographic disparity for Authors groups
        author_demographic_parity = calculate_max_demographic_parity(all_preds, all_authors)
        
        # Add to results dictionary
        results['topic_demographic_parity'] = topic_demographic_parity
        results['platform_demographic_parity'] = platform_demographic_parity
        results['author_demographic_parity'] = author_demographic_parity
        
        # Calculate maximum disparity among the three groups
        results['max_demographic_parity'] = max(topic_demographic_parity, platform_demographic_parity, author_demographic_parity)
    
    return results

def calculate_max_demographic_parity(predictions, group_labels):
    """
    Calculate the maximum demographic parity disparity between a set of groups
    
    Parameters:
        predictions: Model prediction results
        group_labels: Group labels that samples belong to
        
    Returns:
        float: Maximum demographic disparity (maximum difference in positive prediction rate between groups)
    """
    # Get unique group labels
    unique_groups = np.unique(group_labels)
    
    # If there's only one group, return 0
    if len(unique_groups) <= 1:
        return 0.0
    
    # Calculate positive prediction rate for each group
    group_positive_rates = []
    group_sizes = []  # Record sample size for each group
    
    for group in unique_groups:
        # Get indices for current group
        group_idx = (group_labels == group)
        
        # Count samples in current group
        group_size = np.sum(group_idx)
        
        # Skip if no samples
        if group_size == 0:
            continue
            
        # Calculate positive prediction rate for current group (for binary classification, proportion predicted as 1)
        group_predictions = predictions[group_idx]
        # For binary classification, directly calculate proportion predicted as 1
        positive_rate = np.mean(group_predictions == 1) if len(np.unique(predictions)) <= 2 else np.mean(group_predictions)
        
        group_positive_rates.append(positive_rate)
        group_sizes.append(group_size)
    
    # Calculate maximum disparity
    if len(group_positive_rates) <= 1:
        return 0.0
        
    max_disparity = max(group_positive_rates) - min(group_positive_rates)
    
    # Record groups with maximum and minimum positive rates
    max_idx = np.argmax(group_positive_rates)
    min_idx = np.argmin(group_positive_rates)
    max_group = unique_groups[max_idx] if max_idx < len(unique_groups) else "Unknown"
    min_group = unique_groups[min_idx] if min_idx < len(unique_groups) else "Unknown"
    
    logging.debug(f"Maximum disparity: {max_disparity:.4f}, Group with highest positive rate: {max_group} ({group_positive_rates[max_idx]:.4f}), Group with lowest positive rate: {min_group} ({group_positive_rates[min_idx]:.4f})")
    
    return max_disparity

def calculate_partition_loss(partition_data, local_embeddings, labels, classifier, device, sample_indices=None):
    """
    Calculate loss for a specific partition
    
    Parameters:
        partition_data: Partition data dictionary
        local_embeddings: Current batch feature embeddings
        labels: Current batch labels
        classifier: Classifier
        device: Computing device
        sample_indices: Current batch sample indices, used for sample_env type partitioning
    
    Returns:
        float: Partition loss value
    """
    partition_type = partition_data['type']
    partition_tensor = partition_data['data']
    
    # Add debug information: partition and sample index information
    # logging.info(f"Calculating partition loss - Partition type: {partition_type}, Partition name: {partition_data['name']}")
    # logging.info(f"Partition shape: {partition_tensor.shape}, Sample indices shape: {sample_indices.shape if sample_indices is not None else 'None'}")
    
    # Create Invariant Risk Minimization loss function
    invreg_loss = InvRegLoss(num_envs=partition_tensor.size(-1))
    
    # Collect logits, labels and weights for each environment
    logits_list = []
    labels_list = []
    weight_list = []
    
    # Add debug information: current batch sample index range
    # if sample_indices is not None:
    #     logging.info(f"Sample index range: {sample_indices.min().item()}-{sample_indices.max().item()}, Total: {len(sample_indices)}")
    # else:
    #     logging.error("Sample indices is None, cannot calculate partition loss")
    #     return 0.0
    
    # First calculate classification loss for all samples
    loss_ce, _, batch_logits = classifier(local_embeddings, labels, return_logits=True)
    
    if partition_type == 'sample_env' and sample_indices is not None:
        # Use sample-environment matrix (soft assignment)
        env_sample_weights = []  # Record total sample weights for each environment
        
        for env_idx in range(partition_tensor.size(1)):
            try:
                # Get sample weights for current environment
                env_weights = partition_tensor[:, env_idx]
                
               
                
                # Get corresponding environment weight assignments for current batch sample indices
                batch_env_weights = torch.zeros(labels.size(0), device=device)
                
                # Get valid indices
                valid_indices = 0
                invalid_indices = 0
                for i, global_idx in enumerate(sample_indices):
                    if global_idx < env_weights.size(0):  # Ensure index is within valid range
                        batch_env_weights[i] = env_weights[global_idx]
                        valid_indices += 1
                    else:
                        invalid_indices += 1
                
                # Add debug information: number of valid and invalid indices
                if invalid_indices > 0:
                    logging.warning(f"Environment {env_idx}: {invalid_indices} sample indices out of range, valid indices: {valid_indices}")
                
                # Calculate total sample weight for current environment
                env_weight_sum = batch_env_weights.sum().item()
                env_sample_weights.append(env_weight_sum)
                
                # Skip if current environment doesn't have enough sample weight
                if env_weight_sum < 0.1:  # Use a small threshold
                    logging.warning(f"Environment {env_idx}: Total sample weight in current batch too small ({env_weight_sum:.4f} < 0.1)")
                    continue
                
                # Count samples with non-zero weights (for logging only)
                significant_weights = (batch_env_weights > 0.05).float().sum().item()
                logging.warning(f"Environment {env_idx}: Samples with significant weight: {significant_weights}, Total weight: {env_weight_sum:.4f}")
                
                # Skip if not enough valid samples
                if significant_weights < 2:  # Need at least two samples
                    logging.warning(f"Environment {env_idx}: Not enough samples with significant weight ({significant_weights} < 2)")
                    continue
                
                # Use previously calculated batch_logits
                logits_list.append(batch_logits)
                labels_list.append(labels)
                weight_list.append(batch_env_weights)  # Use soft assignment weights
                
                
            except Exception as e:
                logging.error(f"Error calculating sample-environment partition {partition_data['name']} loss for environment {env_idx}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # Add debug information: total sample weights for all environments
        
    else:
        logging.error(f"Unsupported partition type {partition_type} or missing sample indices")
    
    # Calculate weighted inter-environment variance loss
    if len(logits_list) >= 2:
        # Modify InvRegLoss class call, pass weight list
        irm_loss = invreg_loss(logits_list, labels_list, weight_list).item()
        # Return sum of IRM loss and classification loss
        # Ensure loss_ce is a scalar not a tensor
        if isinstance(loss_ce, torch.Tensor) and loss_ce.numel() > 1:
            logging.warning(f"loss_ce is a tensor with {loss_ce.numel()} elements, taking mean to convert to scalar")
            return irm_loss + loss_ce.mean().item()
        else:
            return irm_loss + loss_ce.item()
    else:
        logging.warning(f"Not enough valid environments to calculate IRM loss, valid environments: {len(logits_list)}")
        # If not enough environments, return only classification loss
        return loss_ce.item().mean()

def select_optimal_partitions_elbow(partitions):
    """
    Use Elbow method to select optimal partition set
    
    Parameters:
        partitions: List of partitions, each containing a 'loss' key
        
    Returns:
        list: Selected optimal partition list
    """
    # If fewer than 3 partitions, return all partitions
    if len(partitions) < 3:
        return partitions
    
    # Sort partitions by loss in descending order
    sorted_partitions = sorted(partitions, key=lambda x: x['loss'], reverse=True)
    
    # Extract sorted losses
    losses = [p['loss'] for p in sorted_partitions]
    
    # If maximum loss is 0, means loss not calculated yet, return all partitions
    if losses[0] == 0:
        return partitions
    
    # Calculate normalized losses
    normalized_losses = [loss / losses[0] for loss in losses]
    
    # Calculate first-order differences
    first_diffs = [normalized_losses[i] - normalized_losses[i+1] for i in range(len(normalized_losses)-1)]
    
    # If fewer than 4 partitions, can't calculate second-order differences, use maximum first-order difference as elbow point
    if len(partitions) < 4:
        elbow_idx = first_diffs.index(max(first_diffs))
        return sorted_partitions[:elbow_idx+2]  # Include elbow point and previous partitions
    
    # Calculate second-order differences
    second_diffs = [first_diffs[i] - first_diffs[i+1] for i in range(len(first_diffs)-1)]
    
    # Find index with maximum second-order difference as elbow point
    elbow_idx = second_diffs.index(max(second_diffs))
    
    # Select partitions up to and including elbow point (including first partition after elbow point)
    selected_partitions = sorted_partitions[:elbow_idx+2]
    
    # Record selection process
    # logging.info(f"Elbow method partition selection - Total partitions: {len(partitions)}")
    # logging.info(f"Normalized losses: {[round(l, 4) for l in normalized_losses]}")
    # logging.info(f"First-order differences: {[round(d, 4) for d in first_diffs]}")
    # logging.info(f"Second-order differences: {[round(d, 4) for d in second_diffs]}")
    # logging.info(f"Selected elbow point: {elbow_idx}, Selected {len(selected_partitions)} partitions")
    # logging.info(f"Selected partitions: {[p['name'] for p in selected_partitions]}")
    
    return selected_partitions

def train(args, custom_cfg=None, is_hyperopt=False):
    """
    Main training function
    
    Parameters:
        args: Command line arguments
        custom_cfg: Custom configuration (for hyperparameter optimization)
        is_hyperopt: Whether in hyperparameter optimization mode
        
    Returns:
        float: If in hyperparameter optimization mode, return best validation accuracy and test accuracy
    """
    # Print directly to standard output to ensure visibility
    print("="*50)
    print("Training function starts execution")
    print(f"Command line arguments: {args}")
    print("="*50)
    
    # Get configuration
    if custom_cfg is not None:
        cfg = custom_cfg
    else:
        cfg = get_config()
        
        # If no_mlp command line argument is set, set cfg.invreg.use_mlp to False
        if hasattr(args, 'no_mlp') and args.no_mlp:
            cfg.invreg.use_mlp = False
        if args.data_path is not None:
            cfg.data_path = args.data_path
            
        # If hyperparameter file is specified, load it
        if args.hyperparams is not None and os.path.exists(args.hyperparams):
            sys.stdout.flush()  # Force flush output buffer
            logging.info(f"Loading hyperparameters from {args.hyperparams}...")
            with open(args.hyperparams, 'r') as f:
                hyperparams = json.load(f)
                
                # Check if there's a nested best_params field
                if "best_params" in hyperparams and isinstance(hyperparams["best_params"], dict):
                    hyperparams = hyperparams["best_params"]
                
                # Apply JSON parameters to configuration
                for key, value in hyperparams.items():
                    if key == "learning_rate" or key == "lr":
                        cfg.lr = value
                    elif key == "weight_decay":
                        cfg.weight_decay = value
                    elif key == "momentum":
                        cfg.momentum = value
                    elif key == "dropout":
                        cfg.dropout = value
                    elif key == "classifier_dropout":
                        cfg.classifier.dropout_rate = value
                    elif key == "freeze_bert_layers":
                        cfg.bert.freeze_layers = value
                    elif key == "use_last_n_layers":
                        cfg.bert.use_last_n_layers = value
                    elif key == "use_attention_pooling":
                        cfg.bert.use_attention_pooling = value
                    elif key == "embedding_size":
                        cfg.embedding_size = value
                    elif key == "batch_size":
                        cfg.batch_size = value
                    elif key == "num_epoch":
                        cfg.num_epoch = value
                    elif key == "warmup_epoch":
                        cfg.warmup_epoch = value
                    elif key == "temperature":
                        cfg.classifier.temperature = value
                    elif key == "use_label_smoothing":
                        cfg.label_smoothing.enabled = value
                    elif key == "label_smooth_pos":
                        cfg.label_smoothing.pos_value = value
                    elif key == "label_smooth_neg":
                        cfg.label_smoothing.neg_value = value
                    elif key == "use_invreg":
                        if not value:
                            cfg.invreg.loss_weight_irm = 0.0
                    elif key == "irm_weight":
                        cfg.invreg.loss_weight_irm = value
                    elif key == "irm_weight_anneal":
                        cfg.invreg.loss_weight_irm_anneal = value
                    elif key == "env_num":
                        cfg.invreg.env_num = value
                        # Update environment number list
                        cfg.invreg.env_num_lst = [value] * 10
                    elif key == "irm_train":
                        cfg.invreg.irm_train = value
                    elif key == "use_historical_partitions":
                        cfg.invreg.use_historical_partitions = value
                    elif key == "use_elbow_method":
                        cfg.invreg.use_elbow_method = value
                    elif key == "save_all_partitions":
                        cfg.save_all_partitions = value
                    elif key == "use_mlp":
                        cfg.invreg.use_mlp = value
                    elif key == "mlp_hidden_dim":
                        cfg.invreg.mlp_hidden_dim = value
                    elif key == "mlp_output_dim":
                        cfg.invreg.mlp_output_dim = value
                    elif key == "use_class_weights":
                        cfg.class_weights.enabled = value
                    elif key == "class_weight_ratio" and cfg.class_weights.enabled:
                        if cfg.num_classes == 2:
                            cfg.class_weights.values = [1.0, value]
                        elif cfg.num_classes == 3:
                            cfg.class_weights.values = [1.0, value, 1.0]
                    elif key == "scheduler":
                        cfg.scheduler = value
                    elif key == "scheduler_restarts":
                        cfg.scheduler_restarts = value
                    elif key == "early_stopping_patience":
                        cfg.early_stopping.patience = value
                    elif key == "early_stopping_min_delta":
                        cfg.early_stopping.min_delta = value
                    elif key == "use_augmentation":
                        cfg.augmentation.enabled = value
                    elif key == "augmentation_p" and cfg.augmentation.enabled:
                        cfg.augmentation.p = value
                    elif key == "augmentation_size" and cfg.augmentation.enabled:
                        cfg.augmentation.augment_size = value
                    elif key == "fp16":
                        cfg.fp16 = value
                        cfg.mixed_precision = value
                    elif key == "seed":
                        cfg.seed = value
                    elif key == "mixed_precision":
                        cfg.mixed_precision = value
                    elif key == "early_stopping_enabled":
                        cfg.early_stopping.enabled = value
                    elif key == "network":
                        cfg.network = value
                    elif key == "num_classes":
                        cfg.num_classes = value
                    elif key == "max_length":
                        cfg.max_length = value
                    elif key == "num_workers":
                        cfg.num_workers = value
                    elif key == "frequent":
                        cfg.frequent = value
                    elif key == "min_lr":
                        cfg.min_lr = value
                    elif key == "data_path":
                        cfg.data_path = value
                    elif key == "val_ratio":
                        cfg.data_split.val_ratio = value
                    elif key == "test_ratio":
                        cfg.data_split.test_ratio = value
                    elif key == "output":
                        cfg.output = value
            
            logging.info("Hyperparameter loading completed")
            
    # Apply command line argument settings
    if args.no_mlp:
        cfg.invreg.use_mlp = False
        logging.info("Set via command line parameter: Not using MLP for feature remapping")
    
    set_seed(cfg.seed)
    
    # Create output directory
    os.makedirs(cfg.output, exist_ok=True)
    
    # Initialize logging
    init_logging(cfg.output)
    
    # Force print log test message
    print("Testing logging system...")
    sys.stdout.flush()  # Force flush output buffer
    
    logging.info("="*50)
    logging.info("Logging system initialization test")
    logging.info("="*50)
    
    # Force flush log handlers
    for handler in logging.getLogger('').handlers:
        handler.flush()
    
    sys.stdout.flush()  # Force flush again
    
    # Initialize TensorBoard
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    
    # Set device
    # Use nvidia-smi command to find GPU with most free memory
    device = torch.device("cuda:3")
    # if torch.cuda.is_available():
    #     try:
    #         import subprocess
    #         import re
    #         # Execute nvidia-smi command to get GPU information
    #         result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']).decode('utf-8')
    #         # Parse results to get free memory for each GPU
    #         free_memory = [int(x) for x in result.strip().split('\n')]
    #         # Find GPU with most free memory
    #         best_gpu = free_memory.index(max(free_memory))
    #         logging.info(f"Automatically selected GPU with most free memory: {best_gpu}, Free memory: {max(free_memory)}MB")
    #         device = torch.device(f"cuda:{best_gpu}")
    #     except Exception as e:
    #         logging.warning(f"Failed to automatically select GPU: {e}, using specified local_rank: {args.local_rank}")
    #         device = torch.device(f"cuda:{args.local_rank}")
    # else:
    #     device = torch.device("cpu")
    #     logging.info("No available CUDA devices detected, using CPU")
    
    # Load tokenizer
    if args.language == 'en':
        tokenizer = BertTokenizer.from_pretrained(cfg.network)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
    
    
    # Load dataset and split into train/validation/test sets
    full_dataset = PreprocessedTensorDataset(
        data_path=cfg.data_path, 
        max_length=cfg.max_length,
        language=args.language
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * cfg.data_split.val_ratio)  # Use validation ratio from config
    test_size = int(dataset_size * cfg.data_split.test_ratio)  # Use test ratio from config
    train_size = dataset_size - val_size - test_size

    # Set fixed generator to ensure reproducibility
    generator = torch.Generator().manual_seed(cfg.seed)

    # Split dataset
    train_dataset_orig, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Remove empty tensor samples from each subset
    logging.info("Starting to check and remove empty tensor samples from each subset...")
    
    # Get original indices
    train_indices = train_dataset_orig.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    filtered_train_indices = remove_empty_tensors(full_dataset, train_indices, "Training set")
    filtered_val_indices = remove_empty_tensors(full_dataset, val_indices, "Validation set")
    filtered_test_indices = remove_empty_tensors(full_dataset, test_indices, "Test set")
    
    # Create new subsets
    train_dataset_orig = Subset(full_dataset, filtered_train_indices)
    val_dataset = Subset(full_dataset, filtered_val_indices)
    test_dataset = Subset(full_dataset, filtered_test_indices)
    
    # Update dataset sizes
    train_size = len(train_dataset_orig)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    logging.info(f"Dataset sizes after removing empty tensors: Training set {train_size} samples, Validation set {val_size} samples, Test set {test_size} samples")

    # Create a wrapper class to reset sample indices
    class IndexResetDataset(Dataset):
        """Dataset wrapper class to reset sample indices"""
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Reset sample index to local index
            item['sample_idx'] = torch.tensor(idx, dtype=torch.long)
            return item

    # Apply index reset
    train_dataset = IndexResetDataset(train_dataset_orig)
    logging.info(f"Applied sample index reset, ensuring indices in training set range from 0 to {len(train_dataset)-1}")

    logging.info(f"Dataset split: Training set {train_size} samples, Validation set {val_size} samples, Test set {len(test_dataset)} samples")
    
    # Apply data augmentation (if enabled)
    if cfg.augmentation.enabled:
        train_dataset = AugmentedDataset(
            original_dataset=train_dataset,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            p=cfg.augmentation.p,
            augment_size=cfg.augmentation.augment_size
        )
        logging.info(f"Training set size after data augmentation: {len(train_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size * 2,  # Can use larger batch size for validation
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size * 2,  # Can use larger batch size for testing
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Calculate training steps
    cfg.total_step = len(train_loader) * cfg.num_epoch
    cfg.warmup_step = len(train_loader) * cfg.warmup_epoch
    
    # Print configuration information
    for key, value in cfg.items():
        if isinstance(value, edict):
            logging.info(f"{key}:")
            for subkey, subvalue in value.items():
                num_space = 25 - len(subkey)
                logging.info(f"  - {subkey}{' ' * num_space}{subvalue}")
        else:
            num_space = 25 - len(key)
            logging.info(f"{key}{' ' * num_space}{value}")
    
    # Initialize model
    if args.language == 'en':
        with open("FakeNewsDetection/output/domain_dict.pkl", 'rb') as f:
            import pickle
            domain_dict = pickle.load(f)
    else:
        with open("FakeNewsDetection/output/domain_dict_ch.pkl", 'rb') as f:
            import pickle
            domain_dict = pickle.load(f)
    backbone = get_model(
        name=cfg.network,
        dropout=cfg.dropout,
        fp16=cfg.fp16,
        num_features=cfg.embedding_size,
        freeze_bert_layers=cfg.bert.freeze_layers,
        use_last_n_layers=cfg.bert.use_last_n_layers,
        use_attention_pooling=cfg.bert.use_attention_pooling,
        num_domains=len(domain_dict)
    ).to(device)
    
    # Calculate class weights (if enabled)
    if cfg.class_weights.enabled:
        class_weights = torch.tensor(cfg.class_weights.values, device=device)
        logging.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Initialize classifier
    classifier = FakeNewsFC(
        in_features=cfg.embedding_size,
        num_classes=cfg.num_classes,
        use_invreg=True if cfg.invreg.loss_weight_irm > 0 else False,
        dropout_rate=cfg.classifier.dropout_rate,
        reduction='none' if cfg.invreg.irm_train == 'var' else 'mean',
        class_weights=class_weights,
        use_label_smoothing=cfg.label_smoothing.enabled,
        pos_value=cfg.label_smoothing.pos_value,
        neg_value=cfg.label_smoothing.neg_value,
        temperature=cfg.classifier.temperature
    ).to(device)
    
    # Restore from checkpoint (if needed)
    start_epoch = 0
    global_step = 0
    if args.eval_only and args.checkpoint:
        checkpoint_path = args.checkpoint
        logging.info(f"Evaluation mode: Loading checkpoint {checkpoint_path}")
    elif cfg.resume and cfg.pretrained:
        checkpoint_path = os.path.join(cfg.pretrained, f"checkpoint_{cfg.pretrained_ep}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
            backbone.load_state_dict(checkpoint['backbone'])
            classifier.load_state_dict(checkpoint['classifier'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            logging.info(f"Resuming training from checkpoint, epoch={start_epoch}")
    
    # If only evaluating
    if args.eval_only:
        if args.checkpoint:
            # Load specified checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=False)
            backbone.load_state_dict(checkpoint['backbone'])
            classifier.load_state_dict(checkpoint['classifier'])
            
            # Evaluate on test set
            test_metrics = evaluate(backbone, classifier, test_loader, device)
            logging.info(f"Test set evaluation results: Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, Macro F1={test_metrics['macro_f1']:.4f}")
            logging.info(f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")
            
            # If demographic parity metrics exist, output them as well
            if 'max_demographic_parity' in test_metrics:
                logging.info(f"Test set demographic parity: Maximum disparity={test_metrics['max_demographic_parity']:.4f}, "
                            f"Topic disparity={test_metrics['topic_demographic_parity']:.4f}, "
                            f"Platform disparity={test_metrics['platform_demographic_parity']:.4f}, "
                            f"Author disparity={test_metrics['author_demographic_parity']:.4f}")
            
            # Save test results
            results_path = os.path.join(cfg.output, "test_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Test set accuracy: {test_metrics['accuracy']:.4f}\n")
                f.write(f"Test set F1 score: {test_metrics['f1']:.4f}\n")
                f.write(f"Test set macro F1 score: {test_metrics['macro_f1']:.4f}\n")
                f.write(f"Test set precision: {test_metrics['precision']:.4f}\n")
                f.write(f"Test set recall: {test_metrics['recall']:.4f}\n")
                
                # Add demographic parity metrics
                if 'max_demographic_parity' in test_metrics:
                    f.write(f"Test set maximum demographic disparity: {test_metrics['max_demographic_parity']:.4f}\n")
                    f.write(f"Test set topic demographic disparity: {test_metrics['topic_demographic_parity']:.4f}\n")
                    f.write(f"Test set platform demographic disparity: {test_metrics['platform_demographic_parity']:.4f}\n")
                    f.write(f"Test set author demographic disparity: {test_metrics['author_demographic_parity']:.4f}\n")
                
                f.write(f"Best model path: {args.checkpoint}\n")
                f.write(f"Best validation accuracy: {test_metrics['accuracy']:.4f}\n")
            
            return
        else:
            logging.error("Evaluation mode requires specifying checkpoint path")
            return
    
    # Initialize optimizer
    

    
    optimizer = torch.optim.AdamW(
        params=[
            {"params": backbone.parameters(), "lr": cfg.lr},
            {"params": classifier.parameters(), "lr": cfg.lr * 1.5}  # Higher learning rate for classifier
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Initialize learning rate scheduler
    if cfg.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_step,
            num_training_steps=cfg.total_step
        )
    elif cfg.scheduler == "cosine_with_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_step,
            num_training_steps=cfg.total_step,
            num_cycles=cfg.scheduler_restarts
        )
    else:
        scheduler = None
    
    # Restore optimizer and scheduler states from checkpoint (if needed)
    if cfg.resume and cfg.pretrained and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Initialize callbacks
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )
    
    callback_checkpoint = CallBackModelCheckpoint(
        output_dir=cfg.output,
        save_freq=1
    )
    
    # Initialize early stopping
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode='max'
        )
    
    # Initialize loss recorder
    loss_am = AverageMeter()
    
    # Initialize partition list
    updated_split_all = []
    
    # Save initial InvReg loss weight for annealing later
    loss_weight_irm_init = cfg.invreg.loss_weight_irm
    
    # Record best validation performance
    best_val_accuracy = 0.0
    best_checkpoint_path = None
    
    # Main training loop
    logging.info("Starting training...")
    for epoch in range(start_epoch, cfg.num_epoch):
        # Adjust InvReg loss weight (if annealing enabled)
        if cfg.invreg.loss_weight_irm_anneal and cfg.invreg.loss_weight_irm > 0:
            if epoch > cfg.warmup_epoch:
                cfg.invreg.loss_weight_irm = loss_weight_irm_init * (1 + 0.09) ** (epoch - cfg.warmup_epoch)
                logging.info(f"Adjusting InvReg loss weight to {cfg.invreg.loss_weight_irm:.4f}")
        
        # Check if environment partition needs to be updated
        if epoch in cfg.invreg.stage and cfg.invreg.loss_weight_irm > 0:
            # Update environment count
            cfg.invreg.env_num = cfg.invreg.env_num_lst[cfg.invreg.stage.index(epoch)]
            logging.info(f"Updating environment count to {cfg.invreg.env_num}")
            
            # Create feature save directory
            save_dir = os.path.join(cfg.output, 'saved_feat', f'epoch_{epoch}')
            os.makedirs(save_dir, exist_ok=True)
            
            # Store current partition list, for subsequent loss calculation
            previous_partitions = updated_split_all.copy() if hasattr(updated_split_all, 'copy') else list(updated_split_all)
            
            # Do not empty previous partitions, instead retain them
            # updated_split_all = []  # Comment this line, retain previous partitions
            
            # Create new partition
            logging.info('Starting to extract features for environment partitioning...')
            
            # Use tensor dataset feature extraction method
            features = []
            labels = []
            backbone.eval()
            
            # Add progress information
            logging.info(f"Starting to process {len(train_loader)} batch data feature extraction...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx % 10 == 0:
                        logging.info(f"Feature extraction progress: {batch_idx}/{len(train_loader)} batches")
                        
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    batch_labels = batch['label'].to(device)
                    sample_indices = batch['sample_idx'].to(device)
                    domain_ids = batch['domain_id'].to(device)
                    
                    batch_features = backbone(input_ids, attention_mask,domain_ids)
                    
                    features.append(batch_features.cpu())
                    labels.append(batch_labels.cpu())
                    
            logging.info(f"Feature extraction completed, extracted {len(features)} batch features")
            features = torch.cat(features, dim=0).numpy()
            labels = torch.cat(labels, dim=0).numpy()
            
            # Perform environment partitioning
            logging.info(f'Starting environment partitioning learning, environment count={cfg.invreg.env_num}')
            try:
                # Now only return sample-environment matrix
                sample_env_matrix = update_partition(
                    cfg=cfg,
                    save_dir=save_dir,
                    n_classes=cfg.num_classes,
                    features=features,
                    labels=labels,
                    writer=summary_writer,
                    device=device,
                    rank=args.local_rank,
                    world_size=1
                )
                
                # Add current cycle number as partition name prefix
                current_partition_prefix = f"epoch_{epoch}_"
                
                # Add new partition to list, and mark its source cycle
                updated_split_all.append({
                    'type': 'sample_env',
                    'data': torch.from_numpy(sample_env_matrix).float().to(device),
                    'epoch': epoch,
                    'name': f"{current_partition_prefix}sample_env",
                    'loss': 0.0  # Initialize loss as 0, will be updated later
                })
                logging.info(f"Successfully added sample-environment partition data, current cycle: {epoch}")
                
                # Add debugging information: partition size and relationship with dataset size
                partition_samples = sample_env_matrix.shape[0]
                logging.info(f"New partition '{current_partition_prefix}sample_env' sample count: {partition_samples}")
                
                # Check training loader sample index range
                max_idx = 0
                with torch.no_grad():
                    for test_batch in train_loader:
                        batch_indices = test_batch['sample_idx']
                        max_idx = max(max_idx, batch_indices.max().item())
                        if max_idx > partition_samples:
                            print(f"Sample index out of range: {max_idx}, Partition size: {partition_samples}")
                            break
                
                logging.info(f"Maximum sample index in training loader: {max_idx}, Partition size: {partition_samples}")
                if max_idx >= partition_samples:
                    logging.warning(f"Warning: Sample index ({max_idx}) in training loader exceeds partition size ({partition_samples})!")
                    logging.warning("This may result in insufficient valid samples when calculating IRM loss")
            except Exception as e:
                logging.error(f"Environment partitioning failed: {e}")
            
            del features, labels
            backbone.train()
            
            # Save all accumulated partitions
            if hasattr(cfg, 'save_all_partitions') and cfg.save_all_partitions:
                all_partitions_path = os.path.join(cfg.output, 'all_partitions.pt')
                partitions_to_save = [{
                    'type': p['type'],
                    'epoch': p['epoch'],
                    'name': p['name'],
                    'loss': p['loss'],
                    'data_shape': p['data'].shape
                } for p in updated_split_all]
                torch.save(partitions_to_save, all_partitions_path)
                logging.info(f"All partition metadata saved to {all_partitions_path}, {len(updated_split_all)} partitions")
        
        # One epoch training
        backbone.train()
        classifier.train()
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            
            # Get data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            sample_indices = batch['sample_idx'].to(device)  # Get sample global index
            domain_ids = batch['domain_id'].to(device)
            # Add debugging information: validation sample index
            if global_step % 20 == 0 or global_step < 10:  # Only record first few batches or every 20 batches
                # logging.info(f"Batch {batch_idx}, Global step {global_step}")
                # logging.info(f"Sample index shape: {sample_indices.shape}, Range: {sample_indices.min().item()}-{sample_indices.max().item()}")
                if len(updated_split_all) > 0:
                    first_partition = updated_split_all[0]
                    # logging.info(f"First partition '{first_partition['name']}' shape: {first_partition['data'].shape}")
                    # Check if index is within range
                    in_range = (sample_indices < first_partition['data'].shape[0]).float().mean().item()
                    # logging.info(f"Sample index proportion in partition range: {in_range:.2f}")
            
            # Forward propagation, get features
            local_embeddings = backbone(input_ids, attention_mask,domain_ids)
            
            # Calculate classification loss
            if cfg.invreg.irm_train == 'var':
                loss_ce_tensor, acc, logits = classifier(local_embeddings, labels, return_logits=True)
                loss_ce = loss_ce_tensor.mean()
                loss = loss_ce
            else:
                loss_ce, acc, logits = classifier(local_embeddings, labels, return_logits=True)
                loss = loss_ce
            
            # Calculate InvReg loss
            loss_irm = torch.tensor(0.0, device=device)
            if len(updated_split_all) > 0 and cfg.invreg.loss_weight_irm > 0:
                # Add debugging information: current batch sample index range
                # logging.info(f"Current batch sample index range: {sample_indices.min().item()}-{sample_indices.max().item()}, Total: {len(sample_indices)}")
                
                # Add debugging information: existing partition information
                # logging.info(f"Existing partition count: {len(updated_split_all)}, Partition names: {[p['name'] for p in updated_split_all]}")
                
                # Get Elbow method configuration parameters
                use_elbow_method = hasattr(cfg.invreg, 'use_elbow_method') and cfg.invreg.use_elbow_method
                
                # Calculate or update each partition loss
                for i, partition_data in enumerate(updated_split_all):
                    # Only calculate sample_env type partition loss or partition loss not yet calculated
                    if partition_data['loss'] == 0:
                        # Add debugging information: calculating loss for which partition
                        # logging.info(f"Calculating loss for partition {partition_data['name']}")
                        # # Add partition shape information
                        # logging.info(f"Partition shape: {partition_data['data'].shape}")
                        
                        current_loss = calculate_partition_loss(
                            partition_data=partition_data,
                            local_embeddings=local_embeddings,
                            labels=labels,
                            classifier=classifier,
                            device=device,
                            sample_indices=sample_indices
                        )
                        # Use moving average to update loss
                        if partition_data['loss'] == 0:
                            partition_data['loss'] = current_loss
                        else:
                            # Use exponential moving average to update loss
                            alpha = 0.9  # Smoothing factor
                            partition_data['loss'] = alpha * partition_data['loss'] + (1 - alpha) * current_loss
                        
                        # Add debugging information: calculation result
                        logging.info(f"Partition {partition_data['name']} loss calculation result: {current_loss}")
                
                # Use Elbow method to select optimal partition set
                if use_elbow_method and len(updated_split_all) > 2:
                    # All partitions are sample_env type
                    # logging.info(f"Using Elbow method to select optimal partition set from {len(updated_split_all)} sample-environment partitions")
                    selected_partitions = select_optimal_partitions_elbow(updated_split_all)
                    # logging.info(f"Selected {len(selected_partitions)} partitions: {[p['name'] for p in selected_partitions]}")
                else:
                    # Do not use Elbow method, use all partitions
                    selected_partitions = updated_split_all
                
                # Calculate average InvReg loss for all selected partitions
                invreg_loss = InvRegLoss(num_envs=cfg.invreg.env_num)
                
                if cfg.invreg.irm_train == 'var':
                    # Based on variance IRM
                    all_partition_losses = []
                    
                    for partition_data in selected_partitions:
                        # Only calculate valid loss partitions
                        if partition_data['loss'] > 0:
                            partition_tensor = partition_data['data']
                            
                            logits_list = []
                            labels_list = []
                            weight_list = []
                            
                            # Use sample-environment matrix (soft allocation)
                            for env_idx in range(partition_tensor.size(1)):
                                try:
                                    # Get current environment sample weights
                                    env_weights = partition_tensor[:, env_idx]
                                    
                                    # Use current batch sample indices to get corresponding environment weight allocation
                                    batch_env_weights = torch.zeros(labels.size(0), device=device)
                                    for i, global_idx in enumerate(sample_indices):
                                        if global_idx < env_weights.size(0):  # Ensure index is within valid range
                                            batch_env_weights[i] = env_weights[global_idx]
                                        else:
                                            print(f"Sample index out of range: {global_idx}, Environment weight shape: {env_weights.shape}")
                                    
                                    # Calculate current environment sample weight sum
                                    env_weight_sum = batch_env_weights.sum().item()
                                    
                                    # If current environment does not have enough sample weights, skip
                                    if env_weight_sum < 0.1:  # Use a small threshold
                                        continue
                                    
                                    # Use classifier to get all samples' logits (will be weighted later)
                                    _, _, batch_logits = classifier(local_embeddings, labels, return_logits=True)
                                    
                                    logits_list.append(batch_logits)
                                    labels_list.append(labels)
                                    weight_list.append(batch_env_weights)  # Use soft allocation weights
                                    
                                except Exception as e:
                                    logging.debug(f"Calculating IRM loss for environment {env_idx} failed: {e}")
                                    continue
                            
                            # Calculate weighted environment variance
                            if len(logits_list) >= 2:
                                partition_loss = invreg_loss(logits_list, labels_list, weight_list)
                                all_partition_losses.append(partition_loss)
                    
                    # Calculate total all partitions loss
                    if len(all_partition_losses) > 0:
                        loss_irm = torch.mean(torch.stack(all_partition_losses))
                        
                        # Record used partition count
                        # logging.info(f"Calculated {len(all_partition_losses)} partition IRM loss, Total loss: {loss_irm.item():.4f}")
                    else:
                        logging.warning("No valid partitions to calculate IRM loss")
                
                # Add InvReg loss
                loss += loss_irm * cfg.invreg.loss_weight_irm
            
            # Record loss value
            loss_am.update(loss.item(), input_ids.size(0))
            epoch_loss += loss.item() * input_ids.size(0)
            epoch_samples += input_ids.size(0)
            
            # Backpropagation
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Record log
            if global_step % cfg.frequent == 0:
                lr = optimizer.param_groups[0]['lr']
                callback_logging(global_step, loss_am.avg, acc, lr, epoch)
                loss_am.reset()
                
                # Record InvReg loss
                if loss_irm.item() > 0:
                    summary_writer.add_scalar('loss_irm', loss_irm.item(), global_step)
        
        # Calculate epoch average loss
        avg_epoch_loss = epoch_loss / epoch_samples
        logging.info(f"Epoch {epoch+1}/{cfg.num_epoch} Average training loss: {avg_epoch_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = evaluate(backbone, classifier, val_loader, device)
        logging.info(f"Epoch {epoch+1}/{cfg.num_epoch} Validation set performance: Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, Macro average F1={val_metrics['macro_f1']:.4f}")
        
        # If there's demographic parity metrics, also output
        if 'max_demographic_parity' in val_metrics:
            logging.info(f"Validation set demographic parity: Maximum disparity={val_metrics['max_demographic_parity']:.4f}, "
                         f"Topic disparity={val_metrics['topic_demographic_parity']:.4f}, "
                         f"Platform disparity={val_metrics['platform_demographic_parity']:.4f}, "
                         f"Author disparity={val_metrics['author_demographic_parity']:.4f}")
        
        # Record to TensorBoard
        summary_writer.add_scalar('val/accuracy', val_metrics['accuracy'], global_step)
        summary_writer.add_scalar('val/f1', val_metrics['f1'], global_step)
        summary_writer.add_scalar('val/macro_f1', val_metrics['macro_f1'], global_step)
        summary_writer.add_scalar('val/precision', val_metrics['precision'], global_step)
        summary_writer.add_scalar('val/recall', val_metrics['recall'], global_step)
        
        # Add demographic parity metrics to TensorBoard
        if 'max_demographic_parity' in val_metrics:
            summary_writer.add_scalar('val/max_demographic_parity', val_metrics['max_demographic_parity'], global_step)
            summary_writer.add_scalar('val/topic_demographic_parity', val_metrics['topic_demographic_parity'], global_step)
            summary_writer.add_scalar('val/platform_demographic_parity', val_metrics['platform_demographic_parity'], global_step)
            summary_writer.add_scalar('val/author_demographic_parity', val_metrics['author_demographic_parity'], global_step)
        
        # Only save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            logging.info(f"New best model! Validation set accuracy: {best_val_accuracy:.4f}")
            
            # Save best model
            best_model_path = os.path.join(cfg.output, "best_model.pt")
            torch.save({
                'backbone': backbone.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'global_step': global_step,
                'val_metrics': val_metrics
            }, best_model_path)
            best_checkpoint_path = best_model_path
            logging.info(f"Best model saved to {best_model_path}")
        
        # Check if early stopping is needed
        if cfg.early_stopping.enabled and early_stopping(val_metrics['accuracy']):
            logging.info(f"Triggering early stopping condition, stopping training at {epoch+1} epoch")
            break
    
    logging.info("Training completed!")
    
    # Evaluate best model on test set
    if best_checkpoint_path:
        logging.info(f"Loading best model {best_checkpoint_path} for test...")
        # Use weights_only=False to solve PyTorch 2.6 loading model issue
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        backbone.load_state_dict(checkpoint['backbone'])
        classifier.load_state_dict(checkpoint['classifier'])
        
        test_metrics = evaluate(backbone, classifier, test_loader, device)
        logging.info(f"Test set evaluation results: Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, Macro average F1={test_metrics['macro_f1']:.4f}")
        logging.info(f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")
        
        # If there's demographic parity metrics, also output
        if 'max_demographic_parity' in test_metrics:
            logging.info(f"Test set demographic parity: Maximum disparity={test_metrics['max_demographic_parity']:.4f}, "
                        f"Topic disparity={test_metrics['topic_demographic_parity']:.4f}, "
                        f"Platform disparity={test_metrics['platform_demographic_parity']:.4f}, "
                        f"Author disparity={test_metrics['author_demographic_parity']:.4f}")
        
        # Save test results
        results_path = os.path.join(cfg.output, "test_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Test set accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Test set F1 score: {test_metrics['f1']:.4f}\n")
            f.write(f"Test set macro average F1 score: {test_metrics['macro_f1']:.4f}\n")
            f.write(f"Test set precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Test set recall: {test_metrics['recall']:.4f}\n")
            
            # Add demographic parity metrics
            if 'max_demographic_parity' in test_metrics:
                f.write(f"Test set maximum demographic disparity: {test_metrics['max_demographic_parity']:.4f}\n")
                f.write(f"Test set topic demographic disparity: {test_metrics['topic_demographic_parity']:.4f}\n")
                f.write(f"Test set platform demographic disparity: {test_metrics['platform_demographic_parity']:.4f}\n")
                f.write(f"Test set author demographic disparity: {test_metrics['author_demographic_parity']:.4f}\n")
                
            f.write(f"Best model path: {best_checkpoint_path}\n")
            f.write(f"Best validation set accuracy: {best_val_accuracy:.4f}\n")
    
    # If in hyperparameter optimization mode, return best validation set accuracy and test set accuracy
    if is_hyperopt:
        return best_val_accuracy, test_metrics['accuracy']

if __name__ == '__main__':
    # Parse parameters and start training
    print("Program execution started...")
    print("Parsing command line parameters...")
    args = parse_args()
    print(f"Parsing completed, parameters: {args}")
    
    # Set forced flush standard output
    sys.stdout.flush()
    
    try:
        train(args) 
    except Exception as e:
        print(f"Training process error: {e}")
        import traceback
        traceback.print_exc() 