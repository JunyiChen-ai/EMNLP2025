import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_acc(logits, label):
    """
    Calculate classification accuracy
    
    Args:
        logits (torch.Tensor): predicted scores
        label (torch.Tensor): ground truth labels
        
    Returns:
        float: accuracy
    """
    assert (logits.size()[0] == label.size()[0])
    with torch.no_grad():
        _, max_index = torch.max(logits, dim=1, keepdim=False)
        correct = (max_index == label).sum()
        return correct.item() / label.size(0)

class FakeNewsFC(nn.Module):
    """
    Classification layer and loss function for fake news detection
    Simple binary classifier implementation
    """
    def __init__(
        self,
        in_features,
        num_classes=2,
        use_invreg=True,
        dropout_rate=0.1,
        reduction='mean',
        class_weights=None,
        use_label_smoothing=False,
        pos_value=0.95,
        neg_value=0.05,
        temperature=0.1
    ):
        """
        Initialize

        Args:
            in_features (int): input feature dimension
            num_classes (int): number of classes
            use_invreg (bool): whether to use InvReg method
            dropout_rate (float): dropout rate
            reduction (str): loss reduction method, 'mean' or 'none'
            class_weights (torch.Tensor): class weights for imbalanced data
            use_label_smoothing (bool): whether to use label smoothing
            pos_value (float): target value for positive samples in label smoothing
            neg_value (float): target value for negative samples in label smoothing
            temperature (float): softmax temperature coefficient; lower values sharpen distribution
        """
        super(FakeNewsFC, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.use_invreg = use_invreg
        self.use_label_smoothing = use_label_smoothing
        self.pos_value = pos_value
        self.neg_value = neg_value
        self.temperature = temperature
        
        # Classification layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-layer classifier to enhance expressiveness
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features // 2, num_classes)
        )
        
        # Loss function
        if class_weights is not None:
            self.criteria = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        else:
            self.criteria = nn.CrossEntropyLoss(reduction=reduction)
        
    def forward(self, embeddings, label, return_logits=False):
        """
        Forward pass

        Args:
            embeddings (torch.Tensor): feature embeddings
            label (torch.Tensor): labels
            return_logits (bool): whether to return logits
            
        Returns:
            tuple: (loss, acc, [logits])
        """
        # Ensure embeddings and classifier weights have the same data type
        if embeddings.dtype != next(self.classifier.parameters()).dtype:
            embeddings = embeddings.to(next(self.classifier.parameters()).dtype)
        
        # Apply dropout
        x = self.dropout(embeddings)
        
        # Classification
        logits = self.classifier(x)
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Apply label smoothing
        if self.use_label_smoothing:
            # Create smoothed labels
            smooth_labels = torch.zeros_like(scaled_logits)
            smooth_labels[:, 0] = self.neg_value  # target value for negative samples
            smooth_labels[:, 1] = self.neg_value  # initialize all to negative sample target value
            
            # Set target value for positive samples based on true labels
            batch_size = label.size(0)
            for i in range(batch_size):
                smooth_labels[i, label[i]] = self.pos_value
            
            # Compute binary cross-entropy loss using scaled logits
            probs = torch.sigmoid(scaled_logits)
            loss = F.binary_cross_entropy(probs, smooth_labels, reduction='none')
            loss = loss.sum(dim=1)  # sum over classes for each sample
            
            # Apply class weights if provided
            if hasattr(self.criteria, 'weight') and self.criteria.weight is not None:
                weight = self.criteria.weight
                # Apply weights based on each sample's true label
                sample_weights = torch.tensor([weight[l.item()] for l in label], 
                                             device=loss.device)
                loss = loss * sample_weights
            
            # Apply reduction
            if self.criteria.reduction == 'mean':
                loss = loss.mean()
        else:
            # Use binary cross-entropy loss with scaled logits
            probs = torch.sigmoid(scaled_logits)
            target_one_hot = F.one_hot(label, num_classes=self.num_classes).float()
            loss = F.binary_cross_entropy(probs, target_one_hot, reduction=self.criteria.reduction)
        
        # Calculate accuracy (using original logits for consistent evaluation)
        acc = calc_acc(logits, label)
        
        if return_logits:
            return loss, acc, logits
        
        return loss, acc
        
class InvRegLoss(nn.Module):
    """Compute gradient variance loss across multiple environments"""
    
    def __init__(self, num_envs=2):
        """
        Initialize InvReg loss

        Args:
            num_envs (int): number of environments
        """
        super(InvRegLoss, self).__init__()
        self.num_envs = num_envs
        
    def forward(self, logits_list, labels_list, weights_list=None):
        """
        Compute gradient variance across environments for invariance regularization

        Args:
            logits_list (list): list of logits for each environment
            labels_list (list): list of labels for each environment
            weights_list (list, optional): list of sample weights for each environment for soft assignments
            
        Returns:
            torch.Tensor: gradient variance across environments
        """
        # At least two environments are required
        if len(logits_list) < 2:
            if self.training:
                raise ValueError("At least two environments are required to compute IRM loss")
            else:
                return torch.tensor(0.0, device=logits_list[0].device)
        
        # Check number of environments
        if len(logits_list) != len(labels_list):
            raise ValueError("Length of logits_list and labels_list must match")
        
        # Check weights list
        if weights_list is not None and len(weights_list) != len(logits_list):
            raise ValueError("weights_list must match length of logits_list")
        
        # Collect gradients and losses for all environments
        grads = []
        losses = []
        weighted_loss_values = []
        accuracies = []
        weight_sums = []
        
        # Compute loss separately for each environment
        for i, (logits, labels) in enumerate(zip(logits_list, labels_list)):
            # Compute cross-entropy loss for current environment
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss_tensor = criterion(logits, labels)
            
            # Compute accuracy for each environment
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean().item()
            accuracies.append(accuracy)
            
            # If weights are provided, apply weighted loss
            if weights_list is not None:
                weights = weights_list[i]
                
                # Ensure weights match shape of loss tensor
                if weights.shape != loss_tensor.shape:
                    print(f"Weight shape: {weights.shape}, loss shape: {loss_tensor.shape}")
                    weights = weights.reshape(loss_tensor.shape)
                
                # Normalize weights using softmax
                normalized_weights = F.softmax(weights, dim=0)
                
                # Compute and record sum of weights
                weight_sum = normalized_weights.sum().item()
                weight_sums.append(weight_sum)
                
                # Compute weighted loss
                weighted_loss = loss_tensor * normalized_weights
                loss = weighted_loss.sum()  # use sum of weighted loss as loss
                weighted_loss_values.append(weighted_loss.mean().item())
            else:
                # Use mean loss when no weights
                loss = loss_tensor.mean()
                weighted_loss_values.append(loss.item())
                weight_sums.append(1.0)
            
            losses.append(loss.item())
        
        # Debug information printing (commented out)
        # print("\n==== InvRegLoss Debug Info ====")
        # ...
        
        # Compute mean absolute deviation (MAD)
        losses_tensor = torch.tensor(losses)
        mean_loss = losses_tensor.mean()
        # Compute absolute deviation from mean for each loss and then take average
        mad = (losses_tensor - mean_loss).abs().mean()
        
        # Return mean absolute deviation of losses
        return mad
