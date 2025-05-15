# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# def calc_acc(logits, label):
#     """
#     计算分类准确率
    
#     参数:
#         logits (torch.Tensor): 预测分数
#         label (torch.Tensor): 真实标签
        
#     返回:
#         float: 准确率
#     """
#     assert (logits.size()[0] == label.size()[0])
#     with torch.no_grad():
#         _, max_index = torch.max(logits, dim=1, keepdim=False)
#         correct = (max_index == label).sum()
#         return correct.item() / label.size(0)

# class FakeNewsFC(nn.Module):
#     """
#     假新闻检测的分类层与损失函数
#     简单的二分类器实现
#     """
#     def __init__(
#         self,
#         in_features,
#         num_classes=2,
#         use_invreg=True,
#         dropout_rate=0.1,
#         reduction='mean',
#         class_weights=None,
#         use_label_smoothing=False,
#         pos_value=0.95,
#         neg_value=0.05,
#         temperature=0.1
#     ):
#         """
#         初始化
        
#         参数:
#             in_features (int): 输入特征维度
#             num_classes (int): 类别数量
#             use_invreg (bool): 是否使用InvReg方法
#             dropout_rate (float): Dropout比率
#             reduction (str): 损失的规约方式，'mean'或'none'
#             class_weights (torch.Tensor): 类别权重，用于处理不平衡数据
#             use_label_smoothing (bool): 是否使用标签平滑
#             pos_value (float): 标签平滑中正样本的目标值
#             neg_value (float): 标签平滑中负样本的目标值
#             temperature (float): Softmax温度系数，较低的值使分布更加尖锐
#         """
#         super(FakeNewsFC, self).__init__()
#         self.in_features = in_features
#         self.num_classes = num_classes
#         self.use_invreg = use_invreg
#         self.use_label_smoothing = use_label_smoothing
#         self.pos_value = pos_value
#         self.neg_value = neg_value
#         self.temperature = temperature
        
#         # 分类层
#         self.dropout = nn.Dropout(dropout_rate)
        
#         # 多层分类器，增强表达能力
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features, in_features // 2),
#             nn.LayerNorm(in_features // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(in_features // 2, num_classes)
#         )
        
#         # 损失函数
#         if class_weights is not None:
#             self.criteria = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
#         else:
#             self.criteria = nn.CrossEntropyLoss(reduction=reduction)
        
#     def forward(self, embeddings, label, return_logits=False):
#         """
#         前向传播
        
#         参数:
#             embeddings (torch.Tensor): 特征嵌入
#             label (torch.Tensor): 标签
#             return_logits (bool): 是否返回logits
            
#         返回:
#             tuple: (loss, acc, [logits])
#         """
#         # 确保embeddings和分类器权重具有相同的数据类型
#         if embeddings.dtype != next(self.classifier.parameters()).dtype:
#             embeddings = embeddings.to(next(self.classifier.parameters()).dtype)
        
#         # 应用dropout
#         x = self.dropout(embeddings)
        
#         # 分类
#         logits = self.classifier(x)
        
#         # 应用温度系数
#         scaled_logits = logits / self.temperature
        
#         # 应用标签平滑
#         if self.use_label_smoothing:
#             # 创建平滑标签
#             smooth_labels = torch.zeros_like(scaled_logits)
#             smooth_labels[:, 0] = self.neg_value  # 负样本的目标值
#             smooth_labels[:, 1] = self.neg_value  # 初始化所有为负样本目标值
            
#             # 根据真实标签设置正样本的目标值
#             batch_size = label.size(0)
#             for i in range(batch_size):
#                 smooth_labels[i, label[i]] = self.pos_value
            
#             # 计算二元交叉熵损失，使用缩放后的logits
#             probs = torch.sigmoid(scaled_logits)
#             loss = F.binary_cross_entropy(probs, smooth_labels, reduction='none')
#             loss = loss.sum(dim=1)  # 对每个样本的所有类别求和
            
#             # 应用类别权重（如果有）
#             if hasattr(self.criteria, 'weight') and self.criteria.weight is not None:
#                 weight = self.criteria.weight
#                 # 根据每个样本的真实标签应用权重
#                 sample_weights = torch.tensor([weight[l.item()] for l in label], 
#                                              device=loss.device)
#                 loss = loss * sample_weights
            
#             # 应用reduction
#             if self.criteria.reduction == 'mean':
#                 loss = loss.mean()
#         else:
#             # 使用二元交叉熵损失，使用缩放后的logits
#             probs = torch.sigmoid(scaled_logits)
#             target_one_hot = F.one_hot(label, num_classes=self.num_classes).float()
#             loss = F.binary_cross_entropy(probs, target_one_hot, reduction=self.criteria.reduction)
        
#         # 计算准确率（使用原始logits以保持评估一致性）
#         acc = calc_acc(logits, label)
        
#         if return_logits:
#             return loss, acc, logits
        
#         return loss, acc
        
# class InvRegLoss(nn.Module):
#     """计算多个环境的梯度方差损失"""
    
#     def __init__(self, num_envs=2):
#         """
#         初始化InvReg损失
        
#         参数:
#             num_envs (int): 环境数量
#         """
#         super(InvRegLoss, self).__init__()
#         self.num_envs = num_envs
        
#     def forward(self, logits_list, labels_list, weights_list=None):
#         """
#         计算环境之间的梯度方差，用于实现不变性约束
        
#         参数:
#             logits_list (list): 各环境的logits列表
#             labels_list (list): 各环境的标签列表
#             weights_list (list, optional): 各环境的样本权重列表，用于软分配
            
#         返回:
#             torch.Tensor: 环境之间的梯度方差
#         """
#         # 至少需要两个环境
#         if len(logits_list) < 2:
#             if self.training:
#                 raise ValueError("至少需要两个环境才能计算IRM损失")
#             else:
#                 return torch.tensor(0.0, device=logits_list[0].device)
        
#         # 检查环境数量
#         if len(logits_list) != len(labels_list):
#             raise ValueError("logits列表和labels列表长度必须相同")
        
#         # 检查权重列表
#         if weights_list is not None and len(weights_list) != len(logits_list):
#             raise ValueError("weights_list必须与logits_list长度相同")
        
#         # 收集所有环境的梯度和损失
#         grads = []
#         losses = []
#         weighted_loss_values = []
#         accuracies = []
#         weight_sums = []
        
#         # 对每个环境分别计算损失
#         for i, (logits, labels) in enumerate(zip(logits_list, labels_list)):
#             # 计算当前环境的交叉熵损失
#             criterion = nn.CrossEntropyLoss(reduction='none')
#             loss_tensor = criterion(logits, labels)
            
#             # 计算每个环境的准确率
#             preds = torch.argmax(logits, dim=1)
#             accuracy = (preds == labels).float().mean().item()
#             accuracies.append(accuracy)
            
#             # 如果有权重，使用权重进行加权
#             if weights_list is not None:
#                 weights = weights_list[i]
                
#                 # 确保权重和损失形状匹配
#                 if weights.shape != loss_tensor.shape:
#                     print(f"权重形状: {weights.shape}, 损失形状: {loss_tensor.shape}")
#                     weights = weights.reshape(loss_tensor.shape)
                
#                 # 使用softmax对权重进行归一化
#                 normalized_weights = F.softmax(weights, dim=0)
                
#                 # 计算并记录权重总和
#                 weight_sum = normalized_weights.sum().item()
#                 weight_sums.append(weight_sum)
                
#                 # 计算加权损失
#                 weighted_loss = loss_tensor * normalized_weights
#                 loss = weighted_loss.sum()  # 直接使用加权和作为损失
#                 weighted_loss_values.append(weighted_loss.mean().item())
#             else:
#                 # 无权重时使用平均损失
#                 loss = loss_tensor.mean()
#                 weighted_loss_values.append(loss.item())
#                 weight_sums.append(1.0)
            
#             losses.append(loss.item())
            
#             # # 计算梯度
#             # grad = torch.autograd.grad(
#             #     loss, 
#             #     logits, 
#             #     create_graph=True, 
#             #     retain_graph=True
#             # )[0]
#             # 
#             # grads.append(grad)
        
#         # 打印调试信息
#         # print("\n==== InvRegLoss 调试信息 ====")
#         # print(f"环境数量: {len(logits_list)}")
#         # print(f"各环境准确率: {[f'{acc:.4f}' for acc in accuracies]}")
#         # print(f"准确率方差: {np.var(accuracies):.6f}")
#         # print(f"各环境损失值: {[f'{loss:.4f}' for loss in losses]}")
#         # print(f"损失方差: {np.var(losses):.6f}")
#         # print(f"各环境权重和: {[f'{w:.4f}' for w in weight_sums]}")
#         # print(f"加权损失均值: {[f'{w:.4f}' for w in weighted_loss_values]}")
        
#         # # 计算每个环境梯度的平方和
#         # grad_norms = [g.pow(2).mean().item() for g in grads]
#         # print(f"各环境梯度范数: {[f'{g:.6f}' for g in grad_norms]}")
#         # 
#         # # 计算梯度的方差作为IRM损失
#         # grad_var = torch.stack([g.pow(2).mean() for g in grads]).var()
#         # print(f"梯度方差(IRM损失): {grad_var.item():.6f}")
#         # print("============================\n")
        
#         # 计算平均绝对偏差(MAD)
#         losses_tensor = torch.tensor(losses)
#         mean_loss = losses_tensor.mean()
#         # 计算每个损失与平均值的绝对差，然后取平均
#         mad = (losses_tensor - mean_loss).abs().mean()
        
#         # 返回损失之间的平均绝对偏差
#         return mad
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
