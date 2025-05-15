# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.data import Dataset, DataLoader
# import logging

# # 用于特征重映射的MLP网络
# class MLP(nn.Module):
#     def __init__(self, head='mlp', dim_in=512, feat_dim=128):
#         super(MLP, self).__init__()
#         if head == 'linear':
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(dim_in, dim_in),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(dim_in, feat_dim)
#             )

#     def forward(self, x):
#         x_norm = F.normalize(x.float(), dim=1)
#         mlp_x = self.head(x_norm)
#         return mlp_x

# # 环境分区网络 - 修改为样本级分区
# class Partition(nn.Module):
#     def __init__(self, n_cls, n_env, feat_dim=128):
#         super(Partition, self).__init__()
#         # 使用一个分类+环境分配网络，而不是简单的类别-环境矩阵
#         self.env_classifier = nn.Sequential(
#             nn.Linear(feat_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, n_env)
#         )
#         self.n_env = n_env
#         self.n_cls = n_cls
#         # 保留类别信息，用于类别级约束
#         self.class_env_probs = nn.Parameter(torch.randn((n_cls, n_env)))

#     def forward(self, features, labels=None):
#         # 根据特征预测环境分配
#         env_logits = self.env_classifier(features)
#         # 检查logits是否有NaN
#         if torch.isnan(env_logits).any():
#             print(f"检测到NaN in env_logits - 立即终止执行")
#             import sys
#             sys.exit(1)
            
#         sample_split = F.softmax(env_logits, dim=-1)
        
#         # 如果提供了标签，也返回类别级环境分配（用于约束）
#         if labels is not None:
#             class_split = F.softmax(self.class_env_probs[labels], dim=-1)
#             return sample_split, class_split
        
#         return sample_split

# # 计算熵
# def cal_entropy(x, dim=0):
#     return -torch.sum(x * torch.log(x + 1e-8), dim=dim)

# # 用于对比学习的损失计算
# def scl_logits(logits, logits_mask, mask, loss_weight=None, mode='scl', nonorm=False):
#     """
#     计算带权重的对比学习损失
    
#     参数:
#         logits: 相似度矩阵
#         logits_mask: 非自身掩码
#         mask: 同类样本掩码
#         loss_weight: 损失权重
#         mode: 损失模式
#         nonorm: 是否不进行归一化
#     """
#     # 数值稳定性处理
#     logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#     logits = logits - logits_max.detach()

#     # 计算exp(logits)
#     exp_logits = torch.exp(logits) * logits_mask
    
#     # 检查exp_logits是否有NaN或Inf
#     if torch.isnan(exp_logits).any() or torch.isinf(exp_logits).any():
#         print(f"检测到NaN/Inf in exp_logits")
#         print(f"exp_logits统计: min={exp_logits[~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)].min().item() if (~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)).any() else 'N/A'}, max={exp_logits[~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)].max().item() if (~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)).any() else 'N/A'}")
#         print(f"原始logits统计: min={logits.min().item()}, max={logits.max().item()}")
#         print(f"有多少NaN: {torch.isnan(exp_logits).sum().item()}, 有多少Inf: {torch.isinf(exp_logits).sum().item()}")
#         import sys
#         sys.exit(1)
    
#     # 如果没有提供权重，则所有样本权重相同
#     if loss_weight is None:
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#     else:
#         # 带权重的损失计算
#         weighted_exp_logits = exp_logits * loss_weight
        
#         # 检查weighted_exp_logits是否有NaN
#         if torch.isnan(weighted_exp_logits).any():
#             print(f"检测到NaN in weighted_exp_logits")
#             print(f"loss_weight统计: min={loss_weight[~torch.isnan(loss_weight)].min().item() if (~torch.isnan(loss_weight)).any() else 'N/A'}, max={loss_weight[~torch.isnan(loss_weight)].max().item() if (~torch.isnan(loss_weight)).any() else 'N/A'}")
#             print(f"有多少NaN in loss_weight: {torch.isnan(loss_weight).sum().item()}")
#             import sys
#             sys.exit(1)
            
#         log_prob = logits - torch.log(weighted_exp_logits.sum(1, keepdim=True) + 1e-8)
#         weighted_log_prob = mask * log_prob * loss_weight
        
#         if nonorm:
#             mean_log_prob_pos = weighted_log_prob.sum(1).mean()
#         else:
#             # 检查分母是否为0或NaN
#             denom = (mask * loss_weight).sum(1)
#             # if (denom == 0).any() or torch.isnan(denom).any():
#             #     print(f"检测到分母为0或NaN in scl_logits")
#             #     print(f"分母统计: min={denom[~torch.isnan(denom) & (denom != 0)].min().item() if (~torch.isnan(denom) & (denom != 0)).any() else 'N/A'}, 为0的个数: {(denom == 0).sum().item()}, NaN的个数: {torch.isnan(denom).sum().item()}")
#             #     import sys
#             #     sys.exit(1)
                
#             mean_log_prob_pos = weighted_log_prob.sum(1) / (denom + 1e-8)
#             if torch.isnan(mean_log_prob_pos).any():
#                 print(f"检测到NaN in mean_log_prob_pos")
#                 import sys
#                 sys.exit(1)

#     # 返回损失
#     return -mean_log_prob_pos.mean()

# # 计算对比学习损失中间结果
# def scl_loss_mid(feature, label, temperature=0.3):
#     """
#     计算对比学习损失的中间结果
    
#     参数:
#         feature: 特征向量
#         label: 标签
#         temperature: 温度参数
#     """
#     device = feature.device
#     batch_size = label.shape[0]
    
#     # 特征归一化
#     feature = F.normalize(feature, dim=1)
    
#     # 创建掩码
#     mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
#     # 排除自身
#     logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
#     mask = mask * logits_mask
    
#     # 计算相似度矩阵
#     logits = torch.div(torch.matmul(feature, feature.T), temperature)
    
#     # 索引序列
#     index_sequence = torch.arange(batch_size).to(device)
#     index_sequence = index_sequence.unsqueeze(0).expand(batch_size, batch_size)
    
#     return logits, logits_mask, mask, index_sequence

# # 计算软惩罚项
# def soft_penalty(logits, logits_mask, mask, loss_weight, mode='scl', nonorm=False, temp=1.0):
#     """
#     计算IRM的惩罚项
    
#     参数:
#         logits: 相似度矩阵
#         logits_mask: 非自身掩码
#         mask: 同类样本掩码
#         loss_weight: 损失权重
#         mode: 损失模式
#         nonorm: 是否不进行归一化
#         temp: 温度参数
#     """
#     # 数值稳定性处理
#     logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#     logits = logits - logits_max.detach()
    
#     # 带权重的损失计算    
#     exp_logits = torch.exp(logits) * logits_mask
    
#     # 检查exp_logits是否有NaN或Inf
#     if torch.isnan(exp_logits).any() or torch.isinf(exp_logits).any():
#         print(f"检测到NaN/Inf in exp_logits (soft_penalty)")
#         print(f"logits统计: min={logits.min().item()}, max={logits.max().item()}")
#         import sys
#         sys.exit(1)
    
#     weighted_exp_logits = exp_logits * loss_weight
    
#     # 检查weighted_exp_logits是否有NaN
#     if torch.isnan(weighted_exp_logits).any():
#         print(f"检测到NaN in weighted_exp_logits (soft_penalty)")
#         import sys
#         sys.exit(1)
    
#     # 计算梯度惩罚
#     denom = weighted_exp_logits.sum(1, keepdim=True)
    
#     # 检查分母是否为0或NaN
#     # if (denom == 0).any() or torch.isnan(denom).any():
#     #     print(f"检测到分母为0或NaN in soft_penalty")
#     #     print(f"分母统计: 为0的个数: {(denom == 0).sum().item()}, NaN的个数: {torch.isnan(denom).sum().item()}")
#     #     import sys
#     #     sys.exit(1)
    
#     weighted_prob = weighted_exp_logits / (denom + 1e-8)
#     weighted_log_prob = mask * torch.log(weighted_prob + 1e-8) * loss_weight
    
#     if nonorm:
#         grad_penalty = torch.pow(weighted_log_prob.sum(1).mean(), 2)
#     else:
#         norm_term = (mask * loss_weight).sum(1)
        
#         # 检查norm_term是否为0或NaN
#         # if (norm_term == 0).any() or torch.isnan(norm_term).any():
#         #     print(f"检测到norm_term为0或NaN in soft_penalty")
#         #     print(f"norm_term统计: 为0的个数: {(norm_term == 0).sum().item()}, NaN的个数: {torch.isnan(norm_term).sum().item()}")
#         #     import sys
#         #     sys.exit(1)
            
#         grad_penalty = torch.pow((weighted_log_prob.sum(1) / (norm_term + 1e-8)).mean(), 2)
#         if torch.isnan(grad_penalty).any():
#             print(f"检测到NaN in grad_penalty")
#             import sys
#             sys.exit(1)
    
#     return grad_penalty * temp

# # 数据集类
# class FeatureDataset(Dataset):
#     def __init__(self, features, labels):
#         """
#         初始化特征数据集
        
#         参数:
#             features: 特征矩阵
#             labels: 标签数组
#         """
#         self.features = torch.from_numpy(features).float()
#         self.labels = torch.from_numpy(labels).long()

#     def __getitem__(self, index):
#         return self.features[index], self.labels[index]

#     def __len__(self):
#         return self.features.shape[0]

# def update_partition(cfg, save_dir, n_classes, features, labels, writer=None, device=None, rank=0, world_size=1):
#     """
#     使用网络方法更新数据分区 - 修改为样本级分区
    
#     参数:
#         cfg: 配置对象
#         save_dir: 特征保存目录
#         n_classes: 类别数量
#         features: 特征矩阵
#         labels: 标签数组
#         writer: TensorBoard写入器
#         device: 计算设备
#         rank: 进程排名
#         world_size: 总进程数
        
#     返回:
#         numpy.ndarray: 更新后的分区
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     env_num = cfg.invreg.get("env_num", 2)
#     use_mlp = cfg.invreg.get("use_mlp", True)  # 获取是否使用MLP的配置
    
#     logging.info(f"使用网络方法划分数据到 {env_num} 个环境 (改进的样本级分区)")
#     print(f"使用网络方法划分数据到 {env_num} 个环境 (改进的样本级分区)")  # 直接打印确保可见
#     if use_mlp:
#         logging.info("将使用MLP对特征进行重映射")
#     else:
#         logging.info("将直接使用原始特征进行环境分区，不使用MLP重映射")
    
#     # 创建数据集和数据加载器
#     dataset = FeatureDataset(features, labels)
#     train_loader = DataLoader(
#         dataset=dataset,
#         batch_size=min(2048, len(dataset)),  # 使用较大的批量大小
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=False
#     )
    
#     # 获取MLP配置参数
#     mlp_hidden_dim = cfg.invreg.get("mlp_hidden_dim", 512)
#     mlp_output_dim = cfg.invreg.get("mlp_output_dim", 128)
    
#     if use_mlp:
#         # 初始化特征映射网络
#         mlp_net = MLP(head='mlp', dim_in=features.shape[1], feat_dim=mlp_output_dim).to(device)
#         mlp_net.train()
        
#         # 训练特征映射网络
#         mlp_optimizer = torch.optim.Adam(mlp_net.parameters(), lr=0.2, weight_decay=0.0)
#         mlp_scheduler = MultiStepLR(mlp_optimizer, milestones=[2, 4], gamma=0.1)
        
#         logging.info("训练特征映射网络...")
#         print("训练特征映射网络...")  # 直接打印确保可见
#         global_step = 0
#         for epoch in range(5):  # 训练5个epoch
#             logging.info(f"特征映射网络训练: Epoch {epoch+1}/5")
#             print(f"特征映射网络训练: Epoch {epoch+1}/5")  # 直接打印确保可见
            
#             for batch_idx, (feat, label) in enumerate(train_loader):
#                 if batch_idx % 10 == 0:
#                     logging.info(f"  批次 {batch_idx}/{len(train_loader)}")
#                     print(f"  批次 {batch_idx}/{len(train_loader)}")  # 直接打印确保可见
                
#                 bs = label.size(0)
#                 if bs < 2:  # 需要至少2个样本来计算对比损失
#                     continue
                    
#                 feat = feat.to(device)
#                 label = label.to(device)
                
#                 # 打乱样本顺序
#                 perm = torch.randperm(bs, device=device)
#                 feat = feat[perm]
#                 label = label[perm]
                
#                 # 映射特征
#                 mapped_feat = mlp_net(feat)
                
#                 # 计算对比学习损失
#                 logits, logits_mask, mask, _ = scl_loss_mid(mapped_feat, label, temperature=0.3)
#                 valid_mask = mask.sum(-1) > 0
#                 if valid_mask.sum() == 0:
#                     continue
                    
#                 loss = scl_logits(logits[valid_mask], logits_mask[valid_mask], mask[valid_mask])
                
#                 # 更新网络
#                 mlp_optimizer.zero_grad()
#                 loss.backward()
#                 mlp_optimizer.step()
                
#                 global_step += 1
                
#                 # 记录训练进度
#                 if global_step % 10 == 0 and writer is not None:
#                     writer.add_scalar('partition/mlp_loss', loss.item(), global_step)
            
#             mlp_scheduler.step()
#             logging.info(f"特征映射网络训练: Epoch {epoch+1}/5 完成")
#             print(f"特征映射网络训练: Epoch {epoch+1}/5 完成")  # 直接打印确保可见
        
#         mlp_net.eval()
#         logging.info("特征映射网络训练完成")
#     else:
#         # 如果不使用MLP，创建一个恒等映射的网络
#         class IdentityMLP(nn.Module):
#             def __init__(self):
#                 super(IdentityMLP, self).__init__()
            
#             def forward(self, x):
#                 return x  # 直接返回输入特征
        
#         mlp_net = IdentityMLP().to(device)
#         logging.info("使用恒等映射替代MLP")
    
#     # 初始化分区网络 - 使用新的样本级分区网络
#     # 如果不使用MLP，分区网络的输入维度应该是原始特征维度
#     input_feat_dim = mlp_output_dim if use_mlp else features.shape[1]
#     part_module = Partition(n_classes, env_num, feat_dim=input_feat_dim).to(device)
#     part_module.train()
    
#     # 训练分区网络
#     part_optimizer = torch.optim.Adam(part_module.parameters(), lr=1e-5, weight_decay=0.0)
#     # part_scheduler = MultiStepLR(part_optimizer, milestones=[3, 6, 9, 12], gamma=0.2)
    
#     logging.info("训练环境分区网络...")
#     global_step = 0
#     temperature = 0.3  # 温度参数
#     irm_weight = 0.1  # IRM权重
#     constrain_weight = 0.2  # 平衡约束权重
#     class_consistency_weight = 0.05  # 类别内一致性约束
    
#     for epoch in range(15):  # 训练15个epoch
#         for feat, label in train_loader:
#             bs = label.size(0)
#             if bs < env_num:  # 确保有足够的样本
#                 continue
                
#             feat = feat.to(device)
#             label = label.to(device)
            
#             # 打乱样本顺序
#             perm = torch.randperm(bs, device=device)
#             feat = feat[perm]
#             label = label[perm]
            
#             # 映射特征
#             mapped_feat = mlp_net(feat)
            
#             # 获取样本的环境分配和类别环境分配
#             sample_split, class_split = part_module(mapped_feat, label)
            
#             # 计算对比学习损失和IRM损失
#             loss_cont_list = []
#             loss_penalty_list = []
            
#             for env_idx in range(env_num):
#                 logits, logits_mask, mask, index_sequence = scl_loss_mid(mapped_feat, label, temperature=1.0)
#                 loss_weight = sample_split[:, env_idx][index_sequence]  # 使用样本级环境权重
                
#                 # 计算带权重的对比学习损失
#                 cont_loss_env = scl_logits(logits / temperature, logits_mask, mask, loss_weight, nonorm=False)
                
#                 # 检查对比学习损失是否有NaN
#                 if torch.isnan(cont_loss_env).any():
#                     print(f"检测到NaN in cont_loss_env (环境 {env_idx})")
#                     print(f"logits统计: min={logits.min().item()}, max={logits.max().item()}, 含NaN={torch.isnan(logits).any().item()}")
#                     print(f"loss_weight统计: min={loss_weight.min().item()}, max={loss_weight.max().item()}, 含NaN={torch.isnan(loss_weight).any().item()}")
#                     import sys
#                     sys.exit(1)
                    
#                 loss_cont_list.append(cont_loss_env)
                
#                 # 计算IRM惩罚项
#                 penalty_grad = soft_penalty(logits, logits_mask, mask, loss_weight, nonorm=False, temp=1)
                
#                 # 检查IRM惩罚项是否有NaN
#                 if torch.isnan(penalty_grad).any():
#                     print(f"检测到NaN in penalty_grad (环境 {env_idx})")
#                     import sys
#                     sys.exit(1)
                    
#                 loss_penalty_list.append(penalty_grad)
            
#             # 计算总体损失
#             cont_loss_epoch = torch.stack(loss_cont_list).mean()  # 对比学习损失
#             # inv_loss_epoch = torch.stack(loss_penalty_list).var()  # IRM损失
#             # 计算平均绝对偏差(MAD)作为IRM损失
#             losses_tensor = torch.stack(loss_cont_list)
#             mean_loss = losses_tensor.mean()
#             inv_loss_epoch = (losses_tensor - mean_loss).abs().mean()  # 使用MAD代替方差
            
#             # 环境分布平衡约束
#             constrain_loss = constrain_weight * (
#                 - cal_entropy(sample_split.mean(0), dim=0) +  # 环境分布的熵
#                 cal_entropy(sample_split, dim=1).mean()  # 样本分配的熵
#             )
            
#             # 类别内一致性约束（弱约束，允许同类样本分到不同环境，但保持一定相似性）
#             class_consistency_loss = 0
#             for c in range(n_classes):
#                 class_mask = (label == c)
#                 if class_mask.sum() > 1:
#                     class_samples = sample_split[class_mask]
#                     class_mean = class_samples.mean(0)
#                     # 每个环境的类内样本与类别平均分布的KL散度
#                     kl_div = F.kl_div(
#                         torch.log(class_samples + 1e-8), 
#                         class_mean.expand_as(class_samples),
#                         reduction='batchmean'
#                     )
#                     class_consistency_loss += kl_div
            
#             # 总损失
#             total_loss = - (cont_loss_epoch + irm_weight * inv_loss_epoch) 
#             # + constrain_loss + class_consistency_weight * class_consistency_loss
            
#             # 检查损失是否有NaN
#             if torch.isnan(total_loss).any():
#                 print(f"检测到NaN in total_loss: cont_loss={cont_loss_epoch.item()}, inv_loss={inv_loss_epoch.item()}, constrain_loss={constrain_loss.item()}, class_loss={class_consistency_loss.item()}")
#                 import sys
#                 sys.exit(1)
            
#             # 更新分区网络
#             part_optimizer.zero_grad()
#             total_loss.backward()
        
#             # 检查梯度是否有NaN
#             for name, param in part_module.named_parameters():
#                 if param.grad is not None and torch.isnan(param.grad).any():
#                     print(f"检测到NaN in 分区网络梯度 {name}")
#                     import sys
#                     sys.exit(1)
                    
#             part_optimizer.step()
            
#             # 检查更新后的参数是否有NaN
#             for name, param in part_module.named_parameters():
#                 if torch.isnan(param).any():
#                     print(f"检测到NaN in 分区网络参数 {name}")
#                     import sys
#                     sys.exit(1)
            
#             global_step += 1
            
#             # 记录训练进度
#             if global_step % 10 == 0 and writer is not None:
#                 writer.add_scalar('partition/total_loss', total_loss.item(), global_step)
#                 writer.add_scalar('partition/cont_loss', -cont_loss_epoch.item(), global_step)
#                 writer.add_scalar('partition/inv_loss', -inv_loss_epoch.item(), global_step)
#                 writer.add_scalar('partition/constrain_loss', constrain_loss.item(), global_step)
#                 writer.add_scalar('partition/class_consistency_loss', class_consistency_loss.item(), global_step)
                
#         # part_scheduler.step()
    
#     # 获取整个数据集的最终分区结果
#     logging.info("计算整个数据集的最终分区结果...")
    
#     # 创建包含样本索引的特征数据集
#     class IndexedFeatureDataset(Dataset):
#         def __init__(self, features, labels):
#             self.features = torch.from_numpy(features).float()
#             self.labels = torch.from_numpy(labels).long()
#             self.indices = torch.arange(len(labels))

#         def __getitem__(self, index):
#             return self.features[index], self.labels[index], self.indices[index]

#         def __len__(self):
#             return self.features.shape[0]
    
#     # 创建完整数据加载器（保持原始顺序非常重要）
#     eval_dataset = IndexedFeatureDataset(features, labels)
#     eval_loader = DataLoader(
#         dataset=eval_dataset,
#         batch_size=256,
#         shuffle=False,  # 关键：不打乱顺序
#         num_workers=4,
#         pin_memory=True,
#         drop_last=False
#     )
    
#     # 为每个样本预测环境分配
#     part_module.eval()
#     mlp_net.eval()
    
#     # 创建样本-环境映射矩阵，存储软分配（概率分布）
#     # 维度: [样本数, 环境数]
#     sample_env_matrix = np.zeros((len(labels), env_num))
    
#     with torch.no_grad():
#         for feat, label, indices in eval_loader:
#             feat = feat.to(device)
#             label_np = label.numpy()
#             indices_np = indices.numpy()
            
#             # 获取特征映射
#             # mapped_feat = mlp_net(feat)
#             mapped_feat = feat
#             # 预测环境分配概率分布
#             env_probs = part_module(mapped_feat)
#             env_probs_np = env_probs.cpu().numpy()
            
#             batch_size = feat.size(0)
#             for i in range(batch_size):
#                 sample_idx = indices_np[i]  # 使用原始样本索引
                
#                 # 直接存储样本在各环境的概率分布，不进行硬分配
#                 sample_env_matrix[sample_idx] = env_probs_np[i]
    
#     # 保存样本-环境映射矩阵（软分配）
#     sample_env_file = os.path.join(save_dir, 'sample_env_matrix.npy')
#     np.save(sample_env_file, sample_env_matrix)
    
#     logging.info(f"样本级分区结果（软分配）已保存到 {sample_env_file}")
    
#     # 分析分区统计信息
#     env_counts = np.sum(sample_env_matrix, axis=0)
    
#     logging.info(f"环境分布统计 (软分配权重总和): \n{env_counts}")
#     print(f"环境分布统计 (软分配权重总和): \n{env_counts}")
    
#     return sample_env_matrix  # 返回样本-环境概率矩阵

# def load_past_partition(cfg, epoch):
#     """
#     加载之前的分区结果
    
#     参数:
#         cfg: 配置对象
#         epoch: 当前轮次
        
#     返回:
#         list: 分区列表
#     """
#     save_dir = os.path.join(cfg.output, 'saved_feat', f'epoch_{epoch}')
#     sample_env_file = os.path.join(save_dir, 'sample_env_matrix.npy')
    
#     result = []
    
#     # 加载样本-环境矩阵
#     if os.path.exists(sample_env_file):
#         logging.info(f"加载样本-环境映射文件: {sample_env_file}")
#         sample_env_matrix = np.load(sample_env_file)
#         result.append({
#             'type': 'sample_env',
#             'data': torch.from_numpy(sample_env_matrix).float().to('cuda'),
#             'epoch': epoch,
#             'name': f"epoch_{epoch}_sample_env",
#             'loss': 0.0  # 初始化损失为0
#         })
    
#     if not result:
#         logging.warning(f"分区文件不存在: {sample_env_file}")
    
#     return result

# def assign_losses(losses, labels, partition, env_idx):
#     """
#     将损失分配给特定环境
    
#     参数:
#         losses: 损失列表
#         labels: 标签列表
#         partition: 分区数组
#         env_idx: 环境索引
        
#     返回:
#         torch.Tensor: 分配后的损失
#     """
#     env_mask = partition[:, :, env_idx][labels]
#     if env_mask.sum() == 0:
#         return torch.zeros_like(losses).mean()
    
#     return (losses * env_mask) / env_mask.sum() 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import logging

# MLP network for feature remapping
class MLP(nn.Module):
    def __init__(self, head='mlp', dim_in=512, feat_dim=128):
        super(MLP, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        x_norm = F.normalize(x.float(), dim=1)
        mlp_x = self.head(x_norm)
        return mlp_x

# Environment partition network - modified for sample-level partitioning
class Partition(nn.Module):
    def __init__(self, n_cls, n_env, feat_dim=128):
        super(Partition, self).__init__()
        # Use a classifier + environment assignment network instead of a simple class-environment matrix
        self.env_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_env)
        )
        self.n_env = n_env
        self.n_cls = n_cls
        # Retain class information for class-level constraints
        self.class_env_probs = nn.Parameter(torch.randn((n_cls, n_env)))

    def forward(self, features, labels=None):
        # Predict environment assignment based on features
        env_logits = self.env_classifier(features)
        # Check for NaNs in logits
        if torch.isnan(env_logits).any():
            print(f"Detected NaN in env_logits - exiting immediately")
            import sys
            sys.exit(1)

        sample_split = F.softmax(env_logits, dim=-1)

        # If labels are provided, also return class-level environment assignments (for constraints)
        if labels is not None:
            class_split = F.softmax(self.class_env_probs[labels], dim=-1)
            return sample_split, class_split

        return sample_split

# Compute entropy
def cal_entropy(x, dim=0):
    return -torch.sum(x * torch.log(x + 1e-8), dim=dim)

# Compute weighted contrastive learning loss
def scl_logits(logits, logits_mask, mask, loss_weight=None, mode='scl', nonorm=False):
    """
    Compute weighted contrastive learning loss

    Args:
        logits: similarity matrix
        logits_mask: mask to exclude self comparisons
        mask: mask of same-class samples
        loss_weight: loss weights
        mode: loss mode
        nonorm: whether to skip normalization
    """
    # Numerical stability adjustment
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Compute exp(logits)
    exp_logits = torch.exp(logits) * logits_mask

    # Check for NaN or Inf in exp_logits
    if torch.isnan(exp_logits).any() or torch.isinf(exp_logits).any():
        print(f"Detected NaN/Inf in exp_logits")
        print(f"exp_logits stats: min={exp_logits[~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)].min().item() if (~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)).any() else 'N/A'}, "
              f"max={exp_logits[~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)].max().item() if (~torch.isnan(exp_logits) & ~torch.isinf(exp_logits)).any() else 'N/A'}")
        print(f"Original logits stats: min={logits.min().item()}, max={logits.max().item()}")
        print(f"Number of NaNs: {torch.isnan(exp_logits).sum().item()}, number of Infs: {torch.isinf(exp_logits).sum().item()}")
        import sys
        sys.exit(1)

    if loss_weight is None:
        # If no weights provided, all samples have equal weight
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    else:
        # Weighted loss computation
        weighted_exp_logits = exp_logits * loss_weight

        # Check for NaNs in weighted_exp_logits
        if torch.isnan(weighted_exp_logits).any():
            print(f"Detected NaN in weighted_exp_logits")
            print(f"loss_weight stats: min={loss_weight[~torch.isnan(loss_weight)].min().item() if (~torch.isnan(loss_weight)).any() else 'N/A'}, "
                  f"max={loss_weight[~torch.isnan(loss_weight)].max().item() if (~torch.isnan(loss_weight)).any() else 'N/A'}")
            print(f"Number of NaNs in loss_weight: {torch.isnan(loss_weight).sum().item()}")
            import sys
            sys.exit(1)

        log_prob = logits - torch.log(weighted_exp_logits.sum(1, keepdim=True) + 1e-8)
        weighted_log_prob = mask * log_prob * loss_weight

        if nonorm:
            mean_log_prob_pos = weighted_log_prob.sum(1).mean()
        else:
            # Check denominator for zero or NaN
            denom = (mask * loss_weight).sum(1)
            mean_log_prob_pos = weighted_log_prob.sum(1) / (denom + 1e-8)
            if torch.isnan(mean_log_prob_pos).any():
                print(f"Detected NaN in mean_log_prob_pos")
                import sys
                sys.exit(1)

    # Return loss
    return -mean_log_prob_pos.mean()

# Compute intermediate results for contrastive learning loss
def scl_loss_mid(feature, label, temperature=0.3):
    """
    Compute intermediate results for contrastive learning loss

    Args:
        feature: feature vectors
        label: labels
        temperature: temperature parameter
    """
    device = feature.device
    batch_size = label.shape[0]

    # Normalize features
    feature = F.normalize(feature, dim=1)

    # Create mask of same-class samples
    mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
    # Exclude self comparisons
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
    mask = mask * logits_mask

    # Compute similarity matrix
    logits = torch.div(torch.matmul(feature, feature.T), temperature)

    # Index sequence
    index_sequence = torch.arange(batch_size).to(device)
    index_sequence = index_sequence.unsqueeze(0).expand(batch_size, batch_size)

    return logits, logits_mask, mask, index_sequence

# Compute IRM penalty term
def soft_penalty(logits, logits_mask, mask, loss_weight, mode='scl', nonorm=False, temp=1.0):
    """
    Compute IRM penalty term

    Args:
        logits: similarity matrix
        logits_mask: mask to exclude self comparisons
        mask: mask of same-class samples
        loss_weight: loss weights
        mode: loss mode
        nonorm: whether to skip normalization
        temp: temperature parameter
    """
    # Numerical stability adjustment
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Weighted loss computation    
    exp_logits = torch.exp(logits) * logits_mask

    # Check for NaN or Inf in exp_logits (soft_penalty)
    if torch.isnan(exp_logits).any() or torch.isinf(exp_logits).any():
        print(f"Detected NaN/Inf in exp_logits (soft_penalty)")
        print(f"logits stats: min={logits.min().item()}, max={logits.max().item()}")
        import sys
        sys.exit(1)

    weighted_exp_logits = exp_logits * loss_weight

    # Check for NaNs in weighted_exp_logits (soft_penalty)
    if torch.isnan(weighted_exp_logits).any():
        print(f"Detected NaN in weighted_exp_logits (soft_penalty)")
        import sys
        sys.exit(1)

    # Compute gradient penalty
    denom = weighted_exp_logits.sum(1, keepdim=True)
    weighted_prob = weighted_exp_logits / (denom + 1e-8)
    weighted_log_prob = mask * torch.log(weighted_prob + 1e-8) * loss_weight

    if nonorm:
        grad_penalty = torch.pow(weighted_log_prob.sum(1).mean(), 2)
    else:
        norm_term = (mask * loss_weight).sum(1)
        grad_penalty = torch.pow((weighted_log_prob.sum(1) / (norm_term + 1e-8)).mean(), 2)
        if torch.isnan(grad_penalty).any():
            print(f"Detected NaN in grad_penalty")
            import sys
            sys.exit(1)

    return grad_penalty * temp

# Dataset class for features and labels
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize feature dataset

        Args:
            features: feature matrix
            labels: label array
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]

def update_partition(cfg, save_dir, n_classes, features, labels, writer=None, device=None, rank=0, world_size=1):
    """
    Update data partitions using a network method - sample-level partitioning

    Args:
        cfg: configuration object
        save_dir: directory to save features
        n_classes: number of classes
        features: feature matrix
        labels: label array
        writer: TensorBoard writer
        device: compute device
        rank: process rank
        world_size: total number of processes

    Returns:
        numpy.ndarray: updated partition matrix
    """
    os.makedirs(save_dir, exist_ok=True)
    env_num = cfg.invreg.get("env_num", 2)
    use_mlp = cfg.invreg.get("use_mlp", True)  # whether to use MLP from config

    logging.info(f"Partitioning data into {env_num} environments using network method (improved sample-level partitioning)")
    print(f"Partitioning data into {env_num} environments using network method (improved sample-level partitioning)")
    if use_mlp:
        logging.info("Will use MLP for feature remapping")
    else:
        logging.info("Will partition environments using raw features without MLP remapping")

    # Create dataset and data loader
    dataset = FeatureDataset(features, labels)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=min(2048, len(dataset)),  # use a large batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # Get MLP configuration parameters
    mlp_hidden_dim = cfg.invreg.get("mlp_hidden_dim", 512)
    mlp_output_dim = cfg.invreg.get("mlp_output_dim", 128)

    if use_mlp:
        # Initialize feature mapping network
        mlp_net = MLP(head='mlp', dim_in=features.shape[1], feat_dim=mlp_output_dim).to(device)
        mlp_net.train()

        # Train feature mapping network
        mlp_optimizer = torch.optim.Adam(mlp_net.parameters(), lr=0.2, weight_decay=0.0)
        mlp_scheduler = MultiStepLR(mlp_optimizer, milestones=[2, 4], gamma=0.1)

        logging.info("Training feature mapping network...")
        print("Training feature mapping network...")
        global_step = 0
        for epoch in range(5):  # train for 5 epochs
            logging.info(f"Feature mapping training: Epoch {epoch+1}/5")
            print(f"Feature mapping training: Epoch {epoch+1}/5")

            for batch_idx, (feat, label) in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    logging.info(f"  Batch {batch_idx}/{len(train_loader)}")
                    print(f"  Batch {batch_idx}/{len(train_loader)}")

                bs = label.size(0)
                if bs < 2:  # need at least 2 samples for contrastive loss
                    continue

                feat = feat.to(device)
                label = label.to(device)

                # Shuffle samples
                perm = torch.randperm(bs, device=device)
                feat = feat[perm]
                label = label[perm]

                # Map features
                mapped_feat = mlp_net(feat)

                # Compute contrastive learning loss
                logits, logits_mask, mask, _ = scl_loss_mid(mapped_feat, label, temperature=0.3)
                valid_mask = mask.sum(-1) > 0
                if valid_mask.sum() == 0:
                    continue

                loss = scl_logits(logits[valid_mask], logits_mask[valid_mask], mask[valid_mask])

                # Update network
                mlp_optimizer.zero_grad()
                loss.backward()
                mlp_optimizer.step()

                global_step += 1

                # Log training progress
                if global_step % 10 == 0 and writer is not None:
                    writer.add_scalar('partition/mlp_loss', loss.item(), global_step)

            mlp_scheduler.step()
            logging.info(f"Feature mapping training: Epoch {epoch+1}/5 completed")
            print(f"Feature mapping training: Epoch {epoch+1}/5 completed")

        mlp_net.eval()
        logging.info("Feature mapping network training complete")
    else:
        # If not using MLP, use an identity mapping network
        class IdentityMLP(nn.Module):
            def __init__(self):
                super(IdentityMLP, self).__init__()

            def forward(self, x):
                return x  # return input features directly

        mlp_net = IdentityMLP().to(device)
        logging.info("Using identity mapping in place of MLP")

    # Initialize partition network - sample-level partitioning
    input_feat_dim = mlp_output_dim if use_mlp else features.shape[1]
    part_module = Partition(n_classes, env_num, feat_dim=input_feat_dim).to(device)
    part_module.train()

    # Train partition network
    part_optimizer = torch.optim.Adam(part_module.parameters(), lr=1e-5, weight_decay=0.0)

    logging.info("Training environment partition network...")
    global_step = 0
    temperature = 0.3  # temperature parameter
    irm_weight = 0.1  # IRM weight
    constrain_weight = 0.2  # constraint weight for balanced env distribution
    class_consistency_weight = 0.05  # intra-class consistency weight

    for epoch in range(15):  # train for 15 epochs
        for feat, label in train_loader:
            bs = label.size(0)
            if bs < env_num:  # ensure enough samples
                continue

            feat = feat.to(device)
            label = label.to(device)

            # Shuffle samples
            perm = torch.randperm(bs, device=device)
            feat = feat[perm]
            label = label[perm]

            # Map features
            mapped_feat = mlp_net(feat)

            # Get sample and class environment allocations
            sample_split, class_split = part_module(mapped_feat, label)

            # Compute contrastive loss and IRM penalty
            loss_cont_list = []
            loss_penalty_list = []

            for env_idx in range(env_num):
                logits, logits_mask, mask, index_sequence = scl_loss_mid(mapped_feat, label, temperature=1.0)
                loss_weight_env = sample_split[:, env_idx][index_sequence]  # sample-level env weights

                # Compute weighted contrastive learning loss
                cont_loss_env = scl_logits(logits / temperature, logits_mask, mask, loss_weight_env, nonorm=False)

                if torch.isnan(cont_loss_env).any():
                    print(f"Detected NaN in cont_loss_env (environment {env_idx})")
                    print(f"logits stats: min={logits.min().item()}, max={logits.max().item()}, hasNaN={torch.isnan(logits).any().item()}")
                    print(f"loss_weight stats: min={loss_weight_env.min().item()}, max={loss_weight_env.max().item()}, hasNaN={torch.isnan(loss_weight_env).any().item()}")
                    import sys
                    sys.exit(1)

                loss_cont_list.append(cont_loss_env)

                # Compute IRM penalty
                penalty_grad = soft_penalty(logits, logits_mask, mask, loss_weight_env, nonorm=False, temp=1)

                if torch.isnan(penalty_grad).any():
                    print(f"Detected NaN in penalty_grad (environment {env_idx})")
                    import sys
                    sys.exit(1)

                loss_penalty_list.append(penalty_grad)

            # Compute total loss
            cont_loss_epoch = torch.stack(loss_cont_list).mean()
            losses_tensor = torch.stack(loss_cont_list)
            mean_loss = losses_tensor.mean()
            # Use mean absolute deviation instead of variance for IRM loss
            inv_loss_epoch = (losses_tensor - mean_loss).abs().mean()

            # Constraint for balanced environment distribution
            constrain_loss = constrain_weight * (
                -cal_entropy(sample_split.mean(0), dim=0) +  # entropy of environment distribution
                cal_entropy(sample_split, dim=1).mean()      # average entropy of sample assignments
            )

            # Intra-class consistency constraint
            class_consistency_loss = 0
            for c in range(n_classes):
                class_mask = (label == c)
                if class_mask.sum() > 1:
                    class_samples = sample_split[class_mask]
                    class_mean = class_samples.mean(0)
                    kl_div = F.kl_div(
                        torch.log(class_samples + 1e-8),
                        class_mean.expand_as(class_samples),
                        reduction='batchmean'
                    )
                    class_consistency_loss += kl_div

            total_loss = - (cont_loss_epoch + irm_weight * inv_loss_epoch)
            # + constrain_loss + class_consistency_weight * class_consistency_loss

            if torch.isnan(total_loss).any():
                print(f"Detected NaN in total_loss: cont_loss={cont_loss_epoch.item()}, inv_loss={inv_loss_epoch.item()}, "
                      f"constrain_loss={constrain_loss.item()}, class_loss={class_consistency_loss.item()}")
                import sys
                sys.exit(1)

            # Update partition network
            part_optimizer.zero_grad()
            total_loss.backward()

            # Check gradients for NaNs
            for name, param in part_module.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"Detected NaN in partition network gradient {name}")
                    import sys
                    sys.exit(1)

            part_optimizer.step()

            # Check updated parameters for NaNs
            for name, param in part_module.named_parameters():
                if torch.isnan(param).any():
                    print(f"Detected NaN in partition network parameter {name}")
                    import sys
                    sys.exit(1)

            global_step += 1

            if global_step % 10 == 0 and writer is not None:
                writer.add_scalar('partition/total_loss', total_loss.item(), global_step)
                writer.add_scalar('partition/cont_loss', -cont_loss_epoch.item(), global_step)
                writer.add_scalar('partition/inv_loss', -inv_loss_epoch.item(), global_step)
                writer.add_scalar('partition/constrain_loss', constrain_loss.item(), global_step)
                writer.add_scalar('partition/class_consistency_loss', class_consistency_loss.item(), global_step)

    # Compute final partition results for the full dataset
    logging.info("Computing final partition results for the full dataset...")

    # Dataset class with indices for evaluation
    class IndexedFeatureDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.from_numpy(features).float()
            self.labels = torch.from_numpy(labels).long()
            self.indices = torch.arange(len(labels))

        def __getitem__(self, index):
            return self.features[index], self.labels[index], self.indices[index]

        def __len__(self):
            return self.features.shape[0]

    # Create evaluation loader (preserve original order)
    eval_dataset = IndexedFeatureDataset(features, labels)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=256,
        shuffle=False,  # important: do not shuffle
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    part_module.eval()
    mlp_net.eval()

    # Matrix to store soft assignment probabilities per sample
    sample_env_matrix = np.zeros((len(labels), env_num))

    with torch.no_grad():
        for feat, label, indices in eval_loader:
            feat = feat.to(device)
            indices_np = indices.numpy()

            # Use identity mapping if MLP not used
            mapped_feat = feat if not use_mlp else mlp_net(feat)
            env_probs = part_module(mapped_feat)
            env_probs_np = env_probs.cpu().numpy()

            batch_size = feat.size(0)
            for i in range(batch_size):
                sample_idx = indices_np[i]
                sample_env_matrix[sample_idx] = env_probs_np[i]

    # Save soft assignment matrix
    sample_env_file = os.path.join(save_dir, 'sample_env_matrix.npy')
    np.save(sample_env_file, sample_env_matrix)
    logging.info(f"Soft assignment matrix saved to {sample_env_file}")
    print(f"Environment distribution statistics (soft assignment sums):\n{sample_env_matrix.sum(axis=0)}")

    return sample_env_matrix

def load_past_partition(cfg, epoch):
    """
    Load previous partition results

    Args:
        cfg: configuration object
        epoch: epoch number

    Returns:
        list: list of partition dicts
    """
    save_dir = os.path.join(cfg.output, 'saved_feat', f'epoch_{epoch}')
    sample_env_file = os.path.join(save_dir, 'sample_env_matrix.npy')

    result = []

    if os.path.exists(sample_env_file):
        logging.info(f"Loading sample-environment mapping file: {sample_env_file}")
        sample_env_matrix = np.load(sample_env_file)
        result.append({
            'type': 'sample_env',
            'data': torch.from_numpy(sample_env_matrix).float().to('cuda'),
            'epoch': epoch,
            'name': f"epoch_{epoch}_sample_env",
            'loss': 0.0
        })

    if not result:
        logging.warning(f"Partition file not found: {sample_env_file}")

    return result

def assign_losses(losses, labels, partition, env_idx):
    """
    Assign losses to a specific environment

    Args:
        losses: tensor of losses
        labels: tensor of labels
        partition: partition array
        env_idx: environment index

    Returns:
        torch.Tensor: assigned losses
    """
    env_mask = partition[:, :, env_idx][labels]
    if env_mask.sum() == 0:
        return torch.zeros_like(losses).mean()

    return (losses * env_mask) / env_mask.sum()
