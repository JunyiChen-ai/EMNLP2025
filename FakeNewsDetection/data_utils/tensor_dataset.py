# import os
# import pickle
# import logging
# import torch
# from torch.utils.data import Dataset, DataLoader, Subset

# def remove_empty_tensors(dataset, indices, subset_name="未命名子集"):
#     """
#     移除数据集子集中的空张量样本（input_ids全为0的样本）
#     直接检查每个样本而不使用DataLoader，避免影响随机状态
    
#     参数:
#         dataset: 原始数据集
#         indices: 子集索引列表
#         subset_name: 子集名称，用于日志输出
        
#     返回:
#         list: 过滤后的索引列表
#     """
#     # 保留非空张量的索引
#     valid_indices = []
#     empty_count = 0
    
#     # 直接遍历索引，避免使用DataLoader
#     for i, idx in enumerate(indices):
#         try:
#             # 获取样本
#             sample = dataset[idx]
#             input_ids = sample['input_ids']
            
#             # 检查是否为空张量
#             if torch.sum(input_ids).item() == 0:
#                 empty_count += 1
#                 if empty_count % 10 == 0:
#                     logging.info(f"{subset_name}：已发现 {empty_count} 个空张量样本")
#             else:
#                 # 保留原始索引
#                 valid_indices.append(idx)
#         except Exception as e:
#             logging.error(f"处理索引 {idx} 的样本时出错: {e}")
#             # 出错时默认保留该索引
#             valid_indices.append(idx)
    
#     # 记录处理结果
#     removed_count = len(indices) - len(valid_indices)
#     if removed_count > 0:
#         logging.info(f"{subset_name}：从 {len(indices)} 个样本中移除了 {removed_count} 个空张量样本，剩余 {len(valid_indices)} 个样本")
#     else:
#         logging.info(f"{subset_name}：未发现空张量样本，保留全部 {len(indices)} 个样本")
    
#     return valid_indices

# class PreprocessedTensorDataset(Dataset):
#     """
#     针对已经预处理为张量格式的数据集
#     适用于包含input_ids, attention_mask和labels的数据
#     添加了索引追踪功能
#     """
#     def __init__(self, data_path, max_length=256,language='en'):
#         """
#         初始化预处理张量数据集
        
#         参数:
#             data_path (str): 预处理数据的路径
#             max_length (int): 最大序列长度，用于截断过长的序列
#         """
#         super(PreprocessedTensorDataset, self).__init__()
#         self.max_length = max_length
        
#         # 加载预处理后的数据
#         logging.info(f"加载预处理张量数据: {data_path}")
#         with open(data_path, 'rb') as f:
#             self.data = pickle.load(f)
        
#         # 检查数据格式
#         if not isinstance(self.data, dict):
#             raise ValueError(f"预期数据类型为字典，但得到的是 {type(self.data)}")
        
#         # 检查必要的键是否存在
#         required_keys = ['input_ids', 'attention_mask', 'labels']
#         for key in required_keys:
#             if key not in self.data:
#                 raise ValueError(f"缺少必要的键: {key}")
            
#         # 检查数据尺寸是否匹配
#         if hasattr(self.data['input_ids'], 'size'):
#             self.num_samples = self.data['input_ids'].size(0)
#         else:
#             self.num_samples = len(self.data['input_ids'])
        
#         logging.info(f"成功加载 {self.num_samples} 个样本")
        
#         # 添加样本索引数组，用于追踪样本
#         self.sample_indices = torch.arange(self.num_samples)
        
#         # 标准化authors字段，确保格式统一
#         if 'authors' in self.data:
#             logging.info("检测到authors字段，进行标准化处理...")
#             standardized_authors = []
#             for author in self.data['authors']:
#                 if author is None or author == 'Unknown' or author == 'unknown' or (isinstance(author, list) and len(author) == 0):
#                     standardized_authors.append('unknown')
#                 else:
#                     standardized_authors.append('known')
#             self.data['authors'] = standardized_authors
#             logging.info(f"authors字段标准化完成，共{len(standardized_authors)}个条目")
        
#         # 打印张量形状信息
#         for key, tensor in self.data.items():
#             if hasattr(tensor, 'shape'):
#                 logging.info(f"{key} 形状: {tensor.shape}")
#             elif isinstance(tensor, list):
#                 logging.info(f"{key} 是列表，长度: {len(tensor)}")
#             else:
#                 logging.info(f"{key} 类型: {type(tensor)}")
                
#         # 检查每个样本的输入ID长度
#         input_id_lengths = set()
#         if isinstance(self.data['input_ids'], torch.Tensor):
#             # 如果是张量，检查每个样本的长度
#             for i in range(self.num_samples):
#                 input_id_lengths.add(self.data['input_ids'][i].size(0) if hasattr(self.data['input_ids'][i], 'size') else len(self.data['input_ids'][i]))
#         else:
#             # 如果是列表，检查每个样本的长度
#             for item in self.data['input_ids']:
#                 input_id_lengths.add(item.size(0) if hasattr(item, 'size') else len(item))
        
#         if len(input_id_lengths) > 1:
#             logging.warning(f"数据集中存在不同长度的输入ID: {input_id_lengths}，将统一为 {max_length}")
        
#         # 检查群组信息是否存在
#         self.has_group_info = ('topics' in self.data and 'platforms' in self.data and 'authors' in self.data)
#         if self.has_group_info:
#             logging.info("检测到群组信息: topics, platforms, authors")
            
#         # 加载domain字典（如果存在）
#         self.domain_dict = None
#         if language == 'en':
#             domain_dict_path = os.path.join('FakeNewsDetection', 'output', 'domain_dict.pkl')
#         else:
#             domain_dict_path = os.path.join('FakeNewsDetection', 'output', 'domain_dict_ch.pkl')
#         if os.path.exists(domain_dict_path):
#             try:
#                 with open(domain_dict_path, 'rb') as f:
#                     self.domain_dict = pickle.load(f)
#                 logging.info(f"加载domain字典成功，包含 {len(self.domain_dict)} 个domain")
#             except Exception as e:
#                 logging.warning(f"加载domain字典失败: {e}")
    
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, index):
#         """获取一个样本及其全局索引"""
#         try:
#             # 获取输入ID和注意力掩码
#             if isinstance(self.data['input_ids'], torch.Tensor):
#                 input_ids = self.data['input_ids'][index]
#             else:
#                 input_ids = torch.tensor(self.data['input_ids'][index], dtype=torch.long)
                
#             if isinstance(self.data['attention_mask'], torch.Tensor):
#                 attention_mask = self.data['attention_mask'][index]
#             else:
#                 attention_mask = torch.tensor(self.data['attention_mask'][index], dtype=torch.long)
            
#             # 获取标签
#             if isinstance(self.data['labels'], torch.Tensor):
#                 label = self.data['labels'][index].long()
#             else:
#                 label = torch.tensor(self.data['labels'][index], dtype=torch.long)
            
#             # 获取样本全局索引
#             sample_idx = self.sample_indices[index]
            
#             # 确保一致的长度
#             current_length = input_ids.size(0) if hasattr(input_ids, 'size') else len(input_ids)
            
#             if current_length != self.max_length:
#                 # 处理长度不一致的情况
#                 if current_length > self.max_length:
#                     # 截断过长的序列
#                     input_ids = input_ids[:self.max_length]
#                     attention_mask = attention_mask[:self.max_length]
#                 else:
#                     # 填充过短的序列
#                     padding = torch.zeros(self.max_length - current_length, dtype=input_ids.dtype, device=input_ids.device)
#                     input_ids = torch.cat([input_ids, padding])
#                     attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)])
            
#             result = {
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask,
#                 'label': label,
#                 'sample_idx': sample_idx  # 添加样本索引
#             }
            
#             # 如果存在群组信息，添加到返回字典中
#             if self.has_group_info:
#                 # 获取群组信息
#                 if 'topics' in self.data:
#                     topic = self.data['topics'][index]
#                     result['topic'] = topic
                    
#                     # 添加domain_id（即topic的id）
#                     if self.domain_dict is not None and topic in self.domain_dict:
#                         result['domain_id'] = torch.tensor(self.domain_dict[topic], dtype=torch.long)
#                     else:
#                         # 如果没有找到映射，使用0作为默认值
#                         print(f"没有找到映射，使用0作为默认值: {topic}")
#                         result['domain_id'] = torch.tensor(0, dtype=torch.long)
                        
#                 if 'platforms' in self.data:
#                     result['platform'] = self.data['platforms'][index]
#                 if 'authors' in self.data:
#                     # 确保authors字段为字符串类型
#                     author = self.data['authors'][index]
#                     if isinstance(author, list):
#                         # 如果仍然是列表，则根据列表长度决定值
#                         result['author'] = 'known' if len(author) > 0 else 'unknown'
#                     else:
#                         # 否则直接使用该值
#                         result['author'] = author
            
#             return result
#         except Exception as e:
#             logging.error(f"获取索引 {index} 的样本时出错: {e}")
#             # 返回一个默认样本
#             result = {
#                 'input_ids': torch.zeros(self.max_length, dtype=torch.long),
#                 'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
#                 'label': torch.tensor(0, dtype=torch.long),
#                 'sample_idx': torch.tensor(index, dtype=torch.long),  # 添加样本索引
#                 'domain_id': torch.tensor(0, dtype=torch.long)  # 添加默认domain_id
#             }
            
#             # 如果存在群组信息，添加默认值
#             if self.has_group_info:
#                 result['topic'] = 0
#                 result['platform'] = 0
#                 result['author'] = 0
                
#             return result

# def get_tensor_dataloader(data_path, batch_size=32, max_length=256, num_workers=4, shuffle=True,language='en'):
#     """
#     创建预处理张量数据的加载器
    
#     参数:
#         data_path (str): 数据路径
#         batch_size (int): 批量大小
#         max_length (int): 最大序列长度
#         num_workers (int): 加载数据的工作线程数
#         shuffle (bool): 是否打乱数据
        
#     返回:
#         DataLoader: 训练数据加载器
#     """
#     # 检查文件是否存在
#     if not os.path.exists(data_path):
#         logging.error(f"数据文件不存在: {data_path}")
#         raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
#     dataset = PreprocessedTensorDataset(
#         data_path=data_path,
#         max_length=max_length,
#         language=language
#     )
    
#     # 确保数据集非空
#     if len(dataset) == 0:
#         logging.error("数据集为空")
#         raise ValueError("数据集为空，无法创建DataLoader")
    
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False
#     )
    
#     return dataloader 
import os
import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader, Subset

def remove_empty_tensors(dataset, indices, subset_name="未命名子集"):
    """
    Remove empty tensor samples from a dataset subset (samples whose input_ids are all zeros).
    Directly checks each sample without using DataLoader to avoid affecting random state.

    Args:
        dataset: original dataset
        indices: list of subset indices
        subset_name: name of the subset for logging

    Returns:
        list: filtered list of indices
    """
    # Keep indices of non-empty tensors
    valid_indices = []
    empty_count = 0

    # Iterate indices directly to avoid using DataLoader
    for i, idx in enumerate(indices):
        try:
            # Retrieve sample
            sample = dataset[idx]
            input_ids = sample['input_ids']

            # Check if tensor is empty
            if torch.sum(input_ids).item() == 0:
                empty_count += 1
                if empty_count % 10 == 0:
                    logging.info(f"{subset_name}: found {empty_count} empty tensor samples")
            else:
                # Keep original index
                valid_indices.append(idx)
        except Exception as e:
            logging.error(f"Error processing sample at index {idx}: {e}")
            # Keep index by default if error occurs
            valid_indices.append(idx)

    # Log processing results
    removed_count = len(indices) - len(valid_indices)
    if removed_count > 0:
        logging.info(f"{subset_name}: removed {removed_count} empty tensor samples out of {len(indices)}, {len(valid_indices)} samples remain")
    else:
        logging.info(f"{subset_name}: no empty tensor samples found, kept all {len(indices)} samples")

    return valid_indices


class PreprocessedTensorDataset(Dataset):
    """
    Dataset for preprocessed tensor-format data.
    Suitable for data containing input_ids, attention_mask, and labels.
    Adds sample index tracking.
    """
    def __init__(self, data_path, max_length=256, language='en'):
        """
        Initialize preprocessed tensor dataset

        Args:
            data_path (str): path to preprocessed data
            max_length (int): maximum sequence length for truncation
        """
        super(PreprocessedTensorDataset, self).__init__()
        self.max_length = max_length

        # Load preprocessed data
        logging.info(f"Loading preprocessed tensor data: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # Check data format
        if not isinstance(self.data, dict):
            raise ValueError(f"Expected data type dict but got {type(self.data)}")

        # Check for required keys
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required key: {key}")

        # Determine number of samples
        if hasattr(self.data['input_ids'], 'size'):
            self.num_samples = self.data['input_ids'].size(0)
        else:
            self.num_samples = len(self.data['input_ids'])

        logging.info(f"Successfully loaded {self.num_samples} samples")

        # Add sample indices for tracking
        self.sample_indices = torch.arange(self.num_samples)

        # Standardize authors field if present
        if 'authors' in self.data:
            logging.info("Detected 'authors' field, standardizing...")
            standardized_authors = []
            for author in self.data['authors']:
                if author is None or author == 'Unknown' or author == 'unknown' or (isinstance(author, list) and len(author) == 0):
                    standardized_authors.append('unknown')
                else:
                    standardized_authors.append('known')
            self.data['authors'] = standardized_authors
            logging.info(f"Completed standardization of 'authors' field, total entries: {len(standardized_authors)}")

        # Log tensor shape information
        for key, tensor in self.data.items():
            if hasattr(tensor, 'shape'):
                logging.info(f"{key} shape: {tensor.shape}")
            elif isinstance(tensor, list):
                logging.info(f"{key} is a list, length: {len(tensor)}")
            else:
                logging.info(f"{key} type: {type(tensor)}")

        # Check input_id lengths
        input_id_lengths = set()
        if isinstance(self.data['input_ids'], torch.Tensor):
            for i in range(self.num_samples):
                input_id_lengths.add(
                    self.data['input_ids'][i].size(0)
                    if hasattr(self.data['input_ids'][i], 'size')
                    else len(self.data['input_ids'][i])
                )
        else:
            for item in self.data['input_ids']:
                input_id_lengths.add(item.size(0) if hasattr(item, 'size') else len(item))

        if len(input_id_lengths) > 1:
            logging.warning(f"Found varying input_id lengths: {input_id_lengths}, will standardize to {max_length}")

        # Check for group information
        self.has_group_info = ('topics' in self.data and 'platforms' in self.data and 'authors' in self.data)
        if self.has_group_info:
            logging.info("Detected group information: topics, platforms, authors")

        # Load domain dictionary if exists
        self.domain_dict = None
        if language == 'en':
            domain_dict_path = os.path.join('FakeNewsDetection', 'output', 'domain_dict.pkl')
        else:
            domain_dict_path = os.path.join('FakeNewsDetection', 'output', 'domain_dict_ch.pkl')
        if os.path.exists(domain_dict_path):
            try:
                with open(domain_dict_path, 'rb') as f:
                    self.domain_dict = pickle.load(f)
                logging.info(f"Loaded domain dictionary successfully, contains {len(self.domain_dict)} domains")
            except Exception as e:
                logging.warning(f"Failed to load domain dictionary: {e}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """Get a sample and its global index"""
        try:
            # Retrieve input_ids and attention_mask
            if isinstance(self.data['input_ids'], torch.Tensor):
                input_ids = self.data['input_ids'][index]
            else:
                input_ids = torch.tensor(self.data['input_ids'][index], dtype=torch.long)

            if isinstance(self.data['attention_mask'], torch.Tensor):
                attention_mask = self.data['attention_mask'][index]
            else:
                attention_mask = torch.tensor(self.data['attention_mask'][index], dtype=torch.long)

            # Retrieve label
            if isinstance(self.data['labels'], torch.Tensor):
                label = self.data['labels'][index].long()
            else:
                label = torch.tensor(self.data['labels'][index], dtype=torch.long)

            # Retrieve sample global index
            sample_idx = self.sample_indices[index]

            # Ensure consistent length
            current_length = input_ids.size(0) if hasattr(input_ids, 'size') else len(input_ids)
            if current_length != self.max_length:
                # Handle inconsistent lengths
                if current_length > self.max_length:
                    # Truncate sequences that are too long
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                else:
                    # Pad sequences that are too short
                    padding = torch.zeros(self.max_length - current_length, dtype=input_ids.dtype, device=input_ids.device)
                    input_ids = torch.cat([input_ids, padding])
                    attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)])

            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'sample_idx': sample_idx  # include sample index
            }

            # If group info exists, add to returned dict
            if self.has_group_info:
                # Get group information
                if 'topics' in self.data:
                    topic = self.data['topics'][index]
                    result['topic'] = topic

                    # Add domain_id (i.e., topic's id)
                    if self.domain_dict is not None and topic in self.domain_dict:
                        result['domain_id'] = torch.tensor(self.domain_dict[topic], dtype=torch.long)
                    else:
                        print(f"No mapping found for {topic}, using 0 as default")
                        result['domain_id'] = torch.tensor(0, dtype=torch.long)

                if 'platforms' in self.data:
                    result['platform'] = self.data['platforms'][index]
                if 'authors' in self.data:
                    author = self.data['authors'][index]
                    if isinstance(author, list):
                        # If still a list, decide value based on list length
                        result['author'] = 'known' if len(author) > 0 else 'unknown'
                    else:
                        # Otherwise use value directly
                        result['author'] = author

            return result
        except Exception as e:
            logging.error(f"Error retrieving sample at index {index}: {e}")
            # Return a default sample
            result = {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'label': torch.tensor(0, dtype=torch.long),
                'sample_idx': torch.tensor(index, dtype=torch.long),  # add sample index
                'domain_id': torch.tensor(0, dtype=torch.long)  # add default domain_id
            }
            # If group info exists, add default values
            if self.has_group_info:
                result['topic'] = 0
                result['platform'] = 0
                result['author'] = 0
            return result

def get_tensor_dataloader(data_path, batch_size=32, max_length=256, num_workers=4, shuffle=True, language='en'):
    """
    Create DataLoader for preprocessed tensor data

    Args:
        data_path (str): data path
        batch_size (int): batch size
        max_length (int): maximum sequence length
        num_workers (int): number of worker threads for loading
        shuffle (bool): whether to shuffle data

    Returns:
        DataLoader: training DataLoader
    """
    # Check if file exists
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataset = PreprocessedTensorDataset(
        data_path=data_path,
        max_length=max_length,
        language=language
    )

    # Ensure dataset is not empty
    if len(dataset) == 0:
        logging.error("Dataset is empty")
        raise ValueError("Dataset is empty, cannot create DataLoader")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader
