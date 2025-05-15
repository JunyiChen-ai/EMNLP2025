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
