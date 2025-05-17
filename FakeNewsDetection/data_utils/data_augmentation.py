import torch
import random
import numpy as np
from transformers import BertTokenizer
import logging
import os

# Setting proxy environment variables

# Define a simple stopwords list to avoid depending on nltk download
STOPWORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
            "in", "on", "at", "to", "for", "with", "by", "about", "of", "this",
            "that", "these", "those", "it", "its", "they", "them", "their", "he",
            "she", "him", "her", "his", "hers", "we", "us", "our", "you", "your"}

class TextAugmenter:
    """Text data augmentation tool providing multiple augmentation methods"""
    
    def __init__(self, tokenizer, max_length=256, p=0.5):
        """
        Initialize the text augmenter
        
        Parameters:
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            p: Probability of applying augmentation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.p = p
        
    def random_deletion(self, tokens, p=0.1):
        """
        Randomly delete a portion of tokens (non-stopwords)
        
        Parameters:
            tokens: List of tokens
            p: Deletion probability
            
        Returns:
            list: Processed token list
        """
        if len(tokens) <= 3:
            return tokens
            
        # Preserve special tokens and necessary vocabulary
        keep_indices = [i for i, token in enumerate(tokens) 
                        if token.startswith('[') or token.endswith(']') 
                        or token.lower() in STOPWORDS
                        or random.random() > p]
                        
        return [tokens[i] for i in sorted(keep_indices)]
    
    def random_swap(self, tokens, n=1):
        """
        Randomly swap n pairs of adjacent tokens
        
        Parameters:
            tokens: List of tokens
            n: Number of pairs to swap
            
        Returns:
            list: Processed token list
        """
        if len(tokens) <= 3:
            return tokens
        
        new_tokens = tokens.copy()
        
        # Exclude special tokens
        valid_indices = [i for i, token in enumerate(tokens[:-1]) 
                         if not (token.startswith('[') or token.endswith(']') 
                               or tokens[i+1].startswith('[') or tokens[i+1].endswith(']'))]
        
        # Perform n swaps
        for _ in range(min(n, len(valid_indices))):
            if not valid_indices:
                break
                
            idx = random.choice(valid_indices)
            valid_indices.remove(idx)
            
            # Swap adjacent tokens
            new_tokens[idx], new_tokens[idx+1] = new_tokens[idx+1], new_tokens[idx]
            
        return new_tokens
    
    def synonym_replacement(self, text, n=2):
        """
        Synonym replacement (simplified version for common words)
        
        Parameters:
            text: Original text
            n: Number of replacements
            
        Returns:
            str: Processed text
        """
        # Simplified synonym dictionary
        synonyms = {
            "good": ["nice", "excellent", "great", "positive"],
            "bad": ["poor", "terrible", "negative", "awful"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["tiny", "little", "miniature", "slight"],
            "important": ["essential", "crucial", "significant", "vital"],
            "fake": ["false", "untrue", "fabricated", "counterfeit"],
            "true": ["real", "genuine", "authentic", "factual"],
            "say": ["state", "mention", "express", "declare"],
            "think": ["believe", "consider", "assume", "suppose"],
            "make": ["create", "produce", "generate", "form"],
            "get": ["obtain", "acquire", "receive", "gain"],
            "increase": ["grow", "rise", "expand", "extend"],
            "decrease": ["reduce", "shrink", "decline", "diminish"]
        }
        
        # Use simple space tokenization instead of nltk
        words = text.split()
        
        # Find replaceable words
        candidates = []
        for i, word in enumerate(words):
            if word.lower() in synonyms and len(synonyms[word.lower()]) > 0:
                candidates.append((i, word.lower()))
        
        # Randomly replace n words
        if candidates and n > 0:
            random.shuffle(candidates)
            for i, word in candidates[:min(n, len(candidates))]:
                replacement = random.choice(synonyms[word])
                if words[i][0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement
        
        return ' '.join(words)
    
    def augment(self, text, label):
        """
        Apply augmentation to text
        
        Parameters:
            text: Input text or encoded input_ids
            label: Label
            
        Returns:
            tuple: (augmented input_ids, attention_mask, label)
        """
        # If input is already encoded input_ids, decode first
        if isinstance(text, torch.Tensor) or isinstance(text, list):
            if isinstance(text, torch.Tensor):
                text = text.tolist()
            text = self.tokenizer.decode(text, skip_special_tokens=True)
        
        # Randomly select an augmentation method
        method = random.choice(['none', 'deletion', 'swap', 'synonym'])
        
        if method == 'none' or random.random() > self.p:
            # No augmentation, directly encode
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), label
        
        try:
            if method == 'deletion':
                # Tokenize
                tokens = self.tokenizer.tokenize(text)
                # Apply deletion
                augmented_tokens = self.random_deletion(tokens)
                # Recombine text
                augmented_text = self.tokenizer.convert_tokens_to_string(augmented_tokens)
            
            elif method == 'swap':
                # Tokenize
                tokens = self.tokenizer.tokenize(text)
                # Apply swap
                augmented_tokens = self.random_swap(tokens)
                # Recombine text
                augmented_text = self.tokenizer.convert_tokens_to_string(augmented_tokens)
            
            elif method == 'synonym':
                # Apply synonym replacement
                augmented_text = self.synonym_replacement(text)
            
            # Encode the augmented text
            encoded = self.tokenizer(
                augmented_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), label
        except Exception as e:
            logging.warning(f"Error during text augmentation: {e}, using original text")
            # If augmentation fails, use the original text
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), label

class AugmentedDataset(torch.utils.data.Dataset):
    """Data augmentation dataset wrapper"""
    
    def __init__(self, original_dataset, tokenizer, max_length=256, p=0.3, augment_size=0.5):
        """
        Initialize the augmented dataset
        
        Parameters:
            original_dataset: Original dataset
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            p: Probability of applying augmentation
            augment_size: Proportion of augmented data relative to the original dataset
        """
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.augmenter = TextAugmenter(tokenizer, max_length, p)
        self.max_length = max_length
        
        # Calculate number of augmented samples
        self.original_size = len(original_dataset)
        self.augment_size = int(self.original_size * augment_size)
        
        # Select sample indices for augmentation
        self.augment_indices = np.random.choice(
            self.original_size, 
            self.augment_size, 
            replace=False if self.augment_size <= self.original_size else True
        )
        
        logging.info(f"Original dataset size: {self.original_size}, Number of augmented samples: {self.augment_size}")
        
    def __len__(self):
        return self.original_size + self.augment_size
        
    def __getitem__(self, idx):
        """Get a sample"""
        if idx < self.original_size:
            # Return original sample
            sample = self.original_dataset[idx]
            
            # Ensure all tensors have consistent length
            if isinstance(sample['input_ids'], torch.Tensor) and sample['input_ids'].size(0) != self.max_length:
                # Reprocess to ensure consistent length
                text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'label': sample['label']
                }
            return sample
        else:
            # Return augmented sample
            aug_idx = self.augment_indices[idx - self.original_size]
            orig_sample = self.original_dataset[aug_idx]
            
            # Apply augmentation
            try:
                input_ids, attention_mask, label = self.augmenter.augment(
                    orig_sample['input_ids'],
                    orig_sample['label']
                )
                
                # Ensure consistent length
                if input_ids.size(0) != self.max_length:
                    # Reapply padding and truncation
                    text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    encoded = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    input_ids = encoded['input_ids'].squeeze(0)
                    attention_mask = encoded['attention_mask'].squeeze(0)
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'label': label
                }
            except Exception as e:
                logging.error(f"Error while augmenting sample: {e}, returning original sample")
                # If augmentation fails, return the original sample
                return self.original_dataset[aug_idx] 