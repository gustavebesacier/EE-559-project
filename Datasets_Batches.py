import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        return text, label


def batch_tokenize(texts, tokenizer, batch_size=32):
    tokenized_texts = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        # Tokenize the batch
        batch_tokenized = tokenizer(batch_texts, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_texts.append(batch_tokenized)


    # Concatenate tokenized batches
    tokenized_result = {key: torch.cat([batch[key] for batch in tokenized_texts], dim=0) for key in
                        tokenized_texts[0].keys()}

    return tokenized_result

def tokenize_function(examples,tokenizer):
    return tokenizer(examples, padding="max_length", truncation=True)

def Spliter(hate_target,neutral_target, tokenizer, test_size=0.2, random_state=42):
    # Load hate data and Assign label 1 to hate data
    hate_data = pd.read_csv(hate_target, header=None, names=['text'])
    hate_data_text = hate_data["text"].tolist()
    #Tokenizes before building batches
    #tokenized_hate_data = batch_tokenize(hate_data_text, tokenizer)
    #tokenized_hate_data = hate_data.map(tokenizer, batched=True)
    #hate_data['text'] = hate_data['text'].apply(lambda x: tokenize_function(x, tokenizer))
    #tokenized_hate_data['label'] = 1
    hate_data['label'] = 1

    # Load neutral data and Assign label 0 to neutral data
    neutral_data = pd.read_csv(neutral_target, header=None, names=['text'])
    #neutral_data_text = neutral_data["text"].tolist()
    # Tokenizes before building batches
    #tokenized_neutral_data = neutral_data.map(tokenizer, batched=True)
    #tokenized_neutral_data = batch_tokenize(neutral_data_text, tokenizer)
    #tokenized_neutral_data['label'] = 0
    #neutral_data['text'] = neutral_data['text'].apply(lambda x: tokenize_function(x, tokenizer))
    neutral_data['label'] = 0

    # Concatenate hate and neutral data
    data = pd.concat([hate_data, neutral_data], ignore_index=True)

    #Split the data into a test and training set
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data


def prepare_data_loader(train_data, test_data, batch_size):
    # Create custom datasets
    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    #train_dataset = train_data.shuffle(seed=42)
    #test_dataset = test_data.shuffle(seed=42)

    #train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader