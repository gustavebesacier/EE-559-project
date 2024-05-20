import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

def Spliter_DataLoader(hate_target,neutral_target, tokenizer, batch_size, test_size=0.2, random_state=38, subset_size = 3000):
    hate_data = pd.read_csv(hate_target, header=None, names=['text'])
    neutral_data = pd.read_csv(neutral_target, header=None, names=['text'])

    if subset_size:
        hate_data = hate_data.sample(subset_size)
        neutral_data = neutral_data.sample(subset_size)

    # Tokenize the positive and negative data
    tokenized_pos_data = hate_data['text'].apply(lambda x: tokenize_text(x, tokenizer))
    tokenized_neg_data = neutral_data['text'].apply(lambda x: tokenize_text(x, tokenizer))

    # Add labels and convert to a suitable format
    tokenized_pos_data = [{'input_ids': data['input_ids'], 'attention_mask': data['attention_mask'],
                           'token_type_ids': data['token_type_ids'], 'label': 1} for data in tokenized_pos_data]
    tokenized_neg_data = [{'input_ids': data['input_ids'], 'attention_mask': data['attention_mask'],
                           'token_type_ids': data['token_type_ids'], 'label': 0} for data in tokenized_neg_data]

    # Combine the positive and negative data
    combined_data = tokenized_pos_data + tokenized_neg_data

    # Convert to DataFrame for easy manipulation
    combined_df = pd.DataFrame(combined_data)

    # Split the data into training and test sets
    train_df, test_df = train_test_split(combined_df, test_size=test_size, random_state=random_state)

    # Convert the DataFrames to TensorDatasets
    train_dataset = df_to_tensors(train_df)
    test_dataset = df_to_tensors(test_df)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def tokenize_text(text, tokenizer):
    return tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

# Function to convert DataFrame rows to tensors
def df_to_tensors(df):
    input_ids = torch.cat(df['input_ids'].values.tolist(), dim=0)
    attention_mask = torch.cat(df['attention_mask'].values.tolist(), dim=0)
    token_type_ids = torch.cat(df['token_type_ids'].values.tolist(), dim=0)
    labels = torch.tensor(df['label'].values)
    return TensorDataset(input_ids, attention_mask, token_type_ids, labels)
