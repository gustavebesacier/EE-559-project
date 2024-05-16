import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
def load_data(hate_file, neutral_file):
    # Load hate data
    hate_data = pd.read_csv(hate_file, header=None, names=['text'])
    hate_data['label'] = 1  # Assign label 1 to hate data

    # Load neutral data
    neutral_data = pd.read_csv(neutral_file, header=None, names=['text'])
    neutral_data['label'] = 0  # Assign label 0 to neutral data

    # Concatenate hate and neutral data
    data = pd.concat([hate_data, neutral_data], ignore_index=True)

    return data

def Spliter(hate_target,neutral_target, test_size=0.2, random_state=42):
    # Load hate data and Assign label 1 to hate data
    hate_data = pd.read_csv(hate_target, header=None, names=['text'])
    hate_data['label'] = 1

    # Load neutral data and Assign label 0 to neutral data
    neutral_data = pd.read_csv(neutral_target, header=None, names=['text'])
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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader