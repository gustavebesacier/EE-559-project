import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

MAPPING = {'middle_east': 0,'latino': 1,'chinese': 2,'muslim': 3,'bisexual': 4,'mexican': 5,'lgbtq': 6,'physical_disability': 7,'mental_disability': 8,'asian': 9,'women': 10,'jewish': 11,'immigrant': 12,'native_american': 13,'black': 14, 'trans':15}
MAPPING_INV = {k:v for (k,v) in enumerate(MAPPING)}
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class HATEDataset(torch.utils.data.Dataset):
    """Permits to have correctly composed datasets"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def model_prediction(model, tokenizer, prompt=None, device = DEVICE):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not prompt:
        prompt = input("Prompt? ")
    inputs = tokenizer(prompt, return_tensors="pt",  padding = True, truncation = True)
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(**inputs.to(device)).logits

    predicted_class_id = logits.argmax().item()
    print("Prompt: ", prompt)
    print(" - Predicted class id: ", predicted_class_id)
    print(" - Predicted category: ", MAPPING_INV[predicted_class_id])


def read_target_split(file):
    """convert the dataset into one list for text and one for labels"""
    data = pd.read_csv(file)
    texts = data.text.to_list()
    labels = data.target.replace(MAPPING).to_list()

    return texts, labels


def model_training(model, train_dataset, epochs, optimization, criterion, metrics, device = DEVICE):
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):

        epoch_loss = 0
        epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

        print(f"Epoch {epoch + 1} / {epochs}")

        for batch in tqdm(train_loader):
            optimization.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            with torch.no_grad():
                _, pred = torch.max(outputs.logits, 1)

            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimization.step()
            with torch.no_grad():
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += metrics[k](pred, labels)
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        for k in epoch_metrics.keys():
          epoch_metrics[k] /= len(train_loader)

        print('train Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]), "\n")

    return epoch_loss,  epoch_metrics

def f1(preds, target):
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    return accuracy_score(target, preds)


if __name__ == "__main__":

    train_texts, train_labels = read_target_split('dataset/no_tone_output.csv')
    test_texts, test_labels = read_target_split('dataset/no_tone_output_test.csv')

    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.05)
    
    # Define tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Encode the data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Create instances of HATEDataset (gives all the attributes)
    train_dataset = HATEDataset(train_encodings, train_labels)
    val_dataset = HATEDataset(val_encodings, val_labels)
    test_dataset = HATEDataset(test_encodings, test_labels)

    # Model training
    criterion = nn.CrossEntropyLoss()
    model_ = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(MAPPING))
    optim = AdamW(model_.parameters(), lr=5e-5)

    metrics = {'ACC': acc, 'F1-weighted': f1}

    loss, metric = model_training(model_, train_dataset, 1, optim, criterion, metrics)