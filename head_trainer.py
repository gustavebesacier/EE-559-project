import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import csv
import os
import matplotlib.pyplot as plt
from main import OUR_TARGET

MAPPING = {OUR_TARGET[i]: i for i in range(len(OUR_TARGET))}
INV_MAPPING = {v: k for k, v in MAPPING.items()}

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DIR = os.getcwd() + "/"


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
    """Use of the model to detect the target from a sentence.

    Args:
        model: model with pre-loaded weights
        tokenizer: words tokenizer
        prompt: prompt to the model, if None it asks user to write.
        device: Defaults to DEVICE.
    """
    if not prompt:
        prompt = input("Prompt? ")
    inputs = tokenizer(prompt, return_tensors="pt",  padding = True, truncation = True)
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(**inputs.to(device)).logits

    predicted_class_id = logits.argmax().item()
    print("Prompt: ", prompt)
    print(" - Predicted class id: ", predicted_class_id)
    print(" - Predicted category: ", INV_MAPPING[predicted_class_id])


def read_target_split(file):
    """convert the dataset into one list for text and one for labels"""
    data = pd.read_csv(file)
    texts = data.text.to_list()
    labels = data.target.replace(MAPPING).to_list()

    return texts, labels

def prepare_data(file_train, file_test):
    """Gathers all the prompts from the various files in a file for training, and a file for testing"""
    
    with open(file_train, "w") as csvfile:
        # set the header to ['target', 'text']
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['target', 'text'])

    with open(file_train, "a"):
        for file in os.scandir(DIR + "Data"):
            # loop over all the files from the folder (one per target/tone)
            try:
              if file.name.split(".")[1] == 'csv':
                  # only uses csv files - in case of error
                  df = pd.read_csv(file, header = None, names = ['text'])
                  target = file.name.split("_")[1].split(".")[0] # scrap the target from the file name
                  if target == 'other':
                      target = 'others'
                  df['target'] = MAPPING[target] # set the id of the target in the file

                  df = df[['target', 'text']]
                  df.to_csv(file_train, mode='a', header=False, index=False) # save to csv
            except:
              pass

    # Create test file
    data = pd.read_csv(file_train).sample(1000)
    data.to_csv(file_test, columns = ['target', 'text'], index = False)


def f1(preds, target):
    """
        Compute the F1 score for the given predictions and targets using macro averaging.

        :param preds: The predicted labels.
        :param target: The true labels.
        :return: The macro-averaged F1 score.
        """
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    """
        Compute the accuracy score for the given predictions and targets.

        :param preds: The predicted labels.
        :param target: The true labels.
        :return: The accuracy score.
        """
    return accuracy_score(target, preds)

def model_training(model, train_dataset, test_dataset, epochs, optimization, criterion, metrics, device = DEVICE):
    """This function splits the data into dedicated dataloaders, then performs training on multiple epochs and tests
    Lots of piece of code are extracted from EE559 labs/exercises and were not created by us but developed by Idiap Research Institute."""

    model.to(device)

    model.train() # set the model in train mode

    # load data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader =  DataLoader(test_dataset, batch_size=16, shuffle=True)

    # prepare lists for exporting the metrics
    train_loss = list()
    train_accuray = list()
    train_f1 = list()
    eval_loss = list()
    eval_accuray = list()
    eval_f1 = list()

    all_metrics = list()

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1} / {epochs}")

        ## Do the training
        epoch_loss = 0
        epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

        for batch in tqdm(train_loader):
            optimization.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)#.cpu()

            with torch.no_grad():
                _, pred = torch.max(outputs.logits, 1)

            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimization.step()

            with torch.no_grad():
                # save the metrics
                for k in epoch_metrics.keys():
                    pred_cpu = pred.cpu()
                    labels_cpu = labels.cpu()
                    epoch_metrics[k] += metrics[k](pred_cpu, labels_cpu)

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        epoch_loss = epoch_loss

        for k in epoch_metrics.keys():
          epoch_metrics[k] /= len(train_loader)

        train_loss.append(epoch_loss)
        train_accuray.append(epoch_metrics['ACC'])
        train_f1.append(epoch_metrics['F1-weighted'])

        print('train Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]), "\n")

        # Evaluate the model
        
        model.eval()
        epoch_loss = 0
        epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

        for batch in tqdm(test_loader):
            with torch.no_grad():

                optimization.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)#.cpu()
                _, pred = torch.max(outputs.logits, 1)

                pred_cpu = pred.cpu()
                labels_cpu = labels.cpu()

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += metrics[k](pred_cpu, labels_cpu)
                epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        for k in epoch_metrics.keys():
          epoch_metrics[k] /= len(train_loader)

        eval_loss.append(epoch_loss)
        eval_accuray.append(epoch_metrics['ACC'])
        eval_f1.append(epoch_metrics['F1-weighted'])

        print('eval Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]), "\n")

        file_path = DIR + "model/log.csv"

        # with open(file_path, "w") as file: # export the metrics in a csv file
        #   writer = csv.writer(file)
        #   writer.writerow(["epoch", "type", "metric", "value"])

        # with open(file_path, "a", newline="") as file:
        #     writer = csv.writer(file)

        #     l = [
        #         [epoch, "train", "loss", train_loss[0]], [epoch, "train", "acc", train_accuray[0].item()], [epoch, "train", "f1", train_f1[0].item()],
        #         [epoch, "eval", "loss", eval_loss[0]], [epoch, "eval", "acc", eval_accuray[0].item()], [epoch, "eval", "f1", eval_f1[0].item()]]
            
        #     for item in l:
        #         writer.writerow(item)
            
        #     file.close()

        l = [
            [epoch, "train", "loss", train_loss[0]], [epoch, "train", "acc", train_accuray[0].item()], [epoch, "train", "f1", train_f1[0].item()],
            [epoch, "eval", "loss", eval_loss[0]], [epoch, "eval", "acc", eval_accuray[0].item()], [epoch, "eval", "f1", eval_f1[0].item()]]
        all_metrics.append(l)

    with open(file_path, "w") as file:
        writer = csv.writer(file)
        for epoch in all_metrics:
            writer.writerows(epoch)


    return train_loss, train_accuray, train_f1, eval_loss, eval_accuray, eval_f1


if __name__ == "__main__":
    FILE_TRAIN = DIR + "full_target_id.csv"
    FILE_TEST  = DIR + "full_target_id_test.csv"

    # Create a single file with all sentences
    # prepare_data(FILE_TRAIN, FILE_TEST)


    train_texts, train_labels = read_target_split(FILE_TRAIN)
    test_texts, test_labels = read_target_split(FILE_TRAIN)
    results_models_weights_dir = DIR + 'model/'
    EPOCHS = 5

    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.10)

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

    model_.to(DEVICE)

    train_loss, train_accuray, train_f1, eval_loss, eval_accuray, eval_f1 = model_training(model_, train_dataset, val_dataset, EPOCHS, optim, criterion, metrics)

    torch.save(model_.state_dict(), results_models_weights_dir + 'weights_sentiment_analysis.pth')