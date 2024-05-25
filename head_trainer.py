import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import csv
import os
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
# from main import OUR_TARGET

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



OUR_TARGET = ["women", "jews", "asian", "black", "lgbtq", "latino", "muslim", "indigenous", "arab", "disabilities", "others"]
MAPPING = {OUR_TARGET[i]: i for i in range(len(OUR_TARGET))}
INV_MAPPING = {v: k for k, v in MAPPING.items()}

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DIR = os.getcwd() + "/"
FILE_TRAIN = DIR + "full_target_id.csv"
FILE_TEST  = DIR + "full_target_id_test.csv"
EPOCHS = 5

class WarmupThenCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
        Custom learning rate scheduler that combines warmup and cosine annealing schedules.

        :param optimizer: The optimizer for which to schedule the learning rate.
        :param warmup_scheduler: The scheduler used for the warmup phase.
        :param cosine_scheduler: The scheduler used for the cosine annealing phase.
        :param num_warmup_steps: The number of steps for the warmup phase.
        """
    def __init__(self, optimizer, warmup_scheduler, cosine_scheduler, num_warmup_steps):
        self.warmup_scheduler = warmup_scheduler
        self.cosine_scheduler = cosine_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.step_count = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.step_count < self.num_warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.cosine_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.step_count < self.num_warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step(epoch)
        self.step_count += 1

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




def read_target_split(file):
    """convert the dataset into one list for text and one for labels"""
    data = pd.read_csv(file)
    data = clean_semicolumns(data)
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
    # Remove duplicates
    train = pd.read_csv(file_train)
    train = train.drop_duplicates()
    train = clean_semicolumns(train)
    train.to_csv(file_train)

    # Create test file
    data = train.sample(1000)

    data.to_csv(file_test, columns = ['target', 'text'], index = False)
    

def clean_semicolumns(data):
    for name in data.columns.tolist():
        if ":" in name:
            data = data.drop([name], axis =1)
    return data

def get_weight_classes(data):
    "Takes a dataframe and returns a list of the frequency of each target"
    value_counts = data['target'].value_counts()
    weights = [elem[1]/len(data) for elem in sorted(value_counts.items())]
    return weights


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

def model_training(model, train_dataset, test_dataset, epochs, optimization, criterion, metrics, lr_scheduler, device = DEVICE):
    """This function splits the data into dedicated dataloaders, then performs training on multiple epochs and tests
    Lots of piece of code are extracted from EE559 labs/exercises and were not created by us but developed by Idiap Research Institute."""

    model.to(device)

    model.train() # set the model in train mode

    # load data
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader =  DataLoader(test_dataset, batch_size=50, shuffle=True)

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
            # loss = outputs.loss

            loss.backward()
            optimization.step()
            lr_scheduler.step()

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
        epoch_eval_loss = 0
        epoch_eval_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

        for batch in tqdm(test_loader):
            with torch.no_grad():

                optimization.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)#.cpu()
                _, pred = torch.max(outputs.logits, 1)
                loss = outputs.loss

                pred_cpu = pred.cpu()
                labels_cpu = labels.cpu()

                for k in epoch_eval_metrics.keys():
                    epoch_eval_metrics[k] += metrics[k](pred_cpu, labels_cpu)
                epoch_eval_loss += loss.item()

        epoch_eval_loss /= len(test_loader)

        for k in epoch_eval_metrics.keys():
          epoch_eval_metrics[k] /= len(test_loader)

        eval_loss.append(epoch_eval_loss)
        eval_accuray.append(epoch_eval_metrics['ACC'])
        eval_f1.append(epoch_eval_metrics['F1-weighted'])

        print('eval Loss: {:.4f}, '.format(epoch_eval_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_eval_metrics[k]) for k in epoch_eval_metrics.keys()]), "\n")

        file_path = DIR + "model/log.csv"

        l = [
            [epoch, "train", "loss", epoch_loss], [epoch, "train", "acc", epoch_metrics['ACC'].item()], [epoch, "train", "f1", epoch_eval_metrics['F1-weighted'].item()],
            [epoch, "eval", "loss", epoch_eval_loss], [epoch, "eval", "acc", epoch_eval_metrics['ACC'].item()], [epoch, "eval", "f1", epoch_eval_metrics['F1-weighted'].item()]]
        all_metrics.append(l)

    with open(file_path, "w") as file:
        writer = csv.writer(file)
        for epoch in all_metrics:
            writer.writerows(epoch)


    return train_loss, train_accuray, train_f1, eval_loss, eval_accuray, eval_f1


def run_training(path_train = FILE_TRAIN, path_test = FILE_TEST, epochs= EPOCHS, save = True, show = False):
    """Prepare data, train the model and save weights."""

    # prepare_data(path_train, path_test)

    # Get the relative weight of each classes
    data = pd.read_csv(path_train)
    data = clean_semicolumns(data)
    weights_targets = get_weight_classes(data)
    weights_targets = torch.tensor(weights_targets)

    # Split the data
    train_texts, train_labels = read_target_split(path_train)
    test_texts, test_labels = read_target_split(path_test)
    results_models_weights_dir = DIR + 'model/'

    # Define tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Encode the data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Create instances of HATEDataset (gives all the attributes)
    train_dataset = HATEDataset(train_encodings, train_labels)
    test_dataset = HATEDataset(test_encodings, test_labels)

    # Model training
    criterion = nn.CrossEntropyLoss(weight=weights_targets)
    model_ = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(MAPPING))
    
    # load previous weights
    weights_path = results_models_weights_dir + 'weights_sentiment_analysis.pth'
    
    if weights_path is not None:
        # Load the state dictionary directly
        state_dict = torch.load(weights_path)
        model_.load_state_dict(state_dict)
    
    optimizer = AdamW(model_.parameters(), lr=5e-5)

    num_training_steps = epochs * len(train_dataset)
    T_0 = 1     # Number of epochs for the first restart
    T_mult = 2  # Increase in the cycles
    num_warmup_steps = 50

    # Create the linear warmup scheduler
    warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                       num_training_steps=num_training_steps)

    # Create the cosine annealing with warm restarts scheduler
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=5e-10)

    # Combine the warmup and cosine annealing schedulers
    lr_scheduler = WarmupThenCosineScheduler(optimizer, warmup_scheduler, cosine_scheduler, num_warmup_steps)

    metrics = {'ACC': acc, 'F1-weighted': f1}

    model_.to(DEVICE)

    train_loss, train_accuray, train_f1, eval_loss, eval_accuray, eval_f1 = model_training(model_, train_dataset, test_dataset, epochs, optimizer, criterion, metrics, lr_scheduler=lr_scheduler)

    torch.save(model_.state_dict(), weights_path)

    # get the metrics from the log file    
    train_loss, train_acc, train_f1, eval_loss, eval_acc, eval_f1 = get_values_log(DIR + "model/log.csv")
    
    # plot and save the result
    plot_metrics(train_loss, train_acc, train_f1, eval_loss, eval_acc, eval_f1, show=show, save=save)

    return train_loss, train_accuray, train_f1, eval_loss, eval_accuray, eval_f1, model_


def get_values_log(log_path):
    """Parse the log file to get the list of metrics"""

    data = pd.read_csv(log_path, names=['epoch', 'mode', 'metric', 'value'], header=None)

    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []
    train_f1 = []
    eval_f1 = []

    for _, row in data.iterrows():
        if row['mode'] == 'train':
            if row['metric'] == 'loss':
                train_loss.append(row['value'])
            elif row['metric'] == 'acc':
                train_acc.append(row['value'])
            elif row['metric'] == 'f1':
                train_f1.append(row['value'])
        if row['mode'] == 'eval':
            if row['metric'] == 'loss':
                eval_loss.append(row['value'])
            elif row['metric'] == 'acc':
                eval_acc.append(row['value'])
            elif row['metric'] == 'f1':
                eval_f1.append(row['value'])

    return train_loss, train_acc, train_f1, eval_loss, eval_acc, eval_f1


def plot_metrics(train_loss, train_acc, train_f1, eval_loss, eval_acc, eval_f1, show = False, save=True):
    """Plot the metrics if the training and testing."""

    epochs = [i+1 for i in range(len(train_acc))]

    # We want the 3 plots next to each other
    _, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plotting Loss
    axs[0].plot(epochs, train_loss, marker='o', label='Train Loss')
    axs[0].plot(epochs, eval_loss, marker='o', label='Eval Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting Accuracy
    axs[1].plot(epochs, train_acc, marker='o', label='Train Accuracy')
    axs[1].plot(epochs, eval_acc, marker='o', label='Eval Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].legend()
    axs[1].grid(True)

    # Plotting F1 Score
    axs[2].plot(epochs, train_f1, marker='o', label='Train F1 Score')
    axs[2].plot(epochs, eval_f1, marker='o', label='Eval F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1 Score')
    axs[2].set_title('F1 Score over Epochs')
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()

    if save:
        plt.savefig(DIR + "model/metrics")

    if show:
        plt.show()


if __name__ == "__main__":

    # small_test  = DIR + "small_target_id_test.csv"
    # small_train = DIR + "small_target_id.csv"
    
    # run_training(small_train, small_test, 2)


    FILE_TRAIN = "full_target.csv"
    FILE_TEST  = "full_target_test.csv"
    
    run_training(FILE_TRAIN, FILE_TEST, 2)