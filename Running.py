from tqdm.auto import tqdm
import torch
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_metrics(*metric_names):
    """

    Args:
        *metric_names:

    Returns:

    """
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = load_metric(metric_name)
    return metrics

def trainer(model, optimizer,nbr_steps, lr_scheduler, train_loader, nbr_epochs, device):
    """
    This is a simple train function given a model
    Main goal is to specialize pretrained model
    Args:
        model: model to train
        optimizer: optimizer
        nbr_steps: number of scheduler step
        lr_scheduler: method that updates the learning rate
        train_loader: train set
        nbr_epochs: nbr of epochs
        device:

    Returns: None

    """
    progress_bar = tqdm(range(nbr_steps))
    model.to(device)
    model.train()

    for epoch in range(nbr_epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()

            #clean optimizer
            optimizer.zero_grad()

            if step % 100 == 0 and step != 0:
                print(f"  Batch {step} of {len(train_loader)}. Loss: {loss.item():.4f}")

            progress_bar.update(1)


def evaluate(model, data_loader, device):
    """
    Evaluate the given model on the provided data loader.

    Args:
        model: The model to evaluate.
        data_loader: The DataLoader for the evaluation data.
        device: The device to run the model on (CPU or GPU).

    Returns:
        A dictionary containing the evaluation metrics.
    """
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics


def trainer_distiller(student_model, teacher_model, optimizer,nbr_steps, lr_scheduler, metrics, train_loader, test_loader, nbr_epochs, device, alpha, T, tokenizer):
    """
    Runs the training with distillation on the student model
    Args:
        student_model: the model that needs to be trained
        teacher_model: the trained model
        optimizer: optimizer
        nbr_steps: number of scheduler step
        lr_scheduler: method that updates the learning rate
        metrics: the type of test we want to eval our model
        train_loader: train Dataloader
        test_loader: test Dataloader
        nbr_epochs: number of epochs
        device:
        alpha: distillation parameter
        T: temperature
        tokenizer: method to tokenize

    Returns:

    """
    student_model.to(device)
    teacher_model.to(device)

    # Setup the model to avoid unnecessary computation
    student_model.train()
    teacher_model.eval()

    progress_bar = tqdm(range(nbr_steps))
    train_loss_log, test_loss_log, distil_loss_log = [], [], []

    # Create a dictionary to store lists of metric scores
    metric_name_list = [metric_name for metric_name, metric in metrics.items()]
    train_metric_scores = {metric_name: [] for metric_name in metric_name_list}
    test_metric_scores = {metric_name: [] for metric_name in metric_name_list}

    #define loss metric
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    for epoch in range(nbr_epochs):
        print("Epoch {0} of {1}".format(epoch, nbr_epochs))
        #Run diagnostics
        model_loss = 0
        distil_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)


            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

            # Perform the Kullback–Leibler divergence
            student_p = torch.softmax(student_outputs.logits / T, dim=-1)
            teacher_q = torch.softmax(teacher_outputs.logits / T, dim=-1)

            #Compute the KL divergence
            kl_divergence = kl_loss(student_p, teacher_q) * (T ** 2)

            # Compute the student loss and knowledge loss
            student_loss = student_outputs.loss
            model_loss += student_loss.item()

            # Calculate final loss
            loss = (1. - alpha) * student_loss + alpha * kl_divergence
            distil_loss += loss.item()

            #Update weights
            loss.backward()

            #update the optimizer
            optimizer.step()
            lr_scheduler.step()

            # clean optimizer
            optimizer.zero_grad()

            progress_bar.update(1)

            predictions = torch.argmax(student_outputs.logits, dim=-1)

            # Iterate over metrics to load batch to save metrics
            with torch.no_grad():
                for metric_name, metric in metrics.items():
                    metric.add_batch(predictions=predictions, references=labels)

        model_loss /= len(train_loader)
        train_loss_log.append(model_loss)

        distil_loss /= len(train_loader)
        distil_loss_log.append(distil_loss)

        # Iterate over metrics
        for metric_name, metric in metrics.items():
            # Compute metric score
            score = metric.compute()
            print(f"{metric_name.capitalize()}:", score)
            train_metric_scores[metric_name].append(score)

        #Evaluation on test set
        eval_loss = evaluation(student_model, metrics, test_loader, device, test_metric_scores)
        test_loss_log.append(eval_loss)

    return train_loss_log, test_loss_log, distil_loss_log, train_metric_scores, test_metric_scores

def evaluation(model, metrics, test_loader, device, test_metric_scores):
    model.eval()
    eval_loss = 0
    print("Evaluation on test set")
    for step, batch in enumerate(test_loader):
        input_ids, attention_mask, token_type_ids, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        eval_loss += outputs.loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        with torch.no_grad():
            for metric_name, metric in metrics.items():
                metric.add_batch(predictions=predictions, references=labels)

    eval_loss /= len(test_loader)

    # Iterate over metrics
    for metric_name, metric in metrics.items():
        # Compute metric score
        score = metric.compute()
        print(f"{metric_name.capitalize()}:", score)
        test_metric_scores[metric_name].append(score)

    return eval_loss

def inference(model, input, tokenizer, device):
    model.eval()
    tokens = tokenizer(input, return_tensors="pt", padding=True, truncation = True)
    tokens.to(device)

    #inference
    with torch.no_grad():
        logits = model(**tokens).logits

    #predict
    prediction = logits.argmax().item()

    return prediction

# Custom scheduler to combine warmup and cosine annealing
class WarmupThenCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
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