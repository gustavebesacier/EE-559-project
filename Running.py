from tqdm.auto import tqdm
import torch
from datasets import load_metric

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
    model.train()

    for epoch in range(nbr_epochs):
        for batch_num, (tokens, labels) in tqdm(enumerate(train_loader)):
            inputs = tokens.to(device)
            targets = labels.to(device)

            #assume batchmaker is right
            model_outputs = model(**inputs, labels=targets)

            loss = model_outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            #clean optimizer
            optimizer.zero_grad()

            progress_bar.update(1)

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
    progress_bar = tqdm(range(nbr_steps))
    train_loss_log, test_loss_log, distil_loss_log = [], [], []

    # Create a dictionary to store lists of metric scores
    metric_name_list = [metric_name for metric_name, metric in metrics.items()]
    train_metric_scores = {metric_name: [] for metric_name in metric_name_list}
    test_metric_scores = {metric_name: [] for metric_name in metric_name_list}


    for epoch in range(nbr_epochs):
        print("Epoch {0} of {1}".format(epoch, nbr_epochs))
        #Run diagnostics
        model_loss = 0
        distil_loss = 0

        #Setup the model to avoid unnecessary computation
        student_model.train()
        teacher_model.eval()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        for batch_num, (tokens, labels) in tqdm(enumerate(train_loader)):
            inputs = tokenizer.batch_encode_plus(tokens, return_tensors="pt", padding=True, truncation=True).to(device)
            targets = labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs, labels=targets)
            student_outputs = student_model(**inputs, labels=targets)

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
                    metric.add_batch(predictions=predictions, references=targets)

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
        eval_loss = evaluation(student_model, metrics, test_loader, device, tokenizer, test_metric_scores)
        test_loss_log.append(eval_loss)

    return train_loss_log, test_loss_log, distil_loss_log, train_metric_scores, test_metric_scores

def evaluation(model, metrics, test_loader, device, tokenizer, test_metric_scores):
    model.eval()
    eval_loss = 0
    print("Evaluation on test set")
    for batch_num, (tokens, labels) in tqdm(enumerate(test_loader)):
        inputs = tokenizer.batch_encode_plus(tokens, return_tensors="pt", padding=True, truncation=True)
        targets = labels.to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=targets)

        eval_loss += outputs.loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        with torch.no_grad():
            for metric_name, metric in metrics.items():
                metric.add_batch(predictions=predictions, references=targets)

    eval_loss /= len(test_loader)

    # Iterate over metrics
    for metric_name, metric in metrics.items():
        # Compute metric score
        score = metric.compute()
        print(f"{metric_name.capitalize()}:", score)
        test_metric_scores[metric_name].append(score)

    return eval_loss