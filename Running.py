from tqdm.auto import tqdm
import torch
from datasets import load_metric

def load_metrics(*metric_names):
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = load_metric(metric_name)
    return metrics

def trainer(model, optimizer,nbr_steps, lr_scheduler, train_loader, nbr_epochs, device):
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
    progress_bar = tqdm(range(nbr_steps))

    train_loss_log, test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]

    for epoch in range(nbr_epochs):
        print("Epoch {0} of {1}".format(epoch, nbr_epochs))
        #Run diagnostics
        current_loss = 0
        diagnostics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

        #Setup the model to avoid unnecessary computation
        student_model.train()
        teacher_model.eval()

        for batch_num, (tokens, labels) in tqdm(enumerate(train_loader)):
            #inputs = tokens.to(device)
            inputs = tokenizer.batch_encode_plus(tokens, return_tensors="pt", padding=True, truncation=True)
            #print(type(tokens))
            #print(tokens)
            targets = labels.to(device)
            #return None

            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs, labels=targets)
            student_outputs = student_model(**inputs, labels=targets)

            # Perform the Kullback–Leibler divergence
            student_p = torch.softmax(student_outputs.logits / T, dim=-1)
            teacher_q = torch.softmax(teacher_outputs.logits / T, dim=-1)

            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

            #Compute the KL divergence
            kl_divergence = kl_loss(student_p, teacher_q) * (T ** 2)

            # Compute the student loss and knowledge loss
            student_loss = student_outputs.loss

            # Calculate final loss
            loss = (1. - alpha) * student_loss + alpha * kl_divergence

            loss.backward()

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

        # Iterate over metrics
        for metric_name, metric in metrics.items():
            # Compute metric score
            score = metric.compute()
            print(f"{metric_name.capitalize()}:", score)

        #Evaluation on test set
        student_model.eval()

        for batch_num, (tokens, labels) in tqdm(enumerate(test_loader)):
            inputs = tokenizer.batch_encode_plus(tokens, return_tensors="pt", padding=True, truncation=True)
            targets = labels.to(device)
            with torch.no_grad():
                outputs = student_model(**inputs, labels=targets)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=targets)

        metric.compute()

        #plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, test_metrics_log)
