from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, accuracy_score

#THIS WHOLE FILE IS NOTE CODED BY US AND IS DIRECTLY TAKEN FROM THE COURSE CODE

def train_epoch_with_distillation(student_model, teacher_model, tokenizer, optimizer, criterion, metrics, train_loader, device,
                                  alpha=0.25, T=2):
    '''
    alpha: a hyperparameter.
           alpha - a weight assigned to the extra objective of teacher-student loss.
           1-alpha - a weight assigned to cross-entropy between student outouts and target.
           Tuning these weights pushes the network towards optimizing for either of two objectives.

    T: Temperature controls the smoothness of the output distributions.
       Larger T leads to smoother distributions, thus smaller probabilities get a larger boost.

    '''

    epoch_loss = 0  # Log loss statistics during training
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    # metric computation should be organised the same way as in train_epoch

    # remember that a teacher model should not be trained and should be in inference mode

    # YOUR CODE HERE
    # prepare the models
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    student_model.train()
    teacher_model.eval()

    # Run on the datasets (inspired from top part)
    for batch_num, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
        print(x_batch)
        print(type(x_batch))
        print(y_batch)
        print(type(y_batch))
        inputs = tokenizer.batch_encode_plus(x_batch, return_tensors="pt", padding=True, truncation=True)
        print(inputs)
        print(type(inputs))
        data = inputs.to(device)
        target = y_batch.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_preds = teacher_model(**data).logits
        student_preds = student_model(**data).logits
        
        print(student_preds)
        print("target")
        print(target)

        # loss on students
        loss = criterion(student_preds, target)

        # Perform the Kullback–Leibler divergence
        student_p = torch.softmax(student_preds / T, dim=1)
        teacher_q = torch.softmax(teacher_preds / T, dim=1)

        # Compute the KL divergence
        kl_div = torch.sum(teacher_q * (torch.log(teacher_q) - torch.log(student_p)), dim=1)

        # Take the mean over the batch
        kl_div = torch.mean(kl_div)

        total_loss = (1 - alpha) * loss + alpha * kl_div * T ** 2

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        # Convert probabilities to class predictions
        _, predicted_labels = torch.max(student_preds, 1)

        with torch.no_grad():
            for k in epoch_metrics.keys():
                epoch_metrics[k] += metrics[k](predicted_labels,
                                               target)  # Use predicted labels instead of probabilities

    epoch_loss /= len(train_loader)

    for metric_name, metric_value in epoch_metrics.items():
        epoch_metrics[metric_name] /= len(train_loader)

    print('train Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

    return epoch_loss, epoch_metrics


# Modification of train_cycle function to include the teacher model
def train_cycle_with_distillation(student_model, teacher_model, tokenizer, optimizer, criterion, metrics, train_loader,
                                  test_loader, n_epochs, device, alpha=0.25, T=2):
    train_loss_log, test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]

    for epoch in range(n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs))
        train_loss, train_metrics = train_epoch_with_distillation(student_model, teacher_model, tokenizer, optimizer, criterion,
                                                                  metrics, train_loader, device, alpha=alpha, T=T)
        # evaluation is the same as for normal simple training.
        # Although, take into account that the loss computed for evaluation is a simple CrossEntropy loss and it will differ significantly from training loss
        test_loss, test_metrics = evaluate(student_model, criterion, metrics, test_loader, device)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(metrics_names, test_metrics_log, test_metrics)

        plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, test_metrics_log)

    return train_metrics_log, test_metrics_log

def plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
    fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

    ax[0].plot(train_loss, c='blue', label='train')
    ax[0].plot(test_loss, c='orange', label='test')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    for i in range(len(metrics_names)):
        ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
        ax[i + 1].plot(test_metrics_logs[i], c='orange', label='test')
        ax[i + 1].set_title(metrics_names[i])
        ax[i + 1].set_xlabel('epoch')
        ax[i + 1].legend()

    plt.show()

def evaluate(model, criterion, metrics, test_loader, device):
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    model.eval()  # set a model in a evaluation mode

    for batch_num, (x_batch, y_batch) in enumerate(test_loader):
        with torch.no_grad():  # we don't compute gradients here
            data = x_batch.to(device)
            target = y_batch.to(device)

            # forward
            outputs = model(data)

            # compute loss
            loss = criterion(outputs, target)

            # compute predictions
            _, preds = torch.max(outputs, 1)

            # compute metrics
            for k in epoch_metrics.keys():
                epoch_metrics[k] += metrics[k](preds, target)

            # log loss statistics
            epoch_loss += loss.item()

    epoch_loss /= len(test_loader)

    for k in epoch_metrics.keys():
        epoch_metrics[k] /= len(test_loader)

    print('eval Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

    return epoch_loss, epoch_metrics

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

def f1(preds, target):
    return f1_score(target.detach().cpu(), preds.detach().cpu(), average='macro')

def acc(preds, target):
    return accuracy_score(target.detach().cpu(), preds.detach().cpu())