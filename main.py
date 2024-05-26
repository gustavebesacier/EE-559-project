import os
import torch
from torch.optim import AdamW
import torch.optim as optim
import csv
from Models import create_student_model, Model_summary, create_teacher_model
from Datasets_Batches import Spliter_DataLoader
from Data_Handler import setup_data, full_data_generator, data_summary, test_file
from transformers import (BertTokenizer, get_scheduler, get_linear_schedule_with_warmup,
                          DistilBertForSequenceClassification, DistilBertTokenizerFast)
from Running import load_metrics, trainer_distiller, trainer, evaluate, WarmupThenCosineScheduler
from Octopus import input_parser
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from head_trainer import run_training


OUR_TARGET = ["women", "jews", "asian", "black", "lgbtq", "latino", "muslim", "indigenous", "arab", "disabilities", "others"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
missing_data = False
save = False
gen_full_data = False
head_train = False
teacher_train = False
student_train = False
data_info = False
summary = False
inference = True
hyperparameter = False
#alpha, temperature, lr
hyper_list = {"hyper0": [0.1, 0.7, 0.00005],"hyper1": [0.3, 0.7, 0.00005], "hyper2": [0.5, 0.7, 0.00005],
              "hyper3": [0.7, 0.7, 0.00005], "hyper4": [0.9, 0.7,0.00005], "hyper5": [0.2, 0.7, 0.00005],
              "hyper6": [0.4, 0.7, 0.00005],"hyper7": [0.6, 0.7, 0.00005],"hyper8": [0.8, 0.7, 0.00005]}

def main():
    # Generate the dictionary to find the file
    datasets = {
        target: {"hate": f"Data/hate_{target}.csv", "neutral": f"Data/neutral_{target}.csv"} for
        target in OUR_TARGET}

    if missing_data:
        setup_data(OUR_TARGET, datasets)

    if gen_full_data:
        full_data_generator(OUR_TARGET, datasets)

    if data_info:
        data_summary(OUR_TARGET, datasets)

    weights_paths_student = {target: f'logs_and_weights/{target}_weights.pth' for target in OUR_TARGET}

    # Create the students model
    target_models = {target: create_student_model(weights_path=weights_paths_student[target],
                                                  num_classes=2) for target in OUR_TARGET}

    # Create the tokenizer
    tokenizer = BertTokenizer.from_pretrained("hate_bert")

    # Save the final probability of hate_speech for given sentences (inference mode) in a JSON file
    if inference:
        weights_path_head = "logs_and_weights/weights_first_head.pth"
        # Create the head model
        head_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                         num_labels=len(OUR_TARGET))
        head_model.load_state_dict(
            torch.load('logs_and_weights/weights_first_head.pth', map_location=DEVICE))
        head_model.eval()
        # Define tokenizer for head (not optimal but short on time)
        head_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        test_file("Data/demo_train.csv")

        labels = np.load('labels.npy')

        predictions = input_parser(tokenizer,head_tokenizer, head_model, target_models, DEVICE, filename="output.txt")

        pred_numpy = np.array(predictions)

        # Calculate accuracy
        accuracy = accuracy_score(labels, pred_numpy)

        # Calculate F1 score
        f1 = f1_score(labels, pred_numpy)

        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')

    if summary:
        print("student model")
        #Model_summary(target_models["women"],batch_size=16)
        print("teacher model")
        #Model_summary(teacher_model, batch_size=16)
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(MAPPING))
        Model_summary(model, batch_size=16, classifier = True)

    if head_train:
        FILE_TRAIN = "Data/full_target_id.csv"
        FILE_TEST = "Data/full_target_id_test.csv"

        run_training(FILE_TRAIN, FILE_TEST, 5)

    #Train the teacher model
    if teacher_train:
        # Create the teacher_model
        #teacher_model = create_teacher_model("logs_and_weights/weights_first_head")
        teacher_model = create_teacher_model()
        if summary:
            print("teacher model")
            Model_summary(teacher_model, batch_size=16)
        # Fetch the data and split it into test and training
        train_loader, test_loader = Spliter_DataLoader("Data/hate_all.csv", "Data/neutral_all.csv",
                                                       tokenizer, batch_size=16, test_size=0.2, random_state=38)

        optimizer = AdamW(teacher_model.parameters(), lr=5e-5)
        num_epochs = 1
        num_training_steps = num_epochs * len(train_loader)

        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=2,
                                     num_training_steps=num_training_steps)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer(teacher_model, optimizer, num_training_steps, lr_scheduler, train_loader, num_epochs, device)
        metrics = evaluate(teacher_model, test_loader, device)
        print(metrics)
        torch.save(teacher_model.state_dict(), "logs_and_weights/teacher_weights.pth")

    # train student model
    if student_train:
        # Create the teacher_model
        teacher_model = create_teacher_model("logs_and_weights/teacher_weights.pth")
        if summary:
            print("teacher model")
            Model_summary(teacher_model, batch_size=16)
        for target in OUR_TARGET:
            #generate the training and testing data paths
            hate_target = datasets[target]["hate"]
            neutral_target = datasets[target]["neutral"]

            #Fetch the data and split it into test and training
            train_loader, test_loader = Spliter_DataLoader(hate_target,neutral_target, tokenizer, batch_size=16,
                                                           test_size=0.2, random_state=38)

            # Define optimizer, criterion, and device
            optimizer = AdamW(target_models[target].parameters(), lr=0.001)
            metrics = load_metrics("accuracy", "f1")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            num_epochs = 11
            alpha = 0.25
            temperature = 2
            num_training_steps = num_epochs * len(train_loader)
            T_0 = 1  # Number of epochs for the first restart
            T_mult = 2  # Increase in the cycles
            num_warmup_steps = 50

            # Create the scheduler
            #lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1, num_training_steps=num_training_steps)
            #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
            #lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer = optimizer, num_warmup_steps =  100, num_training_steps = num_training_steps, num_cycles=3)

            # Create the linear warmup scheduler
            warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                               num_training_steps=num_training_steps)

            # Create the cosine annealing with warm restarts scheduler
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min= 5e-10)

            # Combine the warmup and cosine annealing schedulers
            lr_scheduler = WarmupThenCosineScheduler(
                optimizer,
                warmup_scheduler,
                cosine_scheduler,
                num_warmup_steps
            )

            #if train:
                #trainer(teacher_model, optimizer, nbr_steps=num_training_steps, lr_scheduler=lr_scheduler, train_loader=, nbr_epochs, device, tokenizer)

            # Train the student model with distillation
            print("Start training for ",target,"dataset")
            (train_loss_log, test_loss_log, distil_loss_log, train_metric_scores,
             test_metric_scores) = trainer_distiller(student_model=target_models[target], teacher_model=teacher_model,
                        optimizer=optimizer, nbr_steps=num_training_steps, lr_scheduler=lr_scheduler, metrics=metrics,
                        train_loader=train_loader, test_loader=test_loader, nbr_epochs=num_epochs, device=device,
                                                     alpha=alpha, T= temperature)

            print(train_loss_log, test_loss_log, distil_loss_log, train_metric_scores, test_metric_scores)

        # Save logs
        if save:
            log_file = os.path.join("logs_and_weights", f"{target}_log.txt")
            with open(log_file, 'w') as f:
                f.write(
                    f"Train Metrics Log:\n{train_metric_scores}\n\nTest Metrics Log:\n{test_metric_scores}\nTrain losses Log:\n{train_loss_log}\nDistillation losses Log:\n{distil_loss_log}\nTest losses Log:\n{test_loss_log}")

            # Save model weights
            weights_file = os.path.join("logs_and_weights", f"{target}_weights.pth")
            torch.save(target_models[target].state_dict(), weights_file)

    if hyperparameter:
        # Create the teacher_model
        teacher_model = create_teacher_model("logs_and_weights/teacher_weights.pth")
        for hyper in hyper_list:
            alpha, temperature, lr = hyper_list[hyper]
            print(f"{hyper}: alpha={alpha}, temperature={temperature}, learning_rate={lr}")

            # generate the training and testing data paths
            hate_target = datasets["muslim"]["hate"]
            neutral_target = datasets["muslim"]["neutral"]

            # Fetch the data and split it into test and training
            train_loader, test_loader = Spliter_DataLoader(hate_target, neutral_target, tokenizer, batch_size=16,
                                                           test_size=0.2, random_state=38)

            # Define optimizer, criterion, and device
            optimizer = AdamW(target_models["muslim"].parameters(), lr=lr)
            metrics = load_metrics("accuracy", "f1")
            device = torch.device("cpu")

            num_epochs = 7
            alpha = alpha
            temperature = temperature
            num_training_steps = num_epochs * len(train_loader)
            T_0 = 1  # Number of epochs for the first restart
            T_mult = 2  # Increase in the cycles
            num_warmup_steps = 50

            # Create the linear warmup scheduler
            warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                               num_training_steps=num_training_steps)

            # Create the cosine annealing with warm restarts scheduler
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult,
                                                                              eta_min=5e-10)

            # Combine the warmup and cosine annealing schedulers
            lr_scheduler = WarmupThenCosineScheduler(optimizer, warmup_scheduler, cosine_scheduler, num_warmup_steps)

            # Train the student model with distillation
            print("Start training for ", "muslim", "dataset")
            train_loss_log, test_loss_log, distil_loss_log, train_metric_scores, test_metric_scores = trainer_distiller(
                student_model=target_models["muslim"], teacher_model=teacher_model, optimizer=optimizer,
                nbr_steps=num_training_steps, lr_scheduler=lr_scheduler, metrics=metrics, train_loader=train_loader,
                test_loader=test_loader, nbr_epochs=num_epochs, device=device, alpha=alpha, T=temperature)

            print(train_loss_log, test_loss_log, distil_loss_log, train_metric_scores, test_metric_scores)

            # Save logs
            if save:
                log_file = os.path.join("logs_and_weights_hyper", f"muslim{hyper}_log.txt")
                with open(log_file, 'w') as f:
                    f.write(
                        f"Train Metrics Log:\n{train_metric_scores}\n\nTest Metrics Log:\n{test_metric_scores}\nTrain losses Log:\n{train_loss_log}\nDistillation losses Log:\n{distil_loss_log}\nTest losses Log:\n{test_loss_log}")

                # Save model weights
                weights_file = os.path.join("logs_and_weights_hyper", f"muslim{hyper}_weights.pth")
                torch.save(target_models["muslim"].state_dict(), weights_file)

if __name__ == '__main__':
    main()
