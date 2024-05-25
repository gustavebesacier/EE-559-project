import os
import torch
from torch.optim import AdamW
import torch.optim as optim
import csv
from Models import create_student_model, Model_summary, create_teacher_model
from Datasets_Batches import Spliter_DataLoader
from Data_Handler import setup_data, full_data_generator, data_summary
from transformers import BertTokenizer, get_scheduler, get_linear_schedule_with_warmup
from Running import load_metrics, trainer_distiller, trainer, evaluate, WarmupThenCosineScheduler
from Octopus import input_parser


#OUR_TARGET = ["women", "jews", "asian", "black", "lgbtq", "latino", "muslim", "indigenous", "arab", "disabilities", "others"]
OUR_TARGET = ["jews","asian","muslim","disabilities"]
DEVICE = torch.device("cpu")
missing_data = False
save = False
gen_full_data = False
teacher_train = False
student_train = False
data_info = False
summary = False
inference = True
TARGET_SUBSET_DIC = {"jews":1600,"asian":810,"muslim":2800,"disabilities":630}


def main():
    # Generate the dictionary to find the file
    datasets = {
        target: {"hate": f"Data/hate_{target}.csv", "neutral": f"Data/neutral_{target}.csv"} for
        target in OUR_TARGET}


    if missing_data:
        setup_data(OUR_TARGET,datasets)

    if gen_full_data:
        full_data_generator(OUR_TARGET,datasets)

    if data_info:
        data_summary(OUR_TARGET, datasets)


    # get weight path for student and head

    weights_path_head, weights_paths_student = None, None
    if inference:
        weights_path_head = "logs_and_weights\head_path"
        weights_paths_student = {target: f'logs_and_weights/{target}_weights.pth' for target in OUR_TARGET}

    # Create the head_model
    head_model = create_head_model(weights_path_head)

    # TODO: 1)create create_head_model

    # Create the teacher_model
    teacher_model = create_teacher_model(weights_path_head)

    # define a dictionary linking a student model to a specific target
    target_models = {target: create_student_model(weights_path=weights_paths_student[target],
                                                  num_classes=2) for target in OUR_TARGET}

    # Create the tokenizer
    tokenizer = BertTokenizer.from_pretrained("hate_bert")

    # Return the final probability of hate_speech for given sentences (inference mode)
    if inference:
        input_parser(tokenizer, head_model, target_models, DEVICE, filename="sentences.txt")

    if summary:
        print("student model")
        Model_summary(target_models["women"],batch_size=16)
        print("teacher model")
        Model_summary(teacher_model, batch_size=16)

    #Train the teacher model
    if teacher_train:
        # Fetch the data and split it into test and training
        train_loader, test_loader = Spliter_DataLoader("hate_all.csv", "neutral_all.csv",
                                                       tokenizer, batch_size=16, test_size=0.2, random_state=38,
                                                       subset_size=10000)

        optimizer = AdamW(teacher_model.parameters(), lr=5e-5)
        num_epochs = 1
        num_training_steps = num_epochs * len(train_loader)

        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=2,
                                     num_training_steps=num_training_steps)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer(teacher_model, optimizer, num_training_steps, lr_scheduler, train_loader, num_epochs, device)
        metrics = evaluate(teacher_model, test_loader, device)
        print(metrics)
        torch.save(teacher_model.state_dict(), "teacher_weights.pth")

    # train student model
    if student_train:
        for target in OUR_TARGET:
            #generate the training and testing data paths
            hate_target = datasets[target]["hate"]
            neutral_target = datasets[target]["neutral"]

            #Fetch the data and split it into test and training
            train_loader, test_loader = Spliter_DataLoader(hate_target,neutral_target, tokenizer, batch_size=16,
                                                           test_size=0.2, random_state=38,
                                                           subset_size = TARGET_SUBSET_DIC[target])

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
                                                     alpha=alpha, T= temperature, tokenizer=tokenizer)

            print(train_loss_log, test_loss_log, distil_loss_log, train_metric_scores, test_metric_scores)

        # Save logs
        if save:
            log_file = os.path.join("logs_and_weights", f"{target}_log.txt")
            with open(log_file, 'w') as f:
                f.write(f"Train Metrics Log:\n{train_metric_scores}\n\nTest Metrics"
                        f" Log:\n{test_metric_scores}\nTrain losses Log:\n{train_loss_log}\nDistillation "
                        f"losses Log:\n{distil_loss_log}\nTest losses Log:\n{test_loss_log}")

            # Save model weights
            weights_file = os.path.join("logs_and_weights", f"{target}_weights.pth")
            torch.save(target_models[target].state_dict(), weights_file)


if __name__ == '__main__':
    main()
