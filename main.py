import os
import torch
import torch.optim as optim
from Models import create_student_model, create_teacher_model
from Training_Evaluation import train_cycle_with_distillation, f1, acc
from Datasets_Batches import train_test_split, prepare_data_loader

OUR_TARGET = ["women", "jews", "asian", "black", "lgbtq", "latino", "muslim", "indigenous", "arab", "others", "disabilities"]

def main():
    #define a dictionary linking a student model to a specific target
    target_models = {target: create_student_model(num_classes=2) for target in OUR_TARGET}

    # Define the directory path where the CSV files will be stored
    csv_directory = "Data"

    # Generate the dictionary to find the file
    datasets = {
        target: {"hate": f"{csv_directory}/hate_{target}.csv", "neutral": f"{csv_directory}/neutral_{target}.csv"} for
        target in OUR_TARGET}

    #Create the teacher_model
    teacher_model = create_teacher_model()

    # train student model
    for target in OUR_TARGET:
        #generate the training and testing data paths
        hate_target = datasets[target]["hate"]
        neutral_target = datasets[target]["neutral"]

        #Fetch the data and split it into test and training
        train_data, test_data = train_test_split(hate_target,neutral_target, test_size=0.2, random_state=42)

        # Prepare data loaders
        train_loader, test_loader = prepare_data_loader(train_data, test_data, batch_size=32)

        # Define optimizer, criterion, and device
        optimizer = optim.Adam(target_models[target].parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        metrics = {'ACC': acc, 'F1-weighted': f1}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train the student model with distillation
        train_metrics_log, test_metrics_log = train_cycle_with_distillation(target_models[target], teacher_model, optimizer,
                                                                            criterion, metrics, train_loader,
                                                                            test_loader, n_epochs=20, device=device,
                                                                            alpha=0.25, T=2)

        # Save logs
        log_file = os.path.join("logs_and_weights", f"{target}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"Train Metrics Log:\n{train_metrics_log}\n\nTest Metrics Log:\n{test_metrics_log}")

        # Save model weights
        weights_file = os.path.join("logs_and_weights", f"{target}_weights.pth")
        torch.save(target_models[target].state_dict(), weights_file)


if __name__ == '__main__':
    main()
