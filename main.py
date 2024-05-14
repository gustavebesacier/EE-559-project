import torch
import torch.optim as optim
#from Student_model import create_student_model, create_teacher_model
from Models import create_student_model, create_teacher_model
from Training_Evaluation import train_cycle_with_distillation, f1, acc
from Data_Handler import prepare_data_loader, hateXplain_parser

hateXplain_parser()



"""
#test_hateBERT()

# Set up student and teacher models
num_classes = 2  # Assuming binary classification
student_model = create_student_model(num_classes)
teacher_model = create_teacher_model()

# Define optimizer, criterion, and device
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
metrics = {'ACC': acc, 'F1-weighted': f1}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data loaders
#Still need to be built
train_loader, test_loader = prepare_data_loader(train_data, test_data, batch_size=32)

# Train the student model with distillation
train_metrics_log, test_metrics_log = train_cycle_with_distillation(student_model, teacher_model, optimizer, criterion, metrics, train_loader, test_loader, n_epochs=20, device=device, alpha=0.25 , T=2)
"""