from transformers import BertForSequenceClassification, BertConfig
from torchsummary import summary
import torch
import os
import time

def create_student_model(num_classes = 2, transformer_size = 256, nbr_layers = 4, nbr_heads = 4):
    """
    Creates a student model
    :param num_classes: different number of labels
    :param transformer_size: size of the transformer
    :param nbr_layers: nbr of hidden layers
    :param nbr_heads: number of attention heads
    :return: a new student model
    """
    #Configure the student model
    config = BertConfig(
        num_labels=num_classes,
        hidden_size=transformer_size,
        num_hidden_layers=nbr_layers,
        num_attention_heads=nbr_heads
    )

    #Create the model using previous config
    student_model = BertForSequenceClassification(config)

    return student_model

def create_teacher_model(weights_path = None):
    """
    Creates the teacher model
    :return: the teacher model
    """
    teacher_model = BertForSequenceClassification.from_pretrained("hate_bert")

    if weights_path is not None:
        weights_pretrained = torch.load(weights_path)
        teacher_model.load_state_dict(weights_pretrained['model_state_dict'])

    return teacher_model

def Model_summary(model, batch_size, trials = 20):
    """
    Returns the model summary as well as the size
    :param model: the model in question
    :param input_size: a tuple composed of (batch size, max sequence length)
    :return: None
    """
    # Get the maximum sequence length from the model's configuration
    max_sequence_length = model.config.max_position_embeddings
    print(f"Maximum sequence length from model configuration: {max_sequence_length}")
    input_size = (batch_size, max_sequence_length)
    summary(model, input_size=input_size)
    get_model_size(model, "temp_weights.pth")
    num_attention_heads = model.config.num_attention_heads
    print(f"Number of attention heads: {num_attention_heads}")
    timer = 0
    for i in range(trials):
        timer += get_model_inference_time(model, input_size)
    timer /= trials
    print('model average inference time in seconds:', timer)

def get_model_size(model, temp_file='temp_model.pth'):
    """
    Save the model's state dictionary to a temporary file to calculate its size
    :param model: the model in question
    :param temp_file: temporary file to save the model
    :return: model size in MB
    """
    # Save the model's state_dict to a temporary file
    torch.save(model.state_dict(), temp_file)

    # Get the file size in bytes
    size_in_bytes = os.path.getsize(temp_file)

    # Convert bytes to megabytes
    size_in_mb = size_in_bytes / (1024 * 1024)

    # Remove the temporary file
    os.remove(temp_file)

    print(f"Model size: {size_in_mb:.6f} MB")

    return None

def get_model_inference_time(model, input_size):
    """
        Measure the inference time of a given model on CPU with dummy input data.

        :param model: The model for which to measure inference time.
        :param input_size: The size of the dummy input data as a tuple.
        :return: The inference time in seconds.
        """
    model.to(torch.device("cpu"))
    model.eval()

    # Generate dummy input data
    input_ids = torch.randint(0, model.config.vocab_size, input_size, dtype=torch.long)
    attention_mask = torch.ones(input_size,dtype=torch.long)
    token_type_ids = torch.zeros(input_size, dtype=torch.long)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }

    # Measure inference time
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()

    assert outputs.logits.shape[0] == 16
    return end - start