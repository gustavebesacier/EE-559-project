from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from Data_Handler import Batcher
from torchsummary import summary

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

def create_teacher_model():
    """
    Creates the teacher model
    :return: the teacher model
    """
    teacher_model = BertForSequenceClassification.from_pretrained("hate_bert")

    #Putting it in eval mode
    teacher_model.eval()
    return teacher_model

def Model_summary(model, input_size):
    """
    Returns the model summary as well as the size
    :param model: the model in question
    :param input_size: a tuple composed of (batch size, max sequence length)
    :return: None
    """
    summary(model, input_size=input_size)

# Example usage:
num_classes = 2  # Assuming binary classification (hate vs. non-hate)
student_model = create_student_model(num_classes)

# Load the HateBERT model
hatebert_model = BertForSequenceClassification.from_pretrained("hate_bert")
tokenizer = BertTokenizer.from_pretrained("hate_bert")

# Print model architecture
print(hatebert_model)

text = "you can’t run a society if many people are mentally disabled; those few who are not would be constantly at risk and constantly uncomfortable"
print(text)
# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")

# Perform inference
outputs = hatebert_model(**inputs)

print(outputs)

# Get predicted class
predicted_class = outputs.logits.argmax().item()

# Print predicted class
print("Predicted class:", predicted_class)

def classify_batches(batches):
    for batch in batches:
        texts, labels = zip(*batch)
        # Tokenize batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # Perform inference
        outputs = hatebert_model(**inputs)
        # Get predicted classes
        predicted_classes = outputs.logits.argmax(dim=1).tolist()
        # Print predicted classes
        for text, label, predicted_class in zip(texts, labels, predicted_classes):
            text = text.strip()  # Remove leading and trailing whitespace
            text = text.replace("\n", "")  # Remove newline character
            if label != predicted_class:
                print("Text:", text)
                print("Label:", label)
                print("Predicted class:", predicted_class)
                print()

def test_hateBERT():
    neutral_txt_file = "Data/neutral_asian_1k.txt"
    hate_txt_file = "Data/hate_asian_1k.txt"
    batches = Batcher(neutral_txt_file, hate_txt_file, batch_size=4)
    classify_batches(batches)