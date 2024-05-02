from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from Data_Handler import Batcher

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
    teacher_model.eval()  # Ensure teacher model is in evaluation mode
    return teacher_model



# Example usage:
num_classes = 2  # Assuming binary classification (hate vs. non-hate)
student_model = create_student_model(num_classes)

# Load the HateBERT model
hatebert_model = BertForSequenceClassification.from_pretrained("hate_bert")
tokenizer = BertTokenizer.from_pretrained("hate_bert")

# Print model architecture
print(hatebert_model)

text = "all asians make their kids take math lessons and learn karate we do not really need to say it, but asians are obviously trying to take over our country and we should try and prevent the evil yellow invasion because the chinese want to spread their asian evil across the world, asian men all have small peckers moms who are asian are known to smack their kids if they are not behaving if you have been driving lately, it is pretty obvious that the worst drivers are asian women"

text = "all asians make their kids take math lessons and learn karate\n- we do not really need to say it, but asians are obviously trying to take over our country\n- we should try and prevent the evil yellow invasion because the chinese want to spread their asian evil across the world\n- asian men all have small peckers\n- moms who are asian are known to smack their kids if they are not behaving\n- if you have been driving lately, it is pretty obvious that the worst drivers are asian women\n-"

text = "viewing an asian composer or performing musician only through sociological lenses are still relatively common in various countries, but times are changing\n- recently, there has been an increase in attacks against asian americans and we all need to speak out\n- we often refer to asians generally, but there are hundreds of asian cultures\n- through public awareness and education, views on asian americans can change for the better\n- how we talk about and recognize the importance of asian american history can influence how people think about asian americans\n- when people of asian descent are portrayed in the media, they are often shown as criminals, hackers or bad parents\n-"

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
    neutral_txt_file = "neutral_asian_1k.txt"
    hate_txt_file = "hate_asian_1k.txt"
    batches = Batcher(neutral_txt_file, hate_txt_file, batch_size=4)
    classify_batches(batches)