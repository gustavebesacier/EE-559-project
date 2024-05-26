from Running import inference, target_inference, target_classifier
import os
import csv
import json


def octopus(sentence, head_model, target_models, tokenizer,head_tokenizer, device):
    """
    Return the final probability computed for a single sentence.
    Args:
        sentence: The sentence to analyse
        head_model: Model use to infer targets
        target_models: Dict containing target student for each target,
        as an argument to easily create more or less target categories
        tokenizer: Tokenizer used for the sentence
        device: Device used for inference

    Returns: The probability the sentence consist of toxic speech

    """
    probabilities = target_inference(head_model, head_tokenizer, device, sentence)
    head_classifier = target_classifier(probabilities)
    output = 0
    for target in head_classifier:
        model_output = inference(model=target_models[target], input=sentence, tokenizer=tokenizer, device=device)
        print("using the model specified on",target," outputs", model_output)
        output += model_output*head_classifier[target]
    prediction = 1 if output > 0.5 else 0
    return output, prediction


def input_parser(tokenizer,head_tokenizer, head_model, target_models, device, filename):
    """
    Parse every line of the given file to analyze
    Args:
        tokenizer: Tokenizer used to tokenize the sentence
        head_model: model to select targets for a sentence
        target_models: Dict containing target student for each targets
        device: device to run the predictions
        filename: name.csv or name.txt file with a sentence every line to test for hate_speech

    Returns:
        Probability for each sentence (1-hatespeech, 0-neutral)
    """

    probabilities = []
    predictions = []
    extension = os.path.splitext(filename)[1].lower()
    if extension == '.csv':
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            sentences = csv.reader(csvfile)
            for sentence in sentences:
                output, prediction = octopus(sentence, head_model, target_models, tokenizer, head_tokenizer, device)
                probabilities.append(output)
                predictions.append(prediction)

    if extension == '.txt':
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()
                output, prediction = octopus(sentence, head_model, target_models, tokenizer, head_tokenizer, device)
                probabilities.append(output)
                predictions.append(prediction)

    with open("probabilities", 'w', encoding='utf-8') as outfile:
        json.dump(probabilities, outfile)

    print(f"the probability of each sentence to be hateful is {probabilities}")

    return predictions



