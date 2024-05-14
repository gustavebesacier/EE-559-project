import random
import json
import numpy as np
import pandas as pd


#18 targets
HATEXPLAIN_TARGET = ["African","Arabs","Asian","Caucasian","Hispanic","Buddhism","Christian","Hindu","Islam","Jewish",
                     "Men","Women","Heterosexual","Gay","Indigenous","Refugee/Immigrant","None","Others"]

N_TARGET_XPLAIN = 18

def read_text_file(filename):
    """
    This function reads a .txt file and splits it using the \n- delimiter
    This function is specialized for text coming from Toxigen datasets
    :param filename: name of .txt file
    :return: list of phrases
    """
    with open(filename, 'r') as file:
        phrases = file.read().split('\n-')
    # Remove empty strings
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    return phrases


def Batcher(neutral_txt_file, hate_txt_file, batch_size):
    """
    Generate mixed batches of neutral and hate texts from text files with specified batch size.

    Args:
    neutral_txt_file (str): Path to the neutral text file.
    hate_txt_file (str): Path to the hate text file.
    batch_size (int): Number of samples per batch.

    Returns:
    list of tuples: Mixed batches with individual labels for each phrase.
    """
    # Read text files
    neutral_texts = read_text_file(neutral_txt_file)
    hate_texts = read_text_file(hate_txt_file)

    mixed_batches = []
    neutral_batch = []
    hate_batch = []

    # Shuffle the datasets to mix samples
    random.shuffle(neutral_texts)
    random.shuffle(hate_texts)

    # Iterate over the datasets and create mixed batches
    for i in range(0, max(len(neutral_texts), len(hate_texts)), batch_size):
        neutral_batch = neutral_texts[i:i + batch_size]
        hate_batch = hate_texts[i:i + batch_size]

        # Make sure both batches have the same length
        neutral_batch = neutral_batch[:len(hate_batch)]
        hate_batch = hate_batch[:len(neutral_batch)]

        # Assign labels: 0 for neutral and 1 for hate
        mixed_batch = [(text, 0) for text in neutral_batch] + [(text, 1) for text in hate_batch]

        # Shuffle the mixed batch
        random.shuffle(mixed_batch)

        # Append the mixed batch to the list
        mixed_batches.append(mixed_batch)

    return mixed_batches

def prepare_data_loader(train_data, test_data, batch_size):
    #TODO Build DataLoader and find train_set and test_set (must be built)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def hateXplain_parser(filename="dataset_hateXplain.json"):
    """
    Parse the hateXplain_dataset and for each entry: save the three annotation and the tokens in a matrix where
    each line is one data element, the first column are the label, next 3 the target, and the last one the
    hateful speech.
    :param filename: name of  file to parse (.JSON)
        """

    data_matrix = np.array([['toxicity', 'target', 'sentence']])

    with open(filename, 'r') as file:
        data = json.load(file)

        for entry_key, entry_data in data.items():
            labels = [annotator['label'] for annotator in entry_data['annotators']]
            targets = [annotator['target'] for annotator in entry_data['annotators']]
            targets_flat = [item for row in targets for item in row]
            tokens = entry_data['post_tokens']
            sentence = [" ".join(tokens)]

            data_for_a_sentence_without_s = hateXplain_builder(labels + targets_flat)
            if data_for_a_sentence_without_s.size != 0:
                sentences = np.full((data_for_a_sentence_without_s.shape[0],1), sentence)
                data_for_a_sentence = np.hstack((data_for_a_sentence_without_s, sentences))
                data_matrix = np.vstack((data_matrix, data_for_a_sentence))

    df = pd.DataFrame(data_matrix)
    df.to_csv('dataset_hateXplain.csv', index=False)


def hateXplain_builder(brut):
    """
    Take an entry of the dataset hateXplain and process it to choose a unique label and a list of targets.
    :param brut: a list of string coming from the hateXplain dataset
    :return: ndarray = shape (number of targets found, 2) where for a line: the first element is the label
    (1: toxic, 0:non-toxic), the second is a target from hateXplain target
    """

    # select majority label of the three annotator, toxic: 1, non-toxic: 0, need two annotator that label as either
    # hatespeech or offensive
    toxic = 0
    toxic_count = 0
    for i in range(0, 3):
        if brut[i] == "hatespeech" or brut[i] == "offensive":
            toxic_count += 1
    if toxic_count >= 2:
        toxic = 1

    # select the target of the post, need at least two annotators that agree on a target,
    # if multiple targets are retained, they become each a datapoint (same sentence and label, different target)
    annotation = np.zeros(N_TARGET_XPLAIN)
    for j in range(0, N_TARGET_XPLAIN):
        for i in range(3, len(brut)):
            if brut[i] == HATEXPLAIN_TARGET[j]:
                annotation[j] += 1

    n_processed = sum(1 for val in annotation if val >= 2)
    processed_matrix = np.full((n_processed, 1), toxic)
    targets = np.array([HATEXPLAIN_TARGET[i] for i in range(N_TARGET_XPLAIN) if annotation[i] >= 2]).reshape(n_processed,1)
    final_matrix = np.hstack((processed_matrix, targets))

    return final_matrix




