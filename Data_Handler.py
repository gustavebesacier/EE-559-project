import random
import json
import numpy as np
import pandas as pd
import os
from csv import writer
from tqdm import tqdm


#18 targets
HATEXPLAIN_TARGET = ["African","Arabs","Asian","Caucasian","Hispanic","Buddhism","Christian","Hindu","Islam","Jewish",
                     "Men","Women","Heterosexual","Gay","Indigenous","Refugee/Immigrant","None","Others"]

OUR_TARGET = ["women", "jews", "asian", "black", "lgbtq", "latino", "muslim", "indigenous", "arab", "others",
              "disabilities"]

DIC_TARGET = {
    'none':                 'others',
    'african':              'black',
    'asian':                'asian',
    'caucasian':            'others',
    'women':                'women',
    'jewish':               'jews',
    'islam':                'muslim',
    'hispanic':             'latino',
    'indigenous':           'indigenous',
    'men':                  'others',
    'christian':            'others',
    'heterosexual':         'others',
    'hindu':                'others',
    'buddhism':             'others',
    'bisexual':             'lgbtq',
    'chinese':              'asian',
    'black':                'black',
    'immigrant':            'other',
    'lgbt':                 'lgbtq',
    'mental_disability':    'other',
    'mexican':              'other',
    'middle_east':          'arab',
    'muslim':               'muslim',
    'native_american':      'indigenous',
    'physical_disability':  'other',
    'trans':                'lgbtq',
    'lgbtq':                'lgbtq',
    'latino':               'latino'
    }

TARGET_CONVERTER = {1: 'hate', 0: 'neutral'}

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

    # select the target of the post. If toxic = 1: need at least two annotators that agree on a target.
    # If toxic = 0: need only one annotator that agree on a target. if multiple targets are retained,
    # they become each a datapoint (same sentence and label, different target)
    annotation = np.zeros(N_TARGET_XPLAIN)
    for j in range(0, N_TARGET_XPLAIN):
        for i in range(3, len(brut)):
            if brut[i] == HATEXPLAIN_TARGET[j]:
                annotation[j] += 1

    if toxic == 0:
        n_processed = sum(1 for val in annotation if val >= 1)
        processed_matrix = np.full((n_processed, 1), toxic)
        targets = np.array([HATEXPLAIN_TARGET[i] for i in range(N_TARGET_XPLAIN)
                            if annotation[i] >= 1]).reshape(n_processed,1)
    else:
        n_processed = sum(1 for val in annotation if val >= 2)
        processed_matrix = np.full((n_processed, 1), toxic)
        targets = np.array([HATEXPLAIN_TARGET[i] for i in range(N_TARGET_XPLAIN)
                            if annotation[i] >= 2]).reshape(n_processed, 1)

    final_matrix = np.hstack((processed_matrix, targets))

    return final_matrix

def create_files(list_target):
    """For each target and tone, creates an empty file if not already existing or only open it"""
    for target in list_target:
        for tone in ['hate', 'neutral']:
            with open(f'Data/{tone}_{target}.csv', 'w') as _:
                pass

def clean_folder():
    """Clear the folder 'Data'."""
    with os.scandir("Data/") as entries:
        for entry in entries:
            if entry.is_file():
                os.unlink(entry.path)
        print("All files from 'Data' deleted successfully.")

def write_entry(tone, target, entry):
    """Given a tone (hate/neutral)"""
    with open(f'Data/{tone}_{target}.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow([entry])
        f.close()

def assign_target(tone, target, entry):
    """
    Takes the tone, target and entry from the line of the file
    Associates the target as written on the file to one category, according to the mapping of DIC_TARGET
    """
    write_entry(tone, DIC_TARGET[target], entry)

def data_collection(source="dataset_hateXplain.csv"):

    if source == "dataset_hateXplain.csv":
        with open("dataset_hateXplain.csv") as f:
            lines = f.readlines()[2:]
            for line in tqdm(lines):
                entry = line.split(",")
                tone, target, entry = entry[0], entry[1].lower(), entry[2]
                # use the mapping from TARGET_CONVERTER to turn the 0/1 to neutral/hate
                assign_target(TARGET_CONVERTER[int(tone)], target, entry.strip())
    
    else:
        with open(source) as f:
            lines = f.readlines()[1:]
            for line in tqdm(lines):
                entry = line.split(",")
                tone, target, entry = entry[0], entry[1].lower(), entry[2]
                assign_target(tone, target, entry.strip())


if __name__ == '__main__':
    # Empty the folder
    clean_folder()

    # Create the empty files
    create_files(OUR_TARGET)

    # Create the files using hateXplain dataset
    data_collection("dataset_hateXplain.csv")

    # Add data from Toxigen
    data_collection("dataset/output.csv")

    print("Tuto bene!!")