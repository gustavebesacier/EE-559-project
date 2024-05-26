import random
import json
import numpy as np
import pandas as pd
import os
from csv import writer
from tqdm import tqdm
import datasets
from pathlib import Path
import csv

# 18 targets
HATEXPLAIN_TARGET = ["African", "Arabs", "Asian", "Caucasian", "Hispanic", "Buddhism", "Christian", "Hindu", "Islam",
                     "Jewish",
                     "Men", "Women", "Heterosexual", "Gay", "Indigenous", "Refugee/Immigrant", "None", "Others"]

OUR_TARGET = ["women", "jews", "asian", "black", "lgbtq", "latino", "muslim", "indigenous", "arab",
              "disabilities", "others"]

DIC_TARGET = {
    'none': 'others',
    'african': 'black',
    'asian': 'asian',
    'caucasian': 'others',
    'women': 'women',
    'jewish': 'jews',
    'islam': 'muslim',
    'hispanic': 'latino',
    'indigenous': 'indigenous',
    'men': 'others',
    'christian': 'others',
    'heterosexual': 'others',
    'hindu': 'others',
    'buddhism': 'others',
    'bisexual': 'lgbtq',
    'chinese': 'asian',
    'black': 'black',
    'immigrant': 'others',
    'lgbt': 'lgbtq',
    'mental_disability': 'others',
    'mexican': 'others',
    'middle_east': 'arab',
    'muslim': 'muslim',
    'native_american': 'indigenous',
    'physical_disability': 'others',
    'trans': 'lgbtq',
    'lgbtq': 'lgbtq',
    'latino': 'latino'
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
                sentences = np.full((data_for_a_sentence_without_s.shape[0], 1), sentence)
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
        if brut[i] == "hatespeech":  # or brut[i] == "offensive":
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
                            if annotation[i] >= 1]).reshape(n_processed, 1)
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


def setup_data(targets, dataset_list):
    # Empty the folder
    clean_folder()

    # Create the empty files
    create_files(OUR_TARGET)

    hateXplain_parser()

    # Create the files using hateXplain dataset
    data_collection("dataset_hateXplain.csv")

    # Add data from Toxigen
    data_collection("dataset/output.csv")

    # HuggingFace data
    measuring_hate_speech_parser()

    for target in targets:
        # Clean the duplicates from the datasets
        hate_target = dataset_list[target]["hate"]
        neutral_target = dataset_list[target]["neutral"]
        remove_duplicates_in_place(hate_target)
        remove_duplicates_in_place(neutral_target)

    print("Tuto bene!!")


def measuring_hate_speech_file_builder(df_target, target):
    """
           Build two sort of file: toxic, with a hate_speech_score larger than 0.5 and no_toxic, <=0.5
           :param df_target: panda datafile, or a specific target
           :param target: string, the target from OUR_TARGET
           """
    # Toxic or not, starting from hate_speech_score larger than 0.5
    toxic = df_target[df_target['hate_speech_score'] > 0.5]
    non_toxic = df_target[df_target['hate_speech_score'] <= 0.5]

    # Only retain the sentences
    toxic_sentences = toxic[['text']]
    non_toxic_sentences = non_toxic[['text']]

    # Append toxic sentences to Data/hate_{target}.csv
    toxic_sentences.to_csv(f"Data/hate_{target}.csv", mode='a', index=False, header=False)

    # Append non-toxic sentences to Data/neutral_{target}.csv
    non_toxic_sentences.to_csv(f"Data/neutral_{target}.csv", mode='a', index=False, header=False)


def measuring_hate_speech_parser():
    """
        Load the measuring_hate_speech pd datafile, and map the correct column to our targets, call the function to
        create files in the desired format
        """
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
    df = dataset['train'].to_pandas()

    # 13 is hate speech score
    # 14 is text
    # 22 is first target: asian
    # Map every target with the right column of the file
    mapping = {
        OUR_TARGET[0]: [51],
        OUR_TARGET[1]: [35],
        OUR_TARGET[2]: [22],
        OUR_TARGET[3]: [23],
        OUR_TARGET[4]: [47, 48, 49, 50, 54, 55, 56],
        OUR_TARGET[5]: [24],
        OUR_TARGET[6]: [37],
        OUR_TARGET[7]: [26, 27],
        OUR_TARGET[8]: [25],
        OUR_TARGET[9]: [67, 68, 69, 70, 71, 72, 73, 74],
        OUR_TARGET[10]: [28, 29, 30, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 52, 53, 57, 58, 59, 60, 61,
                         62, 63, 65, 66]
    }

    for target in OUR_TARGET:
        for column_index in mapping[target]:
            measuring_hate_speech_file_builder(df[df.iloc[:, column_index] == True], target)


def remove_duplicates_in_place(file_path):
    # Load the dataset, specifying no header and treating each line as a separate row
    df = pd.read_csv(file_path, header=None)

    # Remove duplicates
    df_cleaned = df.drop_duplicates()

    # Overwrite the original file with the cleaned dataset
    df_cleaned.to_csv(file_path, index=False, header=False)


def full_data_generator(targets, data):
    # Initialize empty dataframes for concatenating the data
    all_hate_data = pd.DataFrame()
    all_neutral_data = pd.DataFrame()

    for target in targets:
        # generate the training and testing data paths
        hate_target = data[target]["hate"]
        neutral_target = data[target]["neutral"]

        # Read the data from each file
        hate_data = pd.read_csv(hate_target, header=None, names=['text'])
        neutral_data = pd.read_csv(neutral_target, header=None, names=['text'])

        # Concatenate the data
        all_hate_data = pd.concat([all_hate_data, hate_data], ignore_index=True)
        all_neutral_data = pd.concat([all_neutral_data, neutral_data], ignore_index=True)

        # Save the concatenated data to new CSV files
    all_hate_data.to_csv("hate_all.csv", index=False, header=False)
    all_neutral_data.to_csv("neutral_all.csv", index=False, header=False)

    remove_duplicates_in_place("hate_all.csv")
    remove_duplicates_in_place("neutral_all.csv")
    concatenate_csv("hate_all.csv", "neutral_all.csv", "all.csv")


def data_summary(targets, dataframes):
    for target in targets:
        # generate the training and testing data paths
        hate_target = dataframes[target]["hate"]
        neutral_target = dataframes[target]["neutral"]

        hate_data = pd.read_csv(hate_target, header=None, names=['text'])
        neutral_data = pd.read_csv(neutral_target, header=None, names=['text'])

        print(f"{target} has {len(hate_data)} hate sentences and {len(neutral_data)} neutral ones")

    hate_data = pd.read_csv("hate_all.csv", header=None, names=['text'])
    neutral_data = pd.read_csv("neutral_all.csv", header=None, names=['text'])

    print(f"Averall has {len(hate_data)} hate sentences and {len(neutral_data)} neutral ones")


def data_for_multilabel_classification(folder="dataset", hate_all_path="hate_all.csv",
                                       neutral_all_path="neutral_all.csv"):
    """
    Convert the data from the file dataset in a format adapted for multilabel classification ex: [0,1,0,1,1,0,0,...]

        Args:
            folder: name of the folder containing the datasets organized by hate/neutral_target.csv
            hate_all_path: path to all sentences of hate
            neutral_all_path: path to all neutral sentences

        Save:
            .csv file, with in each row:  sentence,targets. Where target is a list
            of size OUR_TARGET where every 1 in position i indicate that the sentence target the group OUR_TARGET[i]

    """

    # create the files to return
    with open("data_multilabel_head", 'w', newline='') as file:
        writer2 = csv.writer(file)
        writer2.writerow(['text', 'labels'])

    folder_path = Path(folder)

    # append the sentences and the targets
    with open("all.csv", 'r', encoding="utf8") as all:
        for s in all:
            s_to_compare = s.strip()
            targets = [0] * 11
            for file_path in folder_path.iterdir():
                with file_path.open('r', encoding="utf-8") as file:
                    reader = csv.reader(file)
                    for sentence in reader:
                        if sentence[0] == s_to_compare:
                            add_one(extract_target(file_path.name), targets)
            with open("data_multilabel_head", 'a', newline='', encoding="utf8") as file:
                writer2 = csv.writer(file)
                writer2.writerow([s_to_compare, targets])


def add_one(target, targets):
    """
    Takes a list of targets and update at the right position if adding target is needed
    Args:
        target: string,the specific target to add
        targets: list of binary int the list of targets for a sentence

    Returns: updated list of target

    """
    if target in OUR_TARGET:
        index = OUR_TARGET.index(target)
        targets[index] = 1
    return targets


def extract_target(filename):
    """
    Extracts targets from filename
    Args:
        filename: The directory to extract the name
    Returns:
        targets
    """
    target = filename[:-4].split('_')[1]

    return target


def concatenate_csv(file1, file2, output_file):
    """
    Concatenates two CSV files

    Args:
     file1: Path to CSV file.
     file2: Path to CSV file.
     output_file: Path to the output file to save the concatenated CSV data.
    """

    with open(file1, 'r', newline='', encoding="utf-8") as input_csv1:
        reader1 = csv.reader(input_csv1)
        data1 = list(reader1)

    with open(file2, 'r', newline='', encoding="utf-8") as input_csv2:
        reader2 = csv.reader(input_csv2)
        data2 = list(reader2)

    data1.extend(data2)

    with open(output_file, 'w', newline='', encoding="utf-8") as output_csv:
        writer3 = csv.writer(output_csv)
        writer3.writerows(data1)


def test_file(filename="Data/demo_train.csv"):
    # Initialize lists to store labels and text
    labels = []
    text_lines = []

    # Open and read the CSV file
    with open(filename, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Append the number to the labels list
            labels.append(int(row[0]))
            # Append the text to the text_lines list
            text_lines.append(row[1])

    # Convert the labels list to a NumPy array
    labels_array = np.array(labels)

    # Save the text lines to a text file
    with open('output.txt', 'w', encoding='utf-8') as txtfile:
        for line in text_lines:
            txtfile.write(line + '\n')

    # Save the labels array to a .npy file if needed
    np.save('labels.npy', labels_array)