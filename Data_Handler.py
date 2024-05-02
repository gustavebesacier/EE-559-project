import random

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