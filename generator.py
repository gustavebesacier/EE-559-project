import csv
import pandas as pd
import os

pd.set_option('future.no_silent_downcasting', True)

def get_tones_targets(folder="./prompts"):
    """Return 2 lists for a given folder: one contains the kinds of tones, the second contains the targets"""
    list_file = os.scandir(folder)

    tone_list = list()
    target_list = list()
    for file in list_file:
        elements = file.name.split("_")     # scrap parts of the name
        tone = elements[0]                  # tone is the first element
        target = "_".join(elements[1:-1])   # allows to keep several elements (ex: "native" and "american")
        if tone not in tone_list:
            tone_list.append(tone)
        if target not in target_list:
            target_list.append(target)

    return tone_list, target_list

    


def add_entries(folder_prompts="./prompts", output="dataset/output.csv"):
    
    """Creates a csv file with all prompts"""

    tone_list, target_list = get_tones_targets(folder=folder_prompts) # get the lists of tones and targets
    with open(output, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)                       # create csv writer
        writer.writerow(["tone", "target", "text"])     # header of the file

        for tone in tone_list:                              # for each tone, each target:
            for target in target_list:
                if  target == "trans" and tone == "neutral":# there is no neutral + trans prompt
                    pass
                else:
                    path = f"prompts/{tone}_{target}_1k.txt"
                    with open(path) as f:                    
                        # open the txt file with prompts then perform some cleaning 
                        for line in f:
                            line = line.split("\\n")
                            for i in line:
                                i=i.replace("\\n- ", "\n- ").replace("\\n-","").replace("-","")
                                if i != "\n" and i !="":
                                    # if the line contains words, write it in the csv file
                                    writer.writerow([tone, target, str(i.strip())])

                                    
def clean_entries(file):
    """Deletes all the duplicated entries"""
    df = pd.read_csv(file)
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    df.to_csv(file, index=False)

def no_tone_csv(file):
    """
        Remove the 'tone' column from a CSV file and save the result to a new CSV file.

        :param file: The path to the input CSV file.
        """
    df = pd.read_csv(file)#.dropna()
    df.drop(columns=['tone'], inplace=True)
    df.to_csv("dataset/no_tone_output.csv", index=False)

def target_to_nb(file):
    """
       Map target categories in a CSV file to numerical values and save the result to a new CSV file.

       :param file: The path to the input CSV file.
       """
    mapping = {'middle_east': 0,'latino': 1,'chinese': 2,'muslim': 3,'bisexual': 4,'mexican': 5,'lgbtq': 6,
               'physical_disability': 7,'mental_disability': 8,'asian': 9,'women': 10,'jewish':
                   11,'immigrant': 12,'native_american': 13,'black': 14, 'trans':15}
    df = pd.read_csv(file)
    df["target"] = df["target"].replace(mapping)
    df.to_csv("dataset/no_tone_nbtarget_output.csv", index=False)
    
    
if __name__ == "__main__":
    
    print(os.getcwd())

    # add_entries("./prompts", "dataset/output.csv")
    # clean_entries("dataset/output.csv")
    # no_tone_csv("dataset/output.csv")
    target_to_nb("dataset/no_tone_output.csv")
    clean_entries("dataset/no_tone_nbtarget_output.csv")
    print("Files were created")