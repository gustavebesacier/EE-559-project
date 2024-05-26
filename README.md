# EE-559-project

This is a project from Guillaume Ferrer, Clément Renard and Gustave Besacier, students from the course EE-559 Deep Learning from the Swiss Federal Institute of Technology Lausanne (EPFL).

## About the project 📈📊

- Goal: we aim at building a deep learning model for hate speech detection on the internet. This model is innovative by its architecture, as it consists in several (11) students models specialized in detection towards a specific minority group. 

- The full description of the project is available in a short paper, in the repository.


## Installation 💻
The code is optimized for Python 3.11.


### Library

The following library are used:
- Numpy
- Matplotlib
- PyTorch
- Transformers
- Tqdm
- Csv
- Scikit-Learn
- Pandas
- Scipy
- Time
- Json
- Os

## Files 📁

### Main files
- Data_Handler.py: handles all data from different sources.
- Datasets_Batches.py: prepares (tokenization, tensorization) data to input the different models.
- head_trainer.py: trains the head classifier.
- Models.py: creates the students and the teacher models, and computes metrics.
- Octopus.py: link between the head classifier and the student models.
- Running.py: performs the training and evaluation of student models, and inference. 

### Data
The data is in different formats (csv and txt), gathered from different datasets. More details and explanations are in the report. 


## Usage 🫳
The code can be downloaded on the GitHub repository together with the data. Usage is of a standard Python code.
Disclaimer: this repository contains potential harmful sentences towards minority groups. This has been used in a stricly academic purpose, and do not reflect the authors' opinion.

## Contact 📒

Guillaume Ferrer: guillaume.ferrer@epfl.ch
Clément Renard:   clement.renard@epfl.ch
Gustave Besacier  gustave.besacier@epfl.ch


## Acknowledgments 🤗

We also thank the EPFL, the EE-559 team.
