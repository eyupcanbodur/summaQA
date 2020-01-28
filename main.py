import os
import sys
import json
from datetime import datetime
from simpletransformers.question_answering import QuestionAnsweringModel

'''
    Setup
    1 - Install Anaconda or Miniconda Package Manager from here.
    2 - Create a new virtual environment and install packages.
        conda create -n simpletransformers python pandas tqdm
        conda activate simpletransformers
        conda install pytorch cpuonly -c pytorch
        conda install -c anaconda scipy
        conda install -c anaconda scikit-learn
        pip install transformers
        pip install seqeval
        pip install tensorboardx
    3- Install simpletransformers.
        pip install simpletransformers

    Note: Check your environment before running with conda info --envs. (* means that its your active env) 
'''


# Read JSON file as a dictionary, then return it
def takeDataFromSquadDataset(path):
    jsonFile = {}
    with open(path, 'r') as f:
        jsonFile = json.load(f)
    train_data = [item for topic in jsonFile['data'] for item in topic['paragraphs'] ]

    return train_data

'''
    Evaluate the given test_data from squad like format file. 
'''
def evaluate(model, test_data_path):
    dev_data = takeDataFromSquadDataset(test_data_path)
    # Predict the giveen test data file
    preds = model.predict(dev_data)
    # Write results into a json file
    os.makedirs('results', exist_ok=True)
    submission = {pred['id']: pred['answer'] for pred in preds}

    with open('results/submission .json', 'w') as f:
        json.dump(submission, f)



train_args = {
    'fp16': False,
    # 'learning_rate': 3e-5,
    # 'num_train_epochs': 2,
    # 'max_seq_length': 384,
    # 'doc_stride': 128,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    # 'train_batch_size': 2,
    # 'gradient_accumulation_steps': 8,
}

model = QuestionAnsweringModel('bert', 'bert-base-cased', args=train_args, use_cuda=False)
train_data = takeDataFromSquadDataset(sys.argv[1])
model.train_model(train_data)
# We will use dev-v2.json data (4 mb) from Squad dataset.
evaluate(model, sys.argv[2])

