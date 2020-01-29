import os
import sys
import json
from enum import Enum
from datetime import datetime
from simpletransformers.question_answering import QuestionAnsweringModel


class ModelType(Enum):
    TRAINED = "with-trained"
    NOT_TRAINED = "not-trained"


class Params(Enum):
    TRAINING_DATA_PATH = "path_to_training_data"
    TRAINED_MODEL_PATH = "path_to_trained_model"
    EVALUATION_DATA_PATH = "path_to_evaluation_data"


# Global model type. It has to be TRAINED or NOT_TRAINED
modelType = ""
params = {}


def getDateTime():
    return datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

# Read JSON file as a dictionary, then return it


def takeDataFromSquadDataset(path):
    jsonFile = {}
    with open(path, 'r') as f:
        jsonFile = json.load(f)
    train_data = [item for topic in jsonFile['data']
                  for item in topic['paragraphs']]

    return train_data


'''
    Get QuestionAnsweringModel with default bert-base-case or previously trained saved model.
'''


def getModel():
    model = ""
    train_args = get_train_args()
    if(modelType == ModelType.TRAINED):
        model = QuestionAnsweringModel(
            'bert', params[Params.TRAINED_MODEL_PATH], args=train_args, use_cuda=False)
    if(modelType == ModelType.NOT_TRAINED):
        model = QuestionAnsweringModel(
            'bert', 'bert-base-cased', args=train_args, use_cuda=False)
    if(model == ""):
        print("Error on getting the model")
        exit(1)

    return model


'''
    Train given model with given_data.
    If it's been already trained, skip this part.
'''


def trainModel(model):

    if(modelType == ModelType.TRAINED):
        print("Already trained model.")
    if(modelType == ModelType.NOT_TRAINED):
        # Read the given SQUAD file format
        train_data = takeDataFromSquadDataset(
            params[Params.TRAINING_DATA_PATH])
        model.train_model(train_data)


'''
    Evaluate the given test_data from squad like format file.
'''


def evaluate(model):
    dev_data = takeDataFromSquadDataset(params[Params.EVALUATION_DATA_PATH])
    # Predict the giveen test data file
    preds = model.predict(dev_data)
    # Write results into a json file
    os.makedirs('results', exist_ok=True)
    submission = {pred['id']: pred['answer'] for pred in preds}

    created_file_path = 'results/submission-' + getDateTime() + '.json'
    with open(created_file_path, 'w') as f:
        json.dump(submission, f)

    print("Evaluation results finished.\nResults are saved into : " + created_file_path)


'''
    Prints out help to the console
'''


def getHelp():
    print("You can run eith er with:")
    print(ModelType.TRAINED.value + " " + Params.TRAINED_MODEL_PATH.value +
          " " + Params.EVALUATION_DATA_PATH.value)
    print("or:")
    print(ModelType.NOT_TRAINED.value + " " + Params.TRAINING_DATA_PATH.value +
          " " + Params.EVALUATION_DATA_PATH.value)


'''
    Prints out setup to the console
'''


def getSetup():
    print("1 - Install Anaconda or Miniconda Package Manager from here.\n2 - Create a new virtual environment and install packages.\nconda create - n simpletransformers python pandas tqdm\nconda activate simpletransformers\nIf using cuda: \n\tconda install pytorch cudatoolkit=10.0 - c pytorch\nelse: \nconda install pytorch cpuonly - c pytorch\nconda install - c anaconda scipy\nconda install - c anaconda scikit-learn\npip install transformers\npip install seqeval\npip install tensorboardx\n3 - Install simpletransformers.\npip install simpletransformers\nNote: Check your environment before running with conda info - -envs. (* means that its your active env)\n")


'''
    Read and validate given parameters.
'''


def read_and_check_params():
    global modelType
    if(len(sys.argv) != 4):
        if(len(sys.argv) == 2 and sys.argv[1] == "help"):
            getHelp()
            exit(1)
        if(len(sys.argv) == 2 and sys.argv[1] == "setup"):
            getSetup()
            exit(1)
        print("Error: Some parameters are missing.")
        exit(1)
    # Check for model type
    if(sys.argv[1] == ModelType.TRAINED.value or sys.argv[1] == ModelType.NOT_TRAINED.value):
        modelType = ModelType(sys.argv[1])
    else:
        print("You need to give a parameter: with-trained | not-trained")
        exit(1)

    # Set training data path or trained model path
    if(modelType == ModelType.NOT_TRAINED):
        params[Params.TRAINING_DATA_PATH] = sys.argv[2]
    elif(modelType == ModelType.TRAINED):
        params[Params.TRAINED_MODEL_PATH] = sys.argv[2]

    params[Params.EVALUATION_DATA_PATH] = sys.argv[3]


def get_train_args():
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


def run():
    read_and_check_params()
    model = getModel()
    trainModel(model)
    evaluate(model)


run()
