# SummaQA : Experiment on a QuestionAnsweringModel

Experiment on Information Retrieval with summarization.

### Prerequisites and Suggested About Environment

Cuda is a must for our codebase. But you can change freely from the code "use_cuda=True" part to False.

1 - Install Anaconda or Miniconda Package Manager from here.  
2 - Create a new virtual environment and install packages.

- conda create - n simpletransformers python pandas tqdm
  conda activate simpletransformers
- If using cuda (is a must for our codebase):

  - conda install pytorch cudatoolkit=10.0 - c pytorch

- else:
  - conda install pytorch cpuonly - c pytorch + conda install - c anaconda scipy + conda install - c anaconda scikit-learn + pip install transformers + pip install seqeval + pip install tensorboardx

3 - Install simpletransformers.

- pip install simpletransformers  
  Note: Check your environment before running with conda info - -envs. (\* means that its yo
  ur active env)

### How to Run ?

Model that we are using is, takeing a previously model or train a model that using Bert (bert base cased). Also it won't extract the features from Bert every time you run the code. It's caching them both training a model steps and Bert's feature extraction.So do not delete them for the sake of the speed.

We did all the operations on Google Cloud. With using a VM that has 1 GPU,16 Gb Nvdia Tesla V100, , 15 Gb RAM (Intel Skylake). 100 GB storage and with using Ubuntu 18.4.

#### Summarization the Real Dataset

1 - Download any Squad format like dataset.
2 - run with ./summarizer.py path_to_dataset name_of_the_new_dataset

#### Main

You can always get help or how to install instructions with  
./main.py help  
./main.py setup

1 - Download any Squad format like dataset. For both training and for evaluation. Or you can manually split and creating a new evaluation set from training set. We're not supporting splitting training dataset.

2 - You can run in two ways.

- 1 - Training dataset and Evaluation Dataset. In this option, you will be train the model with training dataset. Then code will evaluate the given dataset and give the predictions of that evaluation dataset. Model that trained will be save into "outputs" and evaluation results will be saved into results/timestamp_of_that_time.json. Training time was 50 minutes with official Squad's trainv2.0.json dataset on a machine that we mentioned.

  - /main.py not-trained path_to_training_data path_to_evaluation_data

* 2- With model that previously trained and Evaluation Dataset. Just give the path of the saved model folder.
  - ./main.py with-trained path_to_trained_model path_to_evaluation datas
