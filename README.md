# SciARK

SciARK is a dataset for *Argumentation Mining*.
It is a novel STI (science, technology and innovation)-driven multidisciplinary 
dataset of annotated abstracts from scientific literature that relates with
the [Sustainable Development Goals (SDGs)](https://sdgs.un.org/) of the
United Nations.

The abstracts in our dataset are related with six of the 17 SDGs and are
annotated for their *Argumentative Units*, i.e., sentences in which the 
authors state their *Claims* and the *Evidence* to support them.

Along with the dataset we provide the code described in our [paper](https://aclanthology.org/2021.argmining-1.10/). Specifically
the
* EDA baseline,
* (Sci)BERT only,
* BiLSTM - BiLSTM, and
* (Sci)BERT - BiLST.

If you use our dataset please cite us as:
```
@inproceedings{fergadis-etal-2021-argumentation,
    title = "Argumentation Mining in Scientific Literature for Sustainable Development",
    author = "Fergadis, Aris  and
      Pappas, Dimitris  and
      Karamolegkou, Antonia  and
      Papageorgiou, Haris",
    booktitle = "Proceedings of the 8th Workshop on Argument Mining",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.argmining-1.10",
    pages = "100--111",
}
```

## Usage
### EDA Baseline
```bash
python baseline.py
```

### (Sci)BERT as Sentence Encoder
```
usage: scibert_only.py [-h] [-e SENTENCE_ENCODER] [-s SAVE_DIR]
[-t [TRAIN_SET [TRAIN_SET ...]]]
[-d [DEVELOPMENT_SET [DEVELOPMENT_SET ...]]]
[-x [TEST_SET [TEST_SET ...]]]

optional arguments:
-h, --help            show this help message and exit
-e SENTENCE_ENCODER, --sentence-encoder SENTENCE_ENCODER
A huggingface 🤗 model to use as a sentence encoder.
-s SAVE_DIR, --save-dir SAVE_DIR
The directory name for the saved model.
-t [TRAIN_SET [TRAIN_SET ...]], --train-set [TRAIN_SET [TRAIN_SET ...]]
A list of datasets to use as a training set.
-d [DEVELOPMENT_SET [DEVELOPMENT_SET ...]], --development-set [DEVELOPMENT_SET [DEVELOPMENT_SET ...]]
A list of datasets to use as a development set.
-x [TEST_SET [TEST_SET ...]], --test-set [TEST_SET [TEST_SET ...]]
A list of datasets to use as a test set.
```

The following example will use the SciBERT as sentence encoder and will 
split the SciARK dataset into the default values (60% train, 20% dev, 20% test):
```bash
python scibert_only.py -e allenai/scibert_scivocab_uncased -t dataset/SciARK.json
```

* To use *BERT* model as *Sentence Encoder* pass `bert-based-uncased` in the
  `--sentence-encoder` argument.
* To use *SciBERT* model as *Sentence Encoder* pass
  `allenai/scibert_scivocab_uncased` in the `--sentence-encoder` argument.

Default values for arguments and other hyper-parameters are in the [config()](https://github.com/afergadis/SciARK/blob/bb2931a8c221bf77c34afa82199228e889d395c6/scibert_only.py#L17)
function.

### BiLSTM - BiLSTM
```
usage: bilstm_bilstm.py [-h] [-e EMBEDDINGS_FILE] [-s SAVE_MODEL]
[-t [TRAIN_SET [TRAIN_SET ...]]]
[-d [DEVELOPMENT_SET [DEVELOPMENT_SET ...]]]
[-x [TEST_SET [TEST_SET ...]]]

optional arguments:
-h, --help            show this help message and exit
-e EMBEDDINGS_FILE, --embeddings-file EMBEDDINGS_FILE
A file with word embeddings in GloVe format.
-s SAVE_MODEL, --save-model SAVE_MODEL
The file name for the saved model. The file name may
have a path and *should* have `.h5` extension.
-t [TRAIN_SET [TRAIN_SET ...]], --train-set [TRAIN_SET [TRAIN_SET ...]]
A list of datasets to use as a training set.
-d [DEVELOPMENT_SET [DEVELOPMENT_SET ...]], --development-set [DEVELOPMENT_SET [DEVELOPMENT_SET ...]]
A list of datasets to use as a development set.
-x [TEST_SET [TEST_SET ...]], --test-set [TEST_SET [TEST_SET ...]]
A list of datasets to use as a test set.
```

The following example will use the Glove.6B.200d for word embeddings and will 
split the SciARK dataset into the default values.
```bash
python bilstm_bilstm.py -e embeddings/glove.6B.200d.txt -t dataset/SciARK.json
```

Default values for arguments and other hyper-parameters are in the [config()](https://github.com/afergadis/SciARK/blob/bb2931a8c221bf77c34afa82199228e889d395c6/bilstm_bilstm.py#L18) function.

### (Sci)BERT - BiLSTM
```
usage: scibert_bilstm.py [-h] [-e SENTENCE_ENCODER] [-s SAVE_MODEL]
[-t [TRAIN_SET [TRAIN_SET ...]]]
[-d [DEVELOPMENT_SET [DEVELOPMENT_SET ...]]]
[-x [TEST_SET [TEST_SET ...]]]

optional arguments:
-h, --help            show this help message and exit
-e SENTENCE_ENCODER, --sentence-encoder SENTENCE_ENCODER
A huggingface 🤗 model to use as a sentence encoder.
-s SAVE_MODEL, --save-model SAVE_MODEL
The file name for the saved model. The file name may
have a path and *should* have `.h5` extension.
-t [TRAIN_SET [TRAIN_SET ...]], --train-set [TRAIN_SET [TRAIN_SET ...]]
A list of datasets to use as a training set.
-d [DEVELOPMENT_SET [DEVELOPMENT_SET ...]], --development-set [DEVELOPMENT_SET [DEVELOPMENT_SET ...]]
A list of datasets to use as a development set.
-x [TEST_SET [TEST_SET ...]], --test-set [TEST_SET [TEST_SET ...]]
A list of datasets to use as a test set.
```

The following example will use the SciBERT as sentence encoder and will
split the SciARK dataset into the default values:
```bash
python scibert_bilstm.py -e allenai/scibert_scivocab_uncased -t dataset/SciARK.json
```
* To use *BERT* model as *Sentence Encoder* pass `bert-based-uncased` in the
`--sentence-encoder` argument.
* To use *SciBERT* model as *Sentence Encoder* pass 
`allenai/scibert_scivocab_uncased` in the `--sentence-encoder` argument.

Default values for arguments and other hyper-parameters are in the [config()](https://github.com/afergadis/SciARK/blob/bb2931a8c221bf77c34afa82199228e889d395c6/scibert_bilstm.py#L21)
function.

## 10-fold cross-validation
To run a 10-fold cross-validation use the `10fold-cv-bilstm_bilstm.py` and 
`10fold-cv-scibert_bilstm.py` files. These files run the corresponding 
source passing as arguments the splitted datasets found in the 
`dataset/folds` directory.

## Generalizing to New Domains
To run the cross-domain (leave-out-out) experiment, use the `cross_domain.py`
file. This will use each SDG policy domain successively as a test set and 
the remaining domains as train set.
The experiment continues using again each SDG domain as test set, and this 
time the `AbstRCT` dataset as train set.
