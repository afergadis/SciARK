import json
import numpy as np
import pickle
import random
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from lib import encode_text, pad_array, load_dataset, evaluate
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sacred import Experiment
from tqdm import tqdm
from models import bert_bilstm

ex = Experiment(name='BERT based Context Encoder')
tqdm.pandas(smoothing=0)


@ex.config
def config():
    seed = 2021

    # Argument Labels.
    label2id = {'PAD': 0, 'NONE': 1, 'EVIDENCE': 2, 'CLAIM': 3}
    num_classes = len(label2id)

    # Max sentences per abstract.
    max_sentences = 20
    # Input of an abstract: max_sentences x 768 embedding vector.
    sents_shape = (max_sentences, 768)
    rnn = dict(units=64, dropout=0.3, recurrent_dropout=0.3)
    model_fit = dict(epochs=40, validation_split=0.2)

    # A BERT model to use as a Sentence Encoder,
    # e.g., 'allenai/scibert_scivocab_uncased', 'bert_base_uncased', etc.
    sentence_encoder = 'allenai/scibert_scivocab_uncased'
    # The file name of the trained model.
    save_model = 'trained_models/bert_based_context_encoder.h5'

    # You can have multiple datasets per list. All datasets (per list) will be
    # encoded and stacked to form a new combined one.
    datasets = dict(
        train=['dataset/SciARK.json'],
        dev=[],
        test=[])
    train_test_splits = dict(train=0.6, dev=0.2, test=0.2)


def load_datasets(config):
    datasets = dict(
        train=dict(),
        dev=dict(),
        test=dict(),
        X=dict(train=dict(), dev=dict(), test=dict()),
        y=dict(train=dict(), dev=dict(), test=dict()))
    for dataset in ('train', 'dev', 'test'):
        for src in config['datasets'][dataset]:
            # Try to load pickled datasets.
            pickled = f"{src.replace('.json', '')}_" \
                      f"{config['sentence_encoder'].split('/')[-1]}.p"
            try:
                with open(pickled, 'rb') as fp:
                    X, y = pickle.load(fp)
                    datasets['X'][dataset][src] = X
                    datasets['y'][dataset][src] = y
            except FileNotFoundError:
                datasets[dataset][src] = load_dataset(src, config['seed'])

    return datasets


def stack_encoded_datasets(datasets, key, dataset):
    # Stack the encoded datasets.
    stack = None
    if key in datasets:
        if dataset in datasets[key]:
            for corpus in datasets[key][dataset]:
                if stack is None:
                    stack = datasets[key][dataset][corpus]
                else:
                    stack = np.vstack([stack, datasets[key][dataset][corpus]])

    return stack


def encode_datasets(datasets, config, _log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = AutoTokenizer.from_pretrained(config['sentence_encoder'])
    bert_model = AutoModel.from_pretrained(config['sentence_encoder']).to(
        device)
    label2id = config['label2id']
    max_sentences = config['max_sentences']
    for dataset in ('train', 'dev', 'test'):
        for corpus in datasets[dataset]:
            _log.info(f'Encoding {corpus}')
            data = datasets[dataset][corpus]
            # Encode labels to one-hot vectors.
            data['y'] = data.labels.map(lambda row: [label2id[r] for r in row])
            data['y'] = data.y.map(
                lambda row: pad_sequences([row], maxlen=max_sentences)[0])
            y = to_categorical(data.y.to_list())
            datasets['y'][dataset][corpus] = y

            # Encode sentences.
            data['sentence_embeddings'] = data.sentences.progress_map(
                lambda row: [
                    encode_text(sentence, bert_tokenizer, bert_model, device)
                    for sentence in row
                ])
            # Reshape each document's sentence embeddings to:
            # 1 (doc), #sents, 768
            data['X'] = data.sentence_embeddings.map(
                lambda vec: pad_array(np.vstack(vec), max_sentences))

            # data.X.to_numpy() doesn't return a 3D array.
            # It returns (len(data), ). So, I stack all rows and reshape.
            X = np.vstack(data.X.to_numpy()).reshape(
                (len(data), max_sentences, 768))
            datasets['X'][dataset][corpus] = X

            # Pickle encoded data.
            pickled = '{}_{}.p'.format(
                corpus.replace('.json', ''),
                config['sentence_encoder'].split('/')[-1])
            with open(pickled, 'wb') as fp:
                pickle.dump((X, y), fp)

    X_train = stack_encoded_datasets(datasets, 'X', 'train')
    y_train = stack_encoded_datasets(datasets, 'y', 'train')
    X_dev = stack_encoded_datasets(datasets, 'X', 'dev')
    y_dev = stack_encoded_datasets(datasets, 'y', 'dev')
    X_test = stack_encoded_datasets(datasets, 'X', 'test')
    y_test = stack_encoded_datasets(datasets, 'y', 'test')

    encoded_datasets = dict(
        X=dict(train=X_train, dev=X_dev, test=X_test),
        y=dict(train=y_train, dev=y_dev, test=y_test),
    )
    return encoded_datasets


@ex.main
def main(_config, _log):
    seed = _config['seed']
    np.random.seed(seed)
    torch.seed = seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)

    _log.info(f'Training on: {_config["datasets"]["train"]}')
    if len(_config['datasets']['dev']) > 0:
        _log.info(f'Development set: {_config["datasets"]["dev"]}')
    _log.info(f'Testing on: {_config["datasets"]["test"]}')

    datasets = load_datasets(_config)
    embdeded_datasets = encode_datasets(datasets, _config, _log)
    X_train = embdeded_datasets['X']['train']
    y_train = embdeded_datasets['y']['train']
    X_dev = embdeded_datasets['X']['dev']
    if X_dev is not None:
        y_dev = embdeded_datasets['y']['dev']
        validation_data = (X_dev, y_dev)
    else:
        validation_data = None

    model = bert_bilstm(
        sents_shape=_config['sents_shape'],
        num_classes=_config['num_classes'],
        **_config['rnn'])
    print(model.summary(100))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=_config['save_model'], save_best_only=True, verbose=1)
    ]
    if validation_data is None:
        model.fit(
            X_train,
            y_train,
            callbacks=callbacks,
            **_config['model_fit'],
            verbose=2)
    else:
        model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            callbacks=callbacks,
            epochs=_config['model_fit']['epochs'],
            verbose=2)

    X_test = embdeded_datasets['X']['test']
    y_test = embdeded_datasets['y']['test']
    cr, cm = evaluate(model, _config, X_test, y_test)
    save_model_dir = Path(_config['save_model']).parent
    with open(f'{save_model_dir}/classification_report.json', 'w') as fp:
        json.dump(cr, fp, indent=4, ensure_ascii=True)
    np.savetxt(f'{save_model_dir}/confusion_matrix.txt', cm)
    return


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '-e',
        '--sentence-encoder',
        type=str,
        default=None,
        help='A huggingface 🤗  model to use as a sentence encoder.')
    parser.add_argument(
        '-s',
        '--save-model',
        type=str,
        default='trained_models/bert_based_context_encoder.h5',
        help='The file name for the saved model. '
        'The file name may have a path and *should* have `.h5` extension.')
    parser.add_argument(
        '-t',
        '--train-set',
        nargs='*',
        default=None,
        help='A list of datasets to use as a training set.')
    parser.add_argument(
        '-d',
        '--development-set',
        nargs='*',
        default=None,
        help='A list of datasets to use as a development set.')
    parser.add_argument(
        '-x',
        '--test-set',
        nargs='*',
        default=None,
        help='A list of datasets to use as a test set.')
    args = parser.parse_args()

    config_update = dict()
    if args.sentence_encoder is not None:
        config_update['sentence_encoder'] = args.sentence_encoder
    if args.save_model is not None:
        config_update['save_model'] = args.save_model
        # Create the parent directories.
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    if args.train_set is not None:
        if 'datasets' not in config_update:
            config_update['datasets'] = dict()
        config_update['datasets']['train'] = args.train_set
    if args.development_set is not None:
        if 'datasets' not in config_update:
            config_update['datasets'] = dict()
        config_update['datasets']['dev'] = args.development_set
    if args.test_set is not None:
        if 'datasets' not in config_update:
            config_update['datasets'] = dict()
        config_update['datasets']['test'] = args.test_set
    ex.run(config_updates=config_update)
