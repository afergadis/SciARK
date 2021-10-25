import json
import numpy as np
import pickle
import random
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sacred import Experiment
from models import bilstm_bilstm
from keras_nlp import SentWordVectorizer
from lib import evaluate, load_datasets

ex = Experiment(name='BiLSTM-BiLSTM Encoder')


@ex.config
def config():
    seed = 2021

    # Argument Labels.
    label2id = {'PAD': 0, 'NONE': 1, 'EVIDENCE': 2, 'CLAIM': 3}
    num_classes = len(label2id)

    # Vectorizer parameters.
    num_words, filters, oov_token = 10000, '!@#$^&*"', '__UNK__'

    # Max sentences and words per abstract.
    max_sentences, max_words = 20, 40
    # Input of an abstract: max_sentences x 768 embedding vector.
    doc_shape = (max_sentences, max_words)
    embeddings_file = 'embeddings/glove.6B.200d.txt'
    rnn = dict(units=64, dropout=0.3, recurrent_dropout=0.3)
    model_fit = dict(epochs=20, batch_size=16)
    save_model = 'trained_models/bilstm_bilstm.h5'

    datasets = dict(train=['./dataset/SciARK.json'], dev=None, test=None)
    train_test_splits = dict(train=0.6, dev=0.2, test=0.2)


@ex.capture
def labels_to_categorical(data, _config):
    # Convert labels to numerical values.
    data['y'] = data.labels.map(
        lambda row: [_config['label2id'][r] for r in row])
    data['y'] = data.y.map(
        lambda row: pad_sequences(
            [row],
            maxlen=_config['max_sentences'],
            value=_config['label2id']['PAD']).tolist()[0])
    return to_categorical(data.y.to_list())


@ex.main
def main(_config, _log):
    seed = _config['seed']
    np.random.seed(seed)
    random.seed(seed)

    train, dev, test = load_datasets(
        _config['datasets'], _config['train_test_splits'], seed)
    sent_word_vectorizer = SentWordVectorizer(
        num_words=_config['num_words'],
        filters=_config['filters'],
        oov_token=_config['oov_token'])

    text_train = train.sentences.to_list()
    sent_word_vectorizer.fit_on_texts(text_train)
    # fn = _config['datasets']['train'].replace('.json', '_vectorizer.p')
    with open('train_vectorizer.p', 'wb') as fh:
        pickle.dump(sent_word_vectorizer, fh)

    X_train = sent_word_vectorizer.texts_to_vectors(
        text_train, shape=_config['doc_shape'], truncating='post')
    X_dev = sent_word_vectorizer.texts_to_vectors(
        dev.sentences.to_list(), shape=_config['doc_shape'], truncating='post')
    X_test = sent_word_vectorizer.texts_to_vectors(
        test.sentences.to_list(),
        shape=_config['doc_shape'],
        truncating='post')
    y_train = labels_to_categorical(train)
    y_dev = labels_to_categorical(dev)
    y_test = labels_to_categorical(test)

    model = bilstm_bilstm(
        _config['doc_shape'], _config['num_classes'],
        sent_word_vectorizer.token2id, sent_word_vectorizer.oov_token,
        _config['embeddings_file'], **_config['rnn'])

    callbacks = [
        EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        ModelCheckpoint(
            filepath=_config['save_model'], save_best_only=True, verbose=1)
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_dev, y_dev),
        callbacks=callbacks,
        **_config['model_fit'])

    cr, cm = evaluate(model, _config, X_test, y_test)
    save_model_dir = Path(_config['save_model']).parent
    with open(f'{save_model_dir}/classification_report.json', 'w') as fp:
        json.dump(cr, fp, indent=4, ensure_ascii=True)
    np.savetxt(f'{save_model_dir}/confusion_matrix.txt', cm)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '-e',
        '--embeddings-file',
        type=str,
        default=None,
        help='A file with word embeddings in GloVe format.')
    parser.add_argument(
        '-s',
        '--save-model',
        type=str,
        default='trained_models/bilstm_based_context_encoder.h5',
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
    if args.embeddings_file is not None:
        config_update['embeddings_file'] = args.embeddings_file
    if args.save_model is not None:
        config_update['save_model'] = args.save_model
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
