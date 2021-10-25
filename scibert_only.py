import random
import numpy as np
import torch
from sacred import Experiment
from functools import partial
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
    Trainer)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report)
from lib import load_datasets

ex = Experiment(name='SciBERT-Only')


@ex.config
def config():
    seed = 2021
    # Data configurations
    label2id = {'NONE': 0, 'EVIDENCE': 1, 'CLAIM': 2}
    num_classes = len(label2id)
    max_tokens = 192
    sentence_encoder = "allenai/scibert_scivocab_uncased"
    # The directory name of the trained model.
    save_dir = 'trained_models/scibert_sentence_encoder'

    datasets = dict(
        train=['./dataset/folds/fold1/train.json'],
        dev=['./dataset/folds/fold1/dev.json'],
        test=['./dataset/folds/fold1/test.json'])
    train_test_splits = dict(train=0.6, dev=0.2, test=0.2)
    epochs = 2


class SciArgsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, device='cpu'):
        self.encodings = encodings
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


@ex.main
def main(_config, _log):
    seed = _config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, dev, test = load_datasets(
        _config['datasets'], _config['train_test_splits'], seed)
    train_sentences = train['sentences'].explode().to_list()
    dev_sentences = dev['sentences'].explode().to_list()
    test_sentences = test['sentences'].explode().to_list()

    train_labels = train['labels'].explode().map(
        lambda label: _config['label2id'][label]).to_list()
    dev_labels = dev['labels'].explode().map(
        lambda label: _config['label2id'][label]).to_list()
    test_labels = test['labels'].explode().map(
        lambda label: _config['label2id'][label]).to_list()

    bert_tokenizer = AutoTokenizer.from_pretrained(
        _config['sentence_encoder'], use_fast=True)
    tokenizer = partial(
        bert_tokenizer,
        max_length=_config['max_tokens'],
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    train_encodings = tokenizer(train_sentences)
    dev_encodings = tokenizer(dev_sentences)
    test_encodings = tokenizer(test_sentences)
    train_dataset = SciArgsDataset(train_encodings, train_labels, device.type)
    dev_dataset = SciArgsDataset(dev_encodings, dev_labels, device.type)
    test_dataset = SciArgsDataset(test_encodings, test_labels, device.type)

    model = AutoModelForSequenceClassification.from_pretrained(
        _config['sentence_encoder'], num_labels=_config['num_classes'])

    training_args = TrainingArguments(
        output_dir=_config['save_dir'],  # output directory
        num_train_epochs=_config['epochs'],  # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=10,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        dataloader_pin_memory=False)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
    )
    trainer.train()

    # Predictions
    predictions = trainer.predict(test_dataset)
    preds = torch.nn.Softmax(predictions.predictions).dim.argmax(-1)
    print(
        classification_report(
            test_labels,
            preds,
            labels=[1, 2],
            target_names=['EVIDENCE', 'CLAIM'],
            digits=4))


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
        '--save-dir',
        type=str,
        default=None,
        help='The directory name for the saved model. ')
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
    if args.save_dir is not None:
        config_update['save_dir'] = args.save_dir
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
