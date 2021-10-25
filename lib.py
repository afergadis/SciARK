import json
import torch
import numpy as np
import pandas as pd
from keras_nlp.metrics import classification_report
from keras_nlp.metrics.sequence import flatten
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def mean_confidence_interval(
        data, confidence=0.95, bootstrap=False, n_iterations=1000):
    # https://www.statology.org/confidence-intervals-python/
    a = 1.0 * np.array(data)
    n = len(a)
    if bootstrap:
        means = []
        for _ in range(n_iterations):
            s = np.random.choice(a, size=n)
            means.append(np.mean(s))
        return mean_confidence_interval(means, confidence, bootstrap=False)
    m, se = np.mean(a), stats.sem(a)
    if n < 30:
        ci = stats.t.interval(alpha=confidence, df=n - 1, loc=m, scale=se)
    else:
        ci = stats.norm.interval(alpha=confidence, loc=m, scale=se)
    return m, se, ci[0], ci[1]


def aggregate_classification_reports(files, labels=None, metrics=None):
    """ Aggregate classification report results from k-fold runs.

    Parameters
    ----------
    files : list
        A list of json files with the output from scikit-learn classification
        report.
    labels : list, default=None
        A list of the labels for the aggregated results. If none is given, then
        all labels will be displayed and also the `micro avg`, `macro avg`, and
        `weighted avg` will be displayed if they are present in the json file.
    metrics : list, default=None
        The metrics to aggregate. If none is given, the all the metrics
        (`precision`, `recall`, `f1-score`) will be displayed.

    Returns
    -------
    pandas.DataFrame
        A data frame with statistics for the ``labels`` and the ``metrics``.
    """
    results = []
    for fold, file in enumerate(files, start=1):
        with open(file) as fp:
            cr = json.load(fp)
            iterables = [[f'fold{fold}'], list(cr.keys())]
            multi_index = pd.MultiIndex.from_product(
                iterables, names=['fold', 'label'])
            result = pd.DataFrame(cr).T.reset_index(
                drop=True).set_index(multi_index)
            results.append(result)

    data = pd.concat(results)

    stats = []
    index = []
    if labels is None:
        labels = ('EVIDENCE', 'CLAIM', 'micro avg', 'macro avg')
    if metrics is None:
        metrics = ('precision', 'recall', 'f1-score')
    for label in labels:
        for metric in metrics:
            index.append((label, metric))
            stats.append(
                list(
                    mean_confidence_interval(
                        data.query(f'label=="{label}"').loc[:, metric])))

    multi_index = pd.MultiIndex.from_tuples(index, names=['label', 'metric'])
    statistics = pd.DataFrame(
        stats,
        columns=['mean', 'stderr', '95% CI lower', '95% CI upper'],
        index=multi_index)
    return statistics


def encode_text(text, bert_tokenizer, bert_model, device):
    """ Text embeddings.

    Parameters
    ----------
    text : str
        The text to convert to an embedding vector.
    bert_tokenizer
        The BERT tokenizer to encode the `text`.
    bert_model
        The BERT model that will embed the `text`

    Returns
    -------
    array-like, shape(d_model,)
        A vector of d_model dimensions.
    """
    with torch.no_grad():
        try:
            sent_ids = bert_tokenizer.encode(text.lower())
            bert_input = torch.tensor([sent_ids]).to(device)
            outputs = bert_model(bert_input)
            cls_token = outputs[0][0][0]
            emb = cls_token.cpu().numpy()  # Convert to numpy array
            # pooler = outputs[1]
            # emb = pooler.cpu().numpy()
        except RuntimeError:
            print(text)
            pass
    return emb


def pad_array(array, max_rows, truncate='pre'):
    rows, cols = array.shape
    if rows > max_rows:
        if truncate == 'pre':
            return array[-max_rows:, :]
        else:
            return array[:max_rows, :]
    else:
        fill_rows = max_rows - rows
        if truncate == 'pre':
            return np.lib.pad(
                array, ((fill_rows, 0), (0, 0)),
                'constant',
                constant_values=(0))
        else:
            return np.lib.pad(
                array, ((0, fill_rows), (0, 0)),
                'constant',
                constant_values=(0))


def load_dataset(file_path, random_state=None):
    """ Load the corpus data.

    Parameters
    ----------
    file_path : str, Path
        Path to the file of the corpus.
    random_state : int, default None
        If the parameter has a value, then the data are shuffled and returned.
    """
    with open(file_path) as fp:
        corpus = json.load(fp)

    documents, sdgs, texts, labels = [], [], [], []
    for abstract in corpus:
        try:
            sdgs.append(corpus[abstract]['sdg'])
        except KeyError:
            sdgs.append(None)
        documents.append(abstract)
        texts.append(corpus[abstract]['sentences'])
        labels.append([str(l).upper() for l in corpus[abstract]['labels']])

    assert len(texts) == len(labels)
    data = pd.DataFrame(
        zip(documents, sdgs, texts, labels),
        columns=['document', 'sdg', 'sentences', 'labels'])
    if random_state is not None:
        data = data.sample(frac=1, random_state=random_state)
    return data


def split_stratified_into_train_val_test(
        df_input,
        stratify_colname='y',
        frac_train=0.6,
        frac_val=0.15,
        frac_test=0.25,
        random_state=None):
    """ Data frame stratified split.
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.

    Raises
    ------
    ValueError
        In case that the sum of fractions is different than 1.0 or the
        `stratify_column` is not in the data frame.

    See Also
    --------
    https://stackoverflow.com/a/65571687/1143894
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(f'fractions {frac_train}, {frac_val}, {frac_test} '
                         f'do not add up to 1.0')

    if stratify_colname not in df_input.columns:
        raise ValueError(
            f'{stratify_colname} is not a column in the dataframe')

    X = df_input  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df_input[[stratify_colname]]

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X,
        y,
        stratify=y,
        test_size=(1.0 - frac_train),
        random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def evaluate(model, config, X_test, y_test):
    # For the classification report.
    target_names = list(config['label2id'].keys())[1:]
    labels = list(config['label2id'].values())[1:]

    # The model returns logits. Convert to probabilities.
    logits = model(X_test)
    y_pred = softmax(logits).argmax(-1)
    y_true = y_test.argmax(-1)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names[1:],
            labels=labels[1:],
            digits=4))
    y_gold, y_hat = flatten(y_true, y_pred)
    cm = confusion_matrix(y_gold, y_hat, labels=labels)
    print(cm)
    cr_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names[1:],
        labels=labels[1:],
        digits=4,
        output_dict=True)
    return cr_dict, cm


def load_datasets(datasets, train_test_splits, random_state=None):
    def load_list_of_sets(list_of_sets, random_state=None):
        data = None
        for dataset in list_of_sets:
            if data is None:
                data = load_dataset(dataset, random_state)
            else:
                data = data.append(
                    load_dataset(dataset, random_state), ignore_index=True)
        return data

    train = load_list_of_sets(datasets['train'])
    if datasets['test'] is None and datasets['dev'] is None:
        # We have only one dataset. Split to three.
        train, dev, test = split_stratified_into_train_val_test(
            train,
            'sdg',
            frac_train=train_test_splits['train'],
            frac_val=train_test_splits['dev'],
            frac_test=train_test_splits['test'],
            random_state=random_state)
        return train, dev, test

    if datasets['test'] is not None:
        test = load_list_of_sets(datasets['test'], random_state)
    else:
        # The test set is missing. Create one by splitting training set.
        y = train[['sdg']]
        train, test, _, _ = train_test_split(
            train,
            y,
            stratify=y,
            test_size=datasets['train_test_splits']['test'],
            random_state=random_state)
    if datasets['dev'] is not None:
        dev = load_list_of_sets(datasets['dev'], random_state)
    else:
        # The dev set is missing. Create one by splitting training set.
        y = train[['sdg']]
        train, dev, _, _ = train_test_split(
            train,
            y,
            stratify=y,
            test_size=datasets['train_test_splits']['dev'],
            random_state=random_state)

    return train, dev, test