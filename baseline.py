import json
import pandas as pd
from sklearn.metrics import classification_report


def predict(abstract):
    # Calculate relative positions.
    n_sents = len(abstract)
    relative_position = [i / n_sents for i in range(n_sents)]
    labels = []
    for pos in relative_position:
        if 0.5 <= pos < 0.8:
            labels.append('EVIDENCE')
        elif pos >= 0.8:
            labels.append('CLAIM')
        else:
            labels.append('NONE')

    return labels


if __name__ == '__main__':
    report = {
        'CLAIM': {
            'P': 0,
            'R': 0,
            'F': 0
        },
        'EVIDENCE': {
            'P': 0,
            'R': 0,
            'F': 0
        }
    }
    for i in range(1, 11):
        with open(f'./dataset/folds/fold{i}/test.json') as fp:
            dataset = json.load(fp)

        true_labels, predictions = [], []
        for abstract in dataset:
            true_labels.extend(dataset[abstract]['labels'])
            predictions.extend(predict(dataset[abstract]['sentences']))

        cr = classification_report(
            true_labels,
            predictions,
            labels=['CLAIM', 'EVIDENCE'],
            digits=4,
            output_dict=True)
        report['CLAIM']['P'] += cr['CLAIM']['precision']
        report['CLAIM']['R'] += cr['CLAIM']['recall']
        report['CLAIM']['F'] += cr['CLAIM']['f1-score']
        report['EVIDENCE']['P'] += cr['EVIDENCE']['precision']
        report['EVIDENCE']['R'] += cr['EVIDENCE']['recall']
        report['EVIDENCE']['F'] += cr['EVIDENCE']['f1-score']

    report['CLAIM']['P'] /= 10
    report['CLAIM']['R'] /= 10
    report['CLAIM']['F'] /= 10
    report['EVIDENCE']['P'] /= 10
    report['EVIDENCE']['R'] /= 10
    report['EVIDENCE']['F'] /= 10

    print(pd.DataFrame(report).T)
