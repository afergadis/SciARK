from pathlib import Path
from lib import aggregate_classification_reports
from scibert_bilstm import ex

sentence_encoder = 'allenai/scibert_scivocab_uncased'
# sentence_encoder = 'bert-base-uncased'
save_model_path = 'trained_models/SciARK/folds_{}'.format(
    sentence_encoder.split('/')[0])

for fold in range(1, 11):
    print(f'Fold: {fold:2d}')
    dataset_fold = f'./dataset/folds/fold{fold}'
    datasets = dict(
        train=[f'{dataset_fold}/train.json'],
        dev=[f'{dataset_fold}/dev.json'],
        test=[f'{dataset_fold}/test.json'])
    Path(f'{save_model_path}/fold{fold}').mkdir(parents=True, exist_ok=True)
    r = ex.run(
        config_updates=dict(
            datasets=datasets,
            sentence_encoder=sentence_encoder,
            save_model=f"{save_model_path}/fold{fold}/model.h5"))

reports = Path(save_model_path).glob('**/classification_report.json')
print(aggregate_classification_reports(reports))
