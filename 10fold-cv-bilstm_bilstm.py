from pathlib import Path
from lib import aggregate_classification_reports
from bilstm_bilstm import ex

embeddings_file = 'glove.6B.200d.txt'
save_model_path = f'trained_models/SciARK/folds_bilstm_bilstm_' \
                  f'{embeddings_file.replace(".txt", "")}'

for fold in range(1, 11):
    print(f'Fold: {fold:2d}')
    dataset_fold = f'./dataset/folds/fold{fold}'
    datasets = dict(
        train=f'{dataset_fold}/train.json',
        dev=f'{dataset_fold}/dev.json',
        test=f'{dataset_fold}/test.json')
    Path(f'{save_model_path}/fold{fold}').mkdir(parents=True, exist_ok=True)
    r = ex.run(
        config_updates=dict(
            datasets=datasets,
            embeddings_file=f'embeddings/{embeddings_file}',
            save_model=f"{save_model_path}/fold{fold}/model.h5"))

reports = Path(save_model_path).glob('**/classification_report.json')
print(aggregate_classification_reports(reports))
