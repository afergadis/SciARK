from pathlib import Path
from scibert_bilstm import ex

sentence_encoder = 'allenai/scibert_scivocab_uncased'
# sentence_encoder = 'bert-base-uncased'
domains = ['3', '5', '7', '10', '12', '13']
for train_dataset in ('SciARK', 'AbstRCT', ):
    for idx in range(len(domains)):
        test_domain = domains.pop(idx)
        if train_dataset == 'SciARK':
            print(f'Training on SDG domains: {domains}')
            datasets = dict(
                train=[f'./dataset/domains/test_{test_domain}/train.json'],
                dev=[f'./dataset/domains/test_{test_domain}/dev.json'],
                test=[f'./dataset/domains/test_{test_domain}/test.json'])
            save_model_path = 'trained_models/SciARK/domains/test_{}_{}'.format(
                test_domain,
                sentence_encoder.split("/")[-1])
        else:
            print('Training on AbstRCT')
            datasets = dict(
                train=['./dataset/AbstRCT/AbstRCT_train.json'],
                dev=['./dataset/AbstRCT/AbstRCT_dev.json'],
                test=[f'./dataset/domains/test_{test_domain}/test.json'])
            save_model_path = f'trained_models/AbstRCT/domains/' \
                              f'test_{test_domain}'
        print(f'Testing on SDG domain: {test_domain}')
        Path(save_model_path).mkdir(parents=True, exist_ok=True)
        r = ex.run(
            config_updates=dict(
                datasets=datasets, save_model=f'{save_model_path}/model.h5'))
        print(f'Results directory: "{save_model_path}"')
        domains.insert(idx, test_domain)
