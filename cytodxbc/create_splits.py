import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold


def save_split(dataset, train_ids, val_ids, test_ids, filename):
    print("Train split composition:")
    value_counts = dataset.loc[train_ids, 'label'].value_counts()
    for category, count in value_counts.items():
        print(f"{category}: {count}")
    train_split = dataset.loc[train_ids, 'slide_id']
    val_split = dataset.loc[val_ids, 'slide_id']
    print("Val split composition:")
    value_counts = dataset.loc[val_ids, 'label'].value_counts()
    for category, count in value_counts.items():
        print(f"{category}: {count}")
    test_split = dataset.loc[test_ids, 'slide_id']
    print("Test split composition:")
    value_counts = dataset.loc[test_ids, 'label'].value_counts()
    for category, count in value_counts.items():
        print(f"{category}: {count}")
    df_tr = pd.DataFrame({'train': train_split}).reset_index(drop=True)
    df_v = pd.DataFrame({'val': val_split}).reset_index(drop=True)
    df_t = pd.DataFrame({'test': test_split}).reset_index(drop=True)
    df = pd.concat([df_tr, df_v, df_t], axis=1)
    df.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser(description="Create splits for given dataset")
    parser.add_argument('--dataset', type=str, default='./dataset.csv', help='dataset to use')
    parser.add_argument('--output', type=str, default='./splits', help='output directory')
    parser.add_argument('--n_splits', type=int, default=10, help='number of splits')
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of validation data')
    parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of test data')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--separate_test', action='store_true', help='create seperate test set from cross validation'
                                                                     'splits')
    parser.add_argument('--separate_test_size', type=float, default=0.1, help='fraction of data to keep for '
                                                                              'separate test set')
    parser.add_argument('--regular_cross_validation', action='store_true', help='create regular cross validation '
                                                                                'splits')
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.dataset)
    classes = dataset.label.unique()
    num_classes = len(classes)
    print(f'Number of classes: {num_classes}')
    if args.separate_test:
        train_indices, test_indices = train_test_split(dataset.index, test_size=args.separate_test_size,
                                                       random_state=args.seed, stratify=dataset.label.values)
        pd.DataFrame({'test': dataset.loc[test_indices].slide_id.values}).reset_index(drop=True) \
            .to_csv(os.path.join(args.output, 'seperate_test.csv'), index=False)
        dataset = dataset.loc[train_indices].reset_index(drop=True)

    if args.regular_cross_validation:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for k, (train, test) in enumerate(skf.split(dataset.index, dataset.label.values)):
            print(f"Split {k}:")
            train_ids, val_ids = train_test_split(train, test_size=args.val_frac, random_state=args.seed,
                                                  stratify=dataset.loc[train].label.values)
            save_split(dataset, train_ids, val_ids, test, f'{args.output}/split_{k}.csv')
    else:
        slide_ids = dataset.slide_id.values
        slide_labels = dataset.label.values
        slides_data = {'slide_id': slide_ids, 'label': slide_labels}

        cls_ids = [[] for i in range(num_classes)]
        for i, label in enumerate(classes):
            cls_ids[i] = np.where(slides_data['label'] == label)[0]

        rng = np.random.default_rng(args.seed)
        num_slides_cls = np.array([len(cls_id) for cls_id in cls_ids])
        val_num = np.round(num_slides_cls * args.val_frac).astype(int)
        test_num = np.round(num_slides_cls * args.test_frac).astype(int)
        samples = len(slide_ids)
        indices = np.arange(samples).astype(int)

        def create_splits():
            for i in range(args.n_splits):
                all_val_ids = []
                all_test_ids = []
                sampled_train_ids = []
                for c in range(len(val_num)):
                    possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
                    val_ids = rng.choice(list(possible_indices), val_num[c], replace=False)  # validation ids

                    remaining_ids = np.setdiff1d(possible_indices,
                                                 val_ids)  # indices of this class left after validation
                    all_val_ids.extend(val_ids)

                    test_ids = rng.choice(remaining_ids, test_num[c], replace=False)
                    remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                    all_test_ids.extend(test_ids)

                    sampled_train_ids.extend(remaining_ids)
                yield sampled_train_ids, all_val_ids, all_test_ids

        generated_splits = create_splits()
        for k in range(args.n_splits):
            print(f"Split {k}:")
            ids = next(generated_splits)

            train_ids, val_ids, test_ids = ids
            save_split(dataset, train_ids, val_ids, test_ids, f'{args.output}/split_{k}.csv')


if __name__ == '__main__':
    main()
