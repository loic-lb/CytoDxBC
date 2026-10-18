import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, f1_score
from sklearn.metrics import auc as calc_auc

def apply_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset, test_size, split=None):
    if split is not None:
        split_csv = pd.read_csv(split)
        train_indices = np.nonzero(np.in1d(dataset.slides_ids, split_csv.train.values))[0]
        if "val" not in split_csv.columns:
            train_indices, test_indices = train_test_split(train_indices, test_size=test_size,
                                                           stratify=[dataset.slides_labels[dataset.slides_ids[idx]] for
                                                                     idx in train_indices],
                                                           random_state=0)
        else:
            test_indices = np.nonzero(np.in1d(dataset.slides_ids, split_csv.val.values))[0]
    else:
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        train_indices, test_indices = train_test_split(indices, test_size=test_size,
                                                       stratify=[dataset.slides_labels[dataset.slides_ids[idx]] for
                                                                 idx in indices], random_state=0)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(len(train_dataset))
    print(len(test_dataset))

    return train_dataset, test_dataset


def custom_collate(batch):
    it = iter(batch)
    elem_size = len(next(it))
    if not all(len(elem) == elem_size for elem in it):
        raise RuntimeError('each element in list of batch should be of equal size')
    transposed = list(zip(*batch))
    return [list(transposed[0]),  # tiles
            default_collate(transposed[1]),  # labels
            default_collate(transposed[2])]  # slide ids


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, shuffle, num_workers, custom_collate_bool=True, seed=0):
    custom_collate_fn = custom_collate if custom_collate_bool else default_collate
    if num_workers > 0:
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                                           collate_fn=custom_collate_fn,
                                           worker_init_fn=seed_worker,
                                           generator=g)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                                           collate_fn=custom_collate_fn)

def measure_perf(losses, ground_truths, predicted_classes, probas, n_classes=2):
    loss = np.mean(losses)
    bac = balanced_accuracy_score(ground_truths, predicted_classes)
    f1 = f1_score(ground_truths, predicted_classes, average="weighted")
    if n_classes == 2:
        auc = roc_auc_score(ground_truths, probas[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(ground_truths, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in ground_truths:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probas[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))
    return loss, bac, f1, auc
