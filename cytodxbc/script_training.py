import argparse
import os
import numpy as np
import torch
from torch.nn.functional import softmax
from pathlib import Path

from utils import apply_random_seed, split_dataset, get_dataloader, measure_perf
from dataset import DatasetPretrained
from model import GatedAttention



def _define_args():
    parser = argparse.ArgumentParser(description='CytoDxBC Training Script')

    parser.add_argument('--experiment_path', type=str, default='./experiments', metavar='PATH',
                        help='path of the experiment')
    parser.add_argument('--experiment_name', type=str, default='CytoDxBC', metavar='NAME',
                        help='name of the experiment')
    parser.add_argument('--split_id', type=int, default=0, metavar='N', help='split id')
    parser.add_argument('--feats_folder', type=str, default='./patches', metavar='PATH',
                        help='path of folder containing feats extracted from patch')
    parser.add_argument('--slides_label_filepath', type=str, default='sample_data/labels.csv', metavar='PATH',
                        help='path of CSV-file containing slide labels')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR', help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-6, metavar='R', help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=["Adam", "AdamW"], metavar='OPTIMIZER', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=2, metavar='SIZE',
                        help='number of slides sampled per iteration')

    # Dataset parameters
    parser.add_argument('--split_folder', type=str, default=None, help="predetermined splitting of data")

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')

    args = parser.parse_args()

    if args.split_folder is not None:
        experiment_path = os.path.join(args.experiment_path, args.experiment_name, f"Split {args.split_id}")
    else:
        experiment_path = os.path.join(args.experiment_path, args.experiment_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)

    hyper_parameters = {
        'experiment_path': experiment_path,
        'feats_folder': args.feats_folder,
        'slides_label_filepath': args.slides_label_filepath,
        'epochs': args.epochs,
        'lr': args.lr,
        'reg': args.reg,
        'split_folder': args.split_folder,
        'split_id': args.split_id,
        'batch_size': args.batch_size,
        'j': args.j,
        'optimizer': args.optimizer,
        'seed': args.seed,
    }

    return hyper_parameters


def perform_epoch(mil_model, dataloader, optimizer, loss_function, train=True):
    """
    Perform a complete training epoch by looping through all data of the dataloader.
    :param mil_model: MIL model to be trained
    :param dataloader: loader of the dataset
    :param optimizer: pytorch optimizer
    :param loss_function: loss function to compute gradients
    :return: (mean of losses, balanced accuracy)
    """
    proba_predictions = []
    ground_truths = []
    losses = []

    device = next(mil_model.parameters()).device

    for data, slides_labels, slides_ids in dataloader:
        slides_labels = slides_labels.to(device)

        if train:
            optimizer.zero_grad()
            predictions = mil_model(data)
            loss = loss_function(predictions, slides_labels)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                predictions = mil_model(data)
                loss = loss_function(predictions, slides_labels)

        predictions = softmax(predictions, dim=-1)

        # Store data for finale epoch average measures
        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(predictions.detach().cpu().numpy())
        ground_truths.extend(slides_labels.cpu().numpy())

    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)

    return losses, proba_predictions, ground_truths, predicted_classes

def main(hyper_parameters):
    # Set seed
    apply_random_seed(hyper_parameters['seed'])

    # Loads dataset and dataloader
    print('Loading data')
    dataset = DatasetPretrained(hyper_parameters['feats_folder'], hyper_parameters['slides_label_filepath'])

    n_classes = dataset.n_classes
    if hyper_parameters['split_folder'] is not None:
        split_path = os.path.join(hyper_parameters['split_folder'], f"split_{hyper_parameters['split_id']}.csv")
    else:
        split_path = None
    train_dataset, val_dataset = split_dataset(dataset, 0.1, split_path)

    # Check that every slide folder has a corresponding label
    assert all([slide_id in train_dataset.dataset.slides_labels.keys() for slide_id in
                train_dataset.dataset.slides_ids[train_dataset.indices]])
    assert all([slide_id in val_dataset.dataset.slides_labels.keys() for slide_id in
                val_dataset.dataset.slides_ids[val_dataset.indices]])

    train_dataloader = get_dataloader(train_dataset, hyper_parameters['batch_size'], True, hyper_parameters['j'],
                                      seed=hyper_parameters['seed'])
    val_dataloader = get_dataloader(val_dataset, hyper_parameters['batch_size'], False, hyper_parameters['j'],
                                    seed=hyper_parameters['seed'])
    print('done')

    # Loads MIL model, optimizer and loss function

    print('Loading model')
    feats_size = dataset.feat_size
    print(feats_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedAttention(feats_size, n_classes).to(device)

    print('done')

    if hyper_parameters['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), hyper_parameters['lr'],
                                     weight_decay=hyper_parameters['reg'])
    elif hyper_parameters['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), hyper_parameters['lr'],
                                      weight_decay=hyper_parameters['reg'])
    else:
        raise ValueError('Optimizer not supported')
    loss_function = torch.nn.CrossEntropyLoss()

    # Loop through all epochs
    print('Starting training...')
    best_val_loss = np.inf

    for epoch in range(hyper_parameters["epochs"]):
        model.train()
        train_losses, train_probas, \
            train_ground_truths, train_predicted_classes = perform_epoch(model, train_dataloader, optimizer,
                                                                         loss_function)

        train_loss, train_bac, train_f1, train_auc = measure_perf(train_losses, train_ground_truths,
                                                                  train_predicted_classes, train_probas,
                                                                  n_classes)

        print('Epoch', f'{epoch:3d}/{hyper_parameters["epochs"]}', f'    train_loss={train_loss:.3f}',
              f'    train_bac={train_bac:.3f}', f'    train_f1={train_f1:.3f}', f'    train_auc={train_auc:.3f}')
        model.eval()

        val_losses, val_probas, \
        val_ground_truths, val_predicted_classes = perform_epoch(model,
                                                                val_dataloader, optimizer,
                                                                loss_function, train=False)

        val_loss, val_bac, val_f1, val_auc = measure_perf(val_losses, val_ground_truths,
                                                            val_predicted_classes, val_probas,
                                                            n_classes)

        print('Epoch', f'{epoch:3d}/{hyper_parameters["epochs"]}', f'    val_loss={val_loss:.3f}',
              f'    val_bac {val_bac:.3f}', f'    val_f1={val_f1:.3f}', f'    val_auc={val_auc:.3f}')

        if val_loss < best_val_loss:
            print("New best performing model on val dataset, saving model....")
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': hyper_parameters,
                'epoch': epoch,
            }
            checkpoint_path = os.path.join(hyper_parameters['experiment_path'], f"model_{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)
            best_val_loss = val_loss
            print(f"New best val loss:{best_val_loss}")


print('done')

if __name__ == '__main__':
    main(_define_args())
