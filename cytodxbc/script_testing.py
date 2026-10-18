import argparse
import os
import torch
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from torch.nn.functional import softmax
from dataset import DatasetPretrained
from model import GatedAttention
from utils import get_dataloader
from utils import apply_random_seed

def main():
    parser = argparse.ArgumentParser(description='CytoDxBC Testing script')
    parser.add_argument('--experiment_folder', type=str, default='./experiments', metavar='PATH',
                        help='path to folder containing experiments')
    parser.add_argument('--experiment_name', type=str, default='CytoDxBC', metavar='NAME',
                        help='name of the experiment')
    parser.add_argument('--feats_folder', type=str, default='sample_data', metavar='PATH',
                        help='path of folder containing feats from patches')
    parser.add_argument('--slides_label_filepath', type=str, default='sample_data/labels.csv', metavar='PATH',
                        help='path of CSV-file containing slide labels')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, metavar='SIZE',
                        help='number of slides sampled per iteration')
    parser.add_argument('--validation', action='store_true', help='compute performances on validation set')

    # Dataset parameters
    parser.add_argument('--split_folder', type=str, default=None, help="folder containing splits")
    parser.add_argument('--split_id', type=int, default=None, help="split id to use for testing")
    
    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')

    args = parser.parse_args()
    if (args.split_folder is None) != (args.split_id is None):
        parser.error('--split_folder and --split_id must be provided together')
    apply_random_seed(args.seed)

    print('Loading data')

    test_dataset = DatasetPretrained(args.feats_folder, args.slides_label_filepath)
    print(test_dataset.correspondence_digit_label_name)
    n_classes = test_dataset.n_classes

    result_df_filename = "results_val.csv" if args.validation else "results.csv"
    if os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, result_df_filename)):
        results_df = pd.read_csv(os.path.join(args.experiment_folder, args.experiment_name, result_df_filename))
    else:
        results_df = pd.DataFrame(columns=["split_id", "auc", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "BAC"])

    if args.split_folder is not None:
        split_csv = pd.read_csv(os.path.join(args.split_folder, f"split_{args.split_id}.csv"))
        split_values = split_csv.val.values if args.validation else split_csv.test.values
        test_indices = np.nonzero(np.in1d(test_dataset.slides_ids, split_values))[0]
        test_dataset_split = torch.utils.data.Subset(test_dataset, test_indices)
        # Check that every slide folder has a corresponding label
        assert all([slide_id in test_dataset_split.dataset.slides_labels.keys() for slide_id in
                    test_dataset_split.dataset.slides_ids[test_dataset_split.indices]])
    else:
        test_dataset_split = test_dataset
    
    print(len(test_dataset_split))
    test_dataloader = get_dataloader(test_dataset_split, args.batch_size, False, args.j, seed=args.seed)
    print('done')

    # Loads MIL model, optimizer and loss function
    print('Loading model')
    feats_size = test_dataset.feat_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedAttention(feats_size, n_classes).to(device)
    
    split_id = args.split_id if args.split_id is not None else "test"
    if args.split_folder is not None:
        # Retrieves best validation model by looking last epoch saved
        experiment_path = os.path.join(args.experiment_folder, args.experiment_name, f"Split {split_id}")
    else:
        experiment_path = os.path.join(args.experiment_folder, args.experiment_name)
    if "best_model.pt" in os.listdir(experiment_path):
        best_model = os.path.join(experiment_path, "best_model.pt")
    else:
        checkpoints = sorted([file for file in os.listdir(experiment_path) if file.endswith(".pt")],
                                key=lambda x: int(x.split('_')[1].split('.')[0]))
        best_model = os.path.join(experiment_path, checkpoints[-1])
        print(f'Best model found: {best_model}')
    model.load_state_dict(torch.load(best_model, map_location=device)['model'])

    print('Starting inference...')
    model.eval()

    logits = []
    proba_predictions = []
    ground_truths = []
    slides = []

    for data, slides_labels, slide_ids in test_dataloader:
        slides_labels = slides_labels.to(device)

        with torch.no_grad():
            predictions = model(data)
        predictions = softmax(predictions, dim=-1)
        # Store data for final epoch average measures
        logits.append(predictions)
        proba_predictions.extend(predictions.detach().cpu().numpy())
        ground_truths.extend(slides_labels.cpu().numpy())
        slides.extend(slide_ids)
        
    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)
    
    auc_score = metrics.roc_auc_score(ground_truths, proba_predictions[:, 1])
    TN, FP, FN, TP = metrics.confusion_matrix(ground_truths, predicted_classes).ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    F1 = 2 * (PPV * sensitivity) / (PPV + sensitivity + 1e-12)
    BAC = (sensitivity + specificity) / 2

    results_dict = {"split_id": split_id, "auc": auc_score, "Sensitivity": sensitivity,
                    "Specificity": specificity, "PPV": PPV, "NPV": NPV, "F1": F1, "BAC": BAC}

    # Replaces previous results for this split instead of appending duplicates when re-running
    results_df = results_df[results_df.split_id.astype(str) != str(split_id)]
    results_df = pd.concat([results_df, pd.DataFrame(results_dict, index=[0])], ignore_index=True)

    pred_df = pd.DataFrame({"split_id": [split_id] * len(slides), 
                            "slide_id": slides, 
                            "label": ground_truths,
                            "pred": predicted_classes,
                            f"pred_prob_{test_dataset.correspondence_digit_label_name[0]}":
                                proba_predictions[:, 0],
                            f"pred_prob_{test_dataset.correspondence_digit_label_name[1]}":
                                proba_predictions[:, 1]})

    pred_df_filename = "predictions_val.csv" if args.validation else "predictions.csv"
    if os.path.exists(os.path.join(args.experiment_folder, args.experiment_name, pred_df_filename)):
        previous_pred_df = pd.read_csv(os.path.join(args.experiment_folder, args.experiment_name, pred_df_filename))
        previous_pred_df = previous_pred_df[previous_pred_df.split_id.astype(str) != str(split_id)]
        pred_df = pd.concat([previous_pred_df, pred_df], ignore_index=True)

    results_df.to_csv(os.path.join(args.experiment_folder, args.experiment_name, result_df_filename), index=False)
    pred_df.to_csv(os.path.join(args.experiment_folder, args.experiment_name, pred_df_filename), index=False)


if __name__ == '__main__':
    main()