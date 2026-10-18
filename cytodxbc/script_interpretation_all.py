import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from pathlib import Path
from utils import get_dataloader, apply_random_seed
from model import GatedAttention
from dataset import DatasetPretrained

from utils import split_dataset
from interpretation_utils import create_heatmap, retrieve_top_k_tiles
from calibration_utils import set_temperature, temperature_scale, _ECELoss


def main():
    parser = argparse.ArgumentParser(description="Interpratation of the model predictions")
    parser.add_argument('--experiment_folder', type=str, default='./experiments', metavar='PATH',
                        help='path to folder containing experiments')
    parser.add_argument('--experiment_name', type=str, default='SparseConvMIL', metavar='NAME',
                        help='name of the experiment')
    parser.add_argument('--slides_path', type=str, help="path of folder containing slides")
    parser.add_argument('--patch_folder', type=str, help="path of folder containing patches")
    parser.add_argument('--feats_folder_training', type=str, help="path of folder containing feats"
                                                         "extracted from patches used for training")
    parser.add_argument('--feats_folder_validation', type=str, help="path of folder containing feats"
                                                         "extracted from patches used for validation")
    parser.add_argument('--category_lesions', type=str, required=True, metavar='CATEGORY',
                        help='category of lesions to obtain stratified results')
    parser.add_argument('--slides_label_filepath_training', type=str, help="path of the CSV-file containing"
                                                                  "slide labels used for training")
    parser.add_argument('--slides_label_filepath_validation', type=str, help="path of the CSV-file containing"
                                                                  "slide labels used for validation")
    parser.add_argument('--extension', type=str, default="ndpi", help="extension of the slides")
    parser.add_argument('--model', type=str, default="attention", choices=["attention"],
                        help='mil model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('--k', type=int, default=10, metavar='N', help='number of topk patches to retrieve')
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')
    parser.add_argument('--threshold_attention', type=float, default=None,
                        help='remove attention score below this quantile')
    parser.add_argument('--create_heatmap', action='store_true', help='create heatmap of attention weights')
    parser.add_argument('--save_topk_tiles', action='store_true', help='save topk tiles')
    parser.add_argument('--calibration', action='store_true', help='calibrate predictions')

    args = parser.parse_args()
    apply_random_seed(args.seed)

    cat_df = pd.read_csv(args.category_lesions)
    experiment_path = os.path.join(args.experiment_folder, args.experiment_name)
    Path(os.path.join(experiment_path, "HEATMAPS")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(experiment_path, "TOP_TILES")).mkdir(parents=True, exist_ok=True)
    print('Loading data')
    test_dataset = DatasetPretrained(args.feats_folder_validation, args.slides_label_filepath_validation)
    num_classes = test_dataset.n_classes
    print(len(test_dataset))
    print('done')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_df = pd.DataFrame(columns=["slide_id", "label", "pred",
                                      f"pred_prob_{test_dataset.correspondence_digit_label_name[0]}",
                                      f"pred_prob_{test_dataset.correspondence_digit_label_name[1]}"])

    # Loads MIL model, optimizer and loss function
    print('Loading model')
    feats_size = test_dataset.feat_size
    if args.model == "attention":
        model = GatedAttention(feats_size, num_classes).to(device)
    else:
        raise NotImplementedError

    best_model_path = os.path.join(experiment_path, "best_model.pt")
    if os.path.exists(best_model_path):
        best_model = best_model_path
    else:
        checkpoints = sorted(
            [file for file in os.listdir(experiment_path) if file.startswith("model_") and file.endswith(".pt")],
            key=lambda x: int(x.split('_')[1].split('.')[0]))
        best_model = os.path.join(experiment_path, checkpoints[-1])
    model.load_state_dict(torch.load(best_model, map_location=device)['model'])
    print(f'Best model: {best_model}')
    model.eval()

    if args.calibration:
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()
        print(f"Computing logits on validation set for calibration")
        training_dataset = DatasetPretrained(args.feats_folder_training, args.slides_label_filepath_training)
        _, val_dataset = split_dataset(training_dataset, 0.1)

        # Check that every slide folder has a corresponding label
        assert all([slide_id in val_dataset.dataset.slides_labels.keys() for slide_id in
                    val_dataset.dataset.slides_ids[val_dataset.indices]])
        val_dataloader = get_dataloader(val_dataset, args.batch_size, False, args.j, seed=args.seed)
        logits_val = []
        labels_val = []

        for data, slides_labels, slide_ids in val_dataloader:

            with torch.no_grad():
                predictions = model(data)
            logits_val.extend(predictions.cpu())
            labels_val.extend(slides_labels)

        logits_val = torch.stack(logits_val)
        labels_val = torch.stack(labels_val)
        #labels_val = np.eye(test_dataset.n_classes)[labels_val]

        before_temperature_nll = nll_criterion(logits_val, labels_val).item()
        before_temperature_ece = ece_criterion(logits_val, labels_val).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        t = set_temperature(logits_val, labels_val)
        after_temperature_nll = nll_criterion(temperature_scale(t, logits_val), labels_val).item()
        after_temperature_ece = ece_criterion(temperature_scale(t, logits_val), labels_val).item()
        print('Optimal temperature: %.3f' % t.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    test_dataloader = get_dataloader(test_dataset, args.batch_size, False, args.j, seed=args.seed)

    print('Starting inference...')

    logits_test = []
    prob_test = []
    labels_test = []
    for data, slide_label, slide_id in test_dataloader:
        with torch.no_grad():
            pred, attention_weights = model(data, return_attention=True)
        logits_test.extend(pred.cpu())
        labels_test.extend(slide_label)
        attention_weights = attention_weights[0].squeeze(0).cpu()
        indices = torch.topk(attention_weights, args.k, dim=0)[1].numpy()
        attention_weights = attention_weights.numpy()
        pred_prob = torch.softmax(pred, dim=-1).detach().cpu().numpy()
        prob_test.extend(pred_prob)
        slide_id = slide_id[0]
        slide_label = slide_label[0]
        result_dict = {"slide_id": slide_id,
                        "label": test_dataset.correspondence_digit_label_name[slide_label.item()],
                        "category": cat_df.loc[cat_df.slide_id == slide_id].dg.values[0],
                        "pred": test_dataset.correspondence_digit_label_name[
                            np.argmax(pred_prob, axis=1)[0]],
                        "pred_prob": np.max(pred_prob, axis=1)[0],
                        f"pred_prob_{test_dataset.correspondence_digit_label_name[0]}": pred_prob[
                            0, 0],
                        f"pred_prob_{test_dataset.correspondence_digit_label_name[1]}": pred_prob[
                            0, 1]}
        if args.calibration:
            pred_calibrated = temperature_scale(t, pred.cpu())
            pred_prob_calibrated = torch.softmax(pred_calibrated, dim=-1).detach().cpu().numpy()
            #pred_prob_calibrated = predict_calibrate(pred.cpu().numpy(), t, w, num_classes)
            result_dict["pred_prob_calibrated"] = np.max(pred_prob_calibrated, axis=1)[0]
            result_dict[f"pred_prob_calibrated_{test_dataset.correspondence_digit_label_name[0]}"] = \
                pred_prob_calibrated[0, 0]
            result_dict[f"pred_prob_calibrated_{test_dataset.correspondence_digit_label_name[1]}"] = \
                pred_prob_calibrated[0, 1]
        result_df = pd.concat([result_df, pd.DataFrame([result_dict])], ignore_index=True)

        if args.create_heatmap or args.save_topk_tiles:
            slide_path = os.path.join(args.slides_path, f"{slide_id}.{args.extension}")
            patches_path = os.path.join(args.patch_folder, f"{slide_id}.h5")

        if args.create_heatmap:
            if os.path.exists(os.path.join(experiment_path, "HEATMAPS", f"{slide_id}.png")):
                print(f"Slide {slide_id} heatmap already processed, skipping heatmap processing...")
            else:
                heatmap = create_heatmap(attention_weights, slide_path, patches_path, threshold=args.threshold_attention)
                heatmap.save(os.path.join(experiment_path, "HEATMAPS", f"{slide_id}.png"))
        if args.save_topk_tiles:
            topk_tiles = retrieve_top_k_tiles(indices, slide_path, patches_path)
            if os.path.exists(os.path.join(experiment_path, "TOP_TILES", f"{slide_id}")):
                    print(f"Slide {slide_id} topk tiles already processed, skipping topk tiles processing...")
            else:
                Path(os.path.join(experiment_path, "TOP_TILES", f"{slide_id}")).mkdir(parents=True, exist_ok=True)
                for k, tile in enumerate(topk_tiles):
                    tile.save(os.path.join(experiment_path, "TOP_TILES", f"{slide_id}", f"{k}.jpg"))
    logits_test = torch.stack(logits_test)
    labels_test = torch.stack(labels_test)
    if args.calibration:
        before_temperature_nll = nll_criterion(logits_test, labels_test).item()
        before_temperature_ece = ece_criterion(logits_test, labels_test).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        after_temperature_nll = nll_criterion(temperature_scale(t, logits_test), labels_test).item()
        after_temperature_ece = ece_criterion(temperature_scale(t, logits_test), labels_test).item()
        print('Optimal temperature: %.3f' % t.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        """
        labels_test = np.eye(test_dataset.n_classes)[labels_test]
        ece_before_calibration = ece_eval_binary(prob_test, labels_test)
        print(f"ECE before calibration: {ece_before_calibration}")
        ece_after_calibration = ece_eval_binary(predict_calibrate(logits_test, t, w, test_dataset.n_classes),
                                                labels_test)
        print(f"ECE after calibration: {ece_after_calibration}")
        """
    result_filename = "predictions_detailed_calibrated.csv" if args.calibration else "predictions_detailed.csv"
    result_df.to_csv(os.path.join(experiment_path, result_filename), index=False)
    print("All done")


if __name__ == "__main__":
    main()
