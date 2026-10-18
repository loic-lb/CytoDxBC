# CytoDxBC

Official code release for **Fully automatic breast cancer diagnosis from fine-needle aspiration cytology whole slide images using foundation models**.

## Overview

The processing pipeline is:

1. **Patch extraction** — patches are extracted from whole-slide images (default: 256×256 px, `.ndpi` slides read with `tiffslide` or `openslide`).
2. **Feature extraction** — patch-level embeddings are computed with a pre-trained backbone. The results in the paper were obtained with **CytoFM**, **PathoDuet** and **ResNet50 (ImageNet pretrained)/CLAM**; `clam`, `uni` and `conch` are also supported for comparison.
3. **Training** — a gated-attention MIL aggregator is trained on slide-level labels.
4. **Testing** — models are evaluated under 10-fold stratified cross-validation, each fold holding out a test partition for a final unbiased evaluation.
5. **Interpretation** — attention weights are used to build slide heatmaps and retrieve the top-k most predictive patches, with optional temperature calibration of the predictions.

## Repository layout

```
cytodxbc/
├── script_extract_patch.py           # patch extraction from whole-slide images
├── script_extract_feats.py           # patch-level feature extraction (cytofm, clam, uni, pathoduet, conch)
├── create_splits.py                  # stratified cross-validation split generation
├── script_training.py                # train the MIL model on one fold (or on all data)
├── script_testing.py                 # evaluate a trained model (validation or test partition)
├── script_interpretation_all.py      # interpretation of a model trained on all data
├── script_interpretation_crossval.py # interpretation across cross-validation folds
├── model.py / dataset.py / utils.py  # model, dataset and utility code
├── calibration_utils.py              # temperature-scaling calibration utilities
├── cytofm_model.py                   # CytoFM backbone (expects ./cytofm_weights.pth)
└── stats.R                           # statistical analyses and figures from the CV predictions
```

## Setup

Create the environment from `environment.yml`, then run all commands from the repository root:

```bash
conda env create -f environment.yml
conda activate cyto_env
```

Notes:

- The CytoFM backbone expects its checkpoint at `./cytofm_weights.pth` (repository root). The checkpoint is not distributed with this code (see https://github.com/ivezicV/CytoFM).
- Backbones other than CytoFM are downloaded from the Hugging Face Hub; use `--hug_login` in `script_extract_feats.py` if needed.
- `--engine openslide` requires the OpenSlide C library, which is not installed by `environment.yml`. The default engine `tiffslide` has no system dependency.
- `stats.R` requires the R packages dplyr, ggplot2, pROC, caret, tidyr, stringr, rms and cartographr (for `theme_poster`, needs R ≥ 4.2).

## Input format

The pipeline expects:

- a directory of whole-slide images (default extension `.ndpi`),
- a CSV file of slide labels with `slide_id` and `label` columns. Labels are mapped to digits alphabetically (`benign` → 0, `malignant` → 1); malignant is the positive class in all metrics,
- a CSV file of lesion categories with `slide_id` and `dg` columns, required by the interpretation scripts and `stats.R`.

Everything is matched by `slide_id`: feature files must be named `<slide_id>.npy`.

## Splits

Generate the 10-fold stratified cross-validation splits from the labels file:

```bash
python cytodxbc/create_splits.py \
  --dataset labels.csv \
  --output cytodxbc/splits/splits_regular_crossval \
  --n_splits 10 --val_frac 0.1 --seed 1 \
  --regular_cross_validation
```

Each `split_<i>.csv` has `train`, `val` and `test` columns holding slide IDs. Training uses the `train` and `val` columns for model fitting and validation; the `test` column is the held-out partition used for the final unbiased evaluation. Testing evaluates on `val` if `--validation` is set, otherwise on `test`.

## Reproducing the pipeline

Run each step from the repository root. The paper used three backbones (cytofm, pathoduet and resnet50) and two evaluation settings. The commands below use CytoFM; replace the feature folder for the other backbone.

**1. Extract patches from the slides:**

```bash
python cytodxbc/script_extract_patch.py \
  --slide_directory PATH_TO_SLIDES \
  --patch_directory PATH_TO_PATCHES \
  --stitch_directory PATH_TO_STITCHED
```

**2. Extract patch features:**

```bash
python cytodxbc/script_extract_feats.py \
  --slide_directory PATH_TO_SLIDES \
  --patch_directory PATH_TO_PATCHES \
  --arch_name cytofm \
  --results_directory PATH_TO_OUTPUT
```

Features are written to `<results_directory>/feats_256_cytofm/npy_files/<slide_id>.npy`; use that `npy_files` folder as `--feats_folder` in the next steps.

**3. Cross-validation (main cohort)** — train and evaluate every fold:

```bash
for split_id in 0 1 2 3 4 5 6 7 8 9; do
python cytodxbc/script_training.py \
  --feats_folder PATH_TO_FEATURES/feats_256_cytofm/npy_files \
  --slides_label_filepath labels.csv \
  --split_folder cytodxbc/splits/splits_regular_crossval \
  --split_id $split_id \
  --batch_size 1 --lr 2e-4 --reg 1e-7 --epochs 100 --optimizer AdamW

python cytodxbc/script_testing.py \
  --experiment_folder ./experiments \
  --experiment_name CytoDxBC \
  --feats_folder PATH_TO_FEATURES/feats_256_cytofm/npy_files \
  --slides_label_filepath labels.csv \
  --split_folder cytodxbc/splits/splits_regular_crossval \
  --split_id $split_id
done
```

Per-fold metrics are written to `experiments/CytoDxBC/results.csv` (or `results_val.csv` with `--validation`) and per-slide predictions to `predictions.csv` / `predictions_val.csv`. Re-running a split replaces its previous rows.

**4. Interpretation and calibration:**

```bash
python cytodxbc/script_interpretation_crossval.py \
  --experiment_folder ./experiments \
  --experiment_name CytoDxBC \
  --feats_folder PATH_TO_FEATURES/feats_256_cytofm/npy_files \
  --slides_label_filepath labels.csv \
  --split_folder cytodxbc/splits/splits_regular_crossval \
  --category_lesions category_lesions.csv \
  --calibration
```

This writes `experiments/CytoDxBC/predictions_detailed.csv`, or `predictions_detailed_calibrated.csv` with `--calibration`; the temperature is fitted on each fold's validation split. Add `--slides_path PATH_TO_SLIDES --patch_folder PATH_TO_PATCHES` to also build attention heatmaps (in `HEATMAPS/`) and export the top-k patches (in `TOP_TILES/`).

**5. External validation (second cohort)** — train on all data of the first cohort, then evaluate and interpret on the second cohort:

```bash
python cytodxbc/script_training.py \
  --feats_folder PATH_TO_COHORT1_FEATURES/feats_256_cytofm/npy_files \
  --slides_label_filepath labels_cohort1.csv \
  --batch_size 1 --lr 2e-4 --reg 1e-7 --epochs 100 --optimizer AdamW

python cytodxbc/script_testing.py \
  --experiment_folder ./experiments \
  --experiment_name CytoDxBC \
  --feats_folder PATH_TO_COHORT2_FEATURES/feats_256_cytofm/npy_files \
  --slides_label_filepath labels_cohort2.csv

python cytodxbc/script_interpretation_all.py \
  --experiment_folder ./experiments \
  --experiment_name CytoDxBC \
  --feats_folder_training PATH_TO_COHORT1_FEATURES/feats_256_cytofm/npy_files \
  --feats_folder_validation PATH_TO_COHORT2_FEATURES/feats_256_cytofm/npy_files \
  --slides_label_filepath_training labels_cohort1.csv \
  --slides_label_filepath_validation labels_cohort2.csv \
  --category_lesions category_lesions.csv \
  --slides_path PATH_TO_COHORT2_SLIDES \
  --patch_folder PATH_TO_COHORT2_PATCHES \
  --create_heatmap --save_topk_tiles --calibration
```

**6. Statistics and figures:**

```bash
Rscript cytodxbc/stats.R
```

`stats.R` reproduces the per-fold metrics, confidence intervals and figures of the paper. It reads `experiments/CytoDxBC/predictions_detailed_calibrated.csv`, `experiments/CytoDxBC/results.csv`, `experiments/CytoDxBC/predictions_val.csv` and `./category_lesions.csv` (adjust the paths at the top of the script). Run step 4 with `--calibration` first so that the calibrated probabilities and file name are available.

Checkpoint selection: `script_training.py` saves `model_<epoch>.pt` in `experiments/CytoDxBC/Split <i>/` every time the validation loss improves, so the last saved checkpoint is the best one; testing and interpretation load it automatically.

All scripts expose additional hyperparameters (epochs, learning rate, patch size, number of top-k patches, seeds, …) through `--help`.

## License

This code is released under the [MIT License](LICENSE).
