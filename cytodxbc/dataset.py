import os
import torch
import numpy as np
import pandas as pd

class DatasetPretrained(torch.utils.data.dataset.Dataset):
    def __init__(self, feats_folder, slides_label_filepath):

        self.feats_folder = feats_folder
        assert os.path.exists(feats_folder)
        self.slides_label_filepath = slides_label_filepath
        assert os.path.exists(slides_label_filepath)

        self.slides_feats, self.slides_ids = self.load_tile_paths()
        self.feat_size = self.slides_feats[0].shape[1]
        self.slides_labels, self.correspondence_digit_label_name = self.load_labels()
        self.n_classes = len(self.correspondence_digit_label_name)

    def load_tile_paths(self):
        """
        """
        slides_feats = [np.load(os.path.join(self.feats_folder, file), mmap_mode="r") for file in os.listdir(self.feats_folder)
                                     if file.endswith(".npy")]
        slides_ids = np.array([os.path.basename(file).replace(".npy", "") for file in os.listdir(self.feats_folder) if file.endswith(".npy")])
        return slides_feats, slides_ids

    def load_labels(self):
        csv_filepath = self.slides_label_filepath
        df = pd.read_csv(csv_filepath)
        slides_ids = df.slide_id.values
        slides_labels = df.label.values
        sorted_index_tiles = np.argsort(slides_ids)
        slides_ids = slides_ids[sorted_index_tiles]
        slides_labels = slides_labels[sorted_index_tiles]

        # Converts labels to digits
        unique_labels = list(set(slides_labels))
        unique_labels.sort()
        assert len(unique_labels) > 1, f'Expected at least two labels, found {len(unique_labels)}'
        label_to_digit = {label: digit for digit, label in enumerate(unique_labels)}
        digit_to_label = {digit: label for digit, label in enumerate(unique_labels)}
        slides_labels = {slide_id: label_to_digit[label] for slide_id, label in zip(slides_ids, slides_labels)}

        return slides_labels, digit_to_label

    def __getitem__(self, slide_index):

        # Load associated slide label
        slide_id = self.slides_ids[slide_index]
        slide_label = self.slides_labels[slide_id]
        
        slide_feats = self.slides_feats[slide_index]
        slide_feats = torch.from_numpy(slide_feats)

        return slide_feats, slide_label, slide_id

    def __len__(self):
        # Length is set to the number of slides
        return len(self.slides_ids)
