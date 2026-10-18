import os
import argparse
import torch
import prismtoolbox.wsiemb as ptb_emb
from pathlib import Path

from cytofm_model import create_cytoFM_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from slides with patches already extracted")
    parser.add_argument("--slide_directory", type=str, help="Path to the directory containing the slides")
    parser.add_argument("--engine", type=str, default="tiffslide", choices=["openslide", "tiffslide"],
                        help="Engine to use for reading the slides")
    parser.add_argument("--patch_directory", type=str, help="Path to the directory containing the patches")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to extract (in pixels)")
    parser.add_argument("--slide_ext", type=str, default="ndpi", help="Extension of the slide files")
    parser.add_argument("--arch_name", type=str, default="clam", choices=["clam", "uni", "pathoduet", "conch", "cytofm"],
                        help="Name of the architecture to use for the embeddings")
    parser.add_argument("--hug_login", type=str, default=None, help="Login for the Huggingface API")
    parser.add_argument("--results_directory", type=str, default="./results", help="Path to the directory where the results will be saved")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the following parameters to extract embeddings from the patches
    if args.arch_name == "clam":
        arch_name = "clam"
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms_dict = {"totensor": {}, "normalize": {"mean": mean, "std": std}}
        pretrained_weights = "IMAGENET1K_V1"
        custom_model = None
        custom_model_name = None
    elif args.arch_name == "uni":
        arch_name = "uni"
        pretrained_weights = None
        transforms_dict = None
        custom_model = None
        custom_model_name = None
    elif args.arch_name == "conch":
        arch_name = "conch"
        pretrained_weights = None
        transforms_dict = None
        custom_model = None
        custom_model_name = None
    elif args.arch_name == "pathoduet":
        arch_name = "pathoduet"
        pretrained_weights = "pathoduet_HE"
        transforms_dict = None
        custom_model = None
        custom_model_name = None
    elif args.arch_name == "cytofm":
        arch_name = None
        pretrained_weights = None
        custom_model = create_cytoFM_model()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms_dict = {"resize": {"size": 224}, "totensor": {}, "normalize": {"mean": mean, "std": std}}
        custom_model_name = "cytoFM"
        
    params_embedder = {"arch_name": arch_name,
                       "custom_model": custom_model,
                       "custom_model_name": custom_model_name,
                       "pretrained_weights": pretrained_weights,
                       "transforms_dict": transforms_dict,
                       "batch_size": 512,
                       "num_workers": 4,
                       "device": device,
                       "engine": args.engine,
                       "hug_login": args.hug_login}

    # Path to the directory where the embeddings will be saved as pt files
    directory_embeddings = os.path.join(args.results_directory, f"./feats_{args.patch_size}_{args.arch_name}", "npy_files")
    Path(directory_embeddings).mkdir(parents=True, exist_ok=True)

    slide_embedder = ptb_emb.SlideEmbedder(slide_dir=args.slide_directory,
                                           coords_dir=args.patch_directory,
                                           **params_embedder)
    print(f'Extracting embeddings from slides with {slide_embedder.arch_name} architecture, and transforms: {slide_embedder.get_transforms()}')

    for slide in os.listdir(args.slide_directory):
        if not slide.endswith(f".{args.slide_ext}"):
            continue
        slide_name = slide.split(f".{args.slide_ext}")[0]
        #slide_name = ptb.WSI(os.path.join(args.slide_directory, slide)).slide_name
        if os.path.exists(os.path.join(directory_embeddings, f"{slide_name}.npy")):
            print(f"Embeddings for {slide_name} already exist, skipping...")
            continue
        # Extract the embeddings
        slide_embedder.extract_model_based_embeddings(slide_name, slide_ext=args.slide_ext)
        slide_embedder.save_embeddings(directory_embeddings, flush_memory=True, merge=True, format="npy")
