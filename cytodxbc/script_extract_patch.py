import argparse
import logging
import os
import sys
from pathlib import Path

import prismtoolbox as ptb

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Extract patches and stitched previews from whole-slide images")
    parser.add_argument("--slide_directory", type=str, required=True, help="Directory containing slide files")
    parser.add_argument("--patch_directory", type=str, required=True, help="Output directory for patch files")
    parser.add_argument("--stitch_directory", type=str, required=True, help="Output directory for stitched previews")
    parser.add_argument("--engine", type=str, default="tiffslide", choices=["openslide", "tiffslide"], help="Slide reader")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size in pixels")
    parser.add_argument("--patch_level", type=int, default=0, help="Level used for patch extraction")
    parser.add_argument("--overlap", type=int, default=0, help="Patch overlap in pixels")
    parser.add_argument("--slide_ext", type=str, default="ndpi", help="Slide file extension")
    args = parser.parse_args()

    params_patches = {
        "patch_size": args.patch_size,
        "patch_level": args.patch_level,
        "overlap": args.overlap,
        "mode": "all",
        "units": ("px", "micro"),
        "rgb_threshs": (2, 220),
        "use_padding": False,
        "percentages": (0.6, 0.9),
    }
    params_stitch_WSI = {"vis_level": 3, "draw_grid": False}

    Path(args.patch_directory).mkdir(parents=True, exist_ok=True)
    Path(args.stitch_directory).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(args.slide_directory):
        if not file_name.endswith(f".{args.slide_ext}"):
            continue
        slide_path = os.path.join(args.slide_directory, file_name)
        wsi_object = ptb.WSI(slide_path, engine=args.engine)
        print(wsi_object.level_downsamples)

        patch_filename = f"{file_name.split(f'.{args.slide_ext}')[0]}.h5"
        if patch_filename in os.listdir(args.patch_directory):
            continue

        try:
            wsi_object.extract_patches(**params_patches)
            wsi_object.save_patches(args.patch_directory, file_format="h5")

            img = wsi_object.stitch(**params_stitch_WSI)
            img.save(os.path.join(args.stitch_directory, f"{wsi_object.slide_name}.jpg"))
        except Exception as exc:
            print(exc)
            print(f"Error in {file_name}")
            continue


if __name__ == "__main__":
    main()