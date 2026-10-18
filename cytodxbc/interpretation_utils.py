import os
import cv2
import tiffslide
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


def init_image(w, h, color_bakground=(255, 255, 255), mask=False):
    """
    :param w: width of image
    :param h: height of image
    :param color_bakground: color of the background
    :param mask: set to True to create a binary mask
    :return: PIL image with the dimension (w,h), in grayscale when mask if set to True
    """
    if mask:
        return np.array(ImageOps.grayscale(Image.new(size=(w, h), mode="RGB", color=0)))
    else:
        return np.array(Image.new(size=(w, h), mode="RGB", color=color_bakground))


def update_canvas(canvas, coord, patch, patch_size, color=None, weighted=False, alpha=0.6):
    """
    :param canvas: image to add the patch
    :param coord: coordinates of the patch on the canvas
    :param patch: image of the patch
    :param patch_size: size of the patch on the canvas
    :param color: color shade to add when weighted
    :param weighted: set to True to add a specific color shade to the patch
    :param alpha: value of the shading
    :return: canvas updated with the patch with an overlapping color (if selected)
    """
    if weighted:
        if color is None:
            print("Please provide a color, choosing red by default")
            color = np.array([255, 0, 0], dtype=np.uint8)
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = cv2.addWeighted(
            np.ones((patch_size[1], patch_size[0], 3), dtype=np.uint8) * color, alpha,
            patch[:patch_size[1], :patch_size[0], :], 1 - alpha, 0, patch[:patch_size[1], :patch_size[0], :])
    else:
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch
    return canvas


def get_attention_color(attention_weight):
    """

    :param attention_weight:
    :return:
    """

    cmap = plt.get_cmap('jet')
    rgba = cmap(attention_weight)
    r, g, b, _ = rgba
    return np.array([r * 255, g * 255, b * 255], dtype=np.uint8)


def create_heatmap(attention_weights, slide_path, patches_path, vis_level=5, threshold=None):
    wsi = tiffslide.TiffSlide(slide_path)
    with h5py.File(os.path.join(patches_path), 'r') as file:
        dset = file['coords']
        coords = dset[:]
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']
    if vis_level >= wsi.level_count:
        vis_level = wsi.level_count - 1
    w, h = wsi.level_dimensions[vis_level]
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    canvas = init_image(w, h)
    downsample = wsi.level_downsamples[vis_level]
    idxs = np.arange(len(coords))
    total = len(idxs)
    patch_size = tuple(np.ceil((np.array(patch_size) / np.array(downsample))).astype(np.int32))
    if threshold:
        scores = np.zeros_like(attention_weights)
        thresh = np.quantile(attention_weights, threshold)
        scores[attention_weights < thresh] = -np.ones_like(scores[attention_weights < thresh])
        scores[attention_weights >= thresh] = to_percentiles(attention_weights[attention_weights >= thresh]) / 100
    else:
        scores = to_percentiles(attention_weights)
        scores /= 100
    for idx in range(total):
        patch_id = idxs[idx]
        coord = coords[patch_id]
        attention_weight = scores[patch_id]
        patch = np.array(wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsample).astype(np.int32)
        if attention_weight == -1:
            canvas = update_canvas(canvas, coord, patch, patch_size)
        else:
            color = get_attention_color(attention_weight)
            canvas = update_canvas(canvas, coord, patch, patch_size, color, weighted=True)
    return Image.fromarray(canvas)


def retrieve_top_k_tiles(indices, slide_path, patches_path):
    slide = tiffslide.TiffSlide(slide_path)
    with h5py.File(patches_path, 'r') as patches:
        coords = patches["coords"][:][indices]
        patch_level = patches["coords"].attrs["patch_level"]
        patch_size = patches["coords"].attrs["patch_size"]
    top_k_patches = []
    for coord in coords:
        patch = slide.read_region(coord, patch_level, (patch_size, patch_size))
        top_k_patches.append(patch)
    return top_k_patches


def plot_attention_weights_distri(attention_weights):
    thresh = np.quantile(attention_weights, 0.995)
    scores = attention_weights[attention_weights > thresh]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.hist(scores, bins=1000, density=True, stacked=True)
    ax.set_xlabel("Attention weights")
    ax.set_ylabel("Number of patches")
    ax.set_title("Distribution of attention weights")
    return fig
