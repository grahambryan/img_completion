"""
TODO
"""
import os

import cv2

import structprop as sp

# Source folder
source = "test"
source_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images", source)
img_fn = os.path.join(source_folder, "{}_image.png".format(source))
structure_mask_fn = os.path.join(source_folder, "{}_image_structure_mask.png".format(source))
unknown_mask_fn = os.path.join(source_folder, "{}_image_unknown_mask.png".format(source))

# Read in image and masks
img = sp.read_img(img_fn, color=cv2.IMREAD_COLOR)
structure_mask, unknown_mask, overlap_mask = sp.read_masks(structure_mask_fn, unknown_mask_fn)
inv_overlap_mask = structure_mask != overlap_mask

# Determine normalize length
norm_length = structure_mask.sum()

# Determine patch size dynamically based off of image
patch_size = int(img.shape[0] / 20)
sampling_interval = int(patch_size // 2)
print("Patch size: {} pixels".format(patch_size))
print("Sampling interval: {} pixels".format(sampling_interval))

# Generate anchor points
anchor_points = sp.generate_anchor_points(overlap_mask)

# Generate patch centers
patch_centers = sp.generate_patch_centers(inv_overlap_mask)

# Propagate structure
img, M = sp.propagate_structure(img, anchor_points, patch_centers, unknown_mask)

#print(M)