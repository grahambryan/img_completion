"""
TODO
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import structprop as sp

# Source folder
source = "test2"
source_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sample_images", source
)
img_fn = os.path.join(source_folder, "{}_image.png".format(source))
structure_mask_fn = os.path.join(source_folder, "{}_image_structure_mask.png".format(source))
unknown_mask_fn = os.path.join(source_folder, "{}_image_unknown_mask.png".format(source))

# Read in image and masks
img = sp.read_img(img_fn, color=cv2.IMREAD_COLOR)
structure_mask, unknown_mask = sp.read_masks(structure_mask_fn, unknown_mask_fn)

# Split structure mask by curve
structure_masks = sp.split_structure_mask(structure_mask)

# TODO: Eventually loop through all structure masks
structure_mask = structure_masks[0]
overlap_mask = sp.generate_overlap_mask(structure_mask, unknown_mask)

inv_overlap_mask = structure_mask != overlap_mask

# Determine normalize length
norm_length = structure_mask.sum()

# Determine patch size dynamically based off of image
patch_size = int(img.shape[0] / 20)
sampling_interval = int(patch_size // 2)
print("Patch size: {} pixels".format(patch_size))
print("Sampling interval: {} pixels".format(sampling_interval))

# Generate anchor points
anchor_points = sp.generate_anchor_points(overlap_mask, sampling_interval=sampling_interval)
print("# of anchors: {}".format(len(anchor_points)))
# Generate patch centers
patch_centers = sp.generate_patch_centers(inv_overlap_mask)
print("# of samples: {}".format(len(patch_centers)))

if len(anchor_points) > len(patch_centers):
    print("Uh oh! You have more anchor points than samples")

# Create Cost Matrix
M_0 = np.zeros((len(anchor_points), len(patch_centers)), dtype=np.float64)
M, index_mat_of_min_energy = sp.fill_cost_matrix(
    img, M_0, anchor_points, patch_centers, patch_size, unknown_mask
)
optimal_patches = sp.determine_optimal_patches(M, index_mat_of_min_energy)

# Propagate structure
# img, M = sp.propagate_structure(img, anchor_points, patch_centers, unknown_mask)

# Fake propagate structure
# img[unknown_mask] = (0, 0, 0)
# img = sp.fake_propagate_structure(img, anchor_points, patch_centers, patch_size)
# sp.plot_img(img, "fake-img.png")
# plt.show()
