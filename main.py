"""
TODO
"""
from . import template

PATCH_SIZE = 5

anchor_points = template.generate_anchor_points()
one_dim_graph = template.generate_one_dimensional_graph(anchor_points)
patch_centers = template.generate_patch_centers()

for anchor_point in anchor_points:
    # TODO: Pick applicable patch_center
    patch_center = (0, 0)
    img = template.apply_patch(img, patch_center, anchor_point, PATCH_SIZE
