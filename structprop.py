"""
Template

Assumptions:
* Ei equation; not provided in paper; assume Ei is 0
* Relative weights, ks & ki will be set to 1
"""
import numpy as np
import cv2

# Preparation


def generate_overlap_mask(structure_mask, unknown_mask):
    """
    Generates the mask where the structure mask and unknown mask overlap.

    Args:
        structure_mask (np.array): dtype=np.float64
        unknown_mask (np.array): dtype=np.float64

    Returns:
        set(tuple): Overlapping points, e.g. [(0, 1), (2, 3), ...]
    """
    return np.bitwise_and(structure_mask, unknown_mask)


# Structure Propagation


def generate_anchor_points(img, structure_mask, unknown_mask):
    """
    Sparsely samples curve C in the unknown region, Omega, to generate a set of L
    anchor points.

    Args:
        img (np.array): dtype=np.float64
        structure_mask (np.array): dtype=np.float64
        unknown_mask (np.array): dtype=np.float64

    Returns:
       set(tuple): Set of anchor points, L, e.g. [(0, 1), (2, 3), ...]
    """


def generate_one_dimensional_graph(anchor_points):
    """
    Generates a one-dimensional graph, G = {V, E}, where V is the set of L nodes (indices)
      corresponding to the anchor points, and E is the set of all edges connecting adjacent
      nodes on C (sample curve).

    Args:
        anchor_points (set(tuple)): Set of anchor points

    Returns:
        dict: Edges connecting adjacent nodes on C (keys=tuple for anchor point, value=list of
          points (tuples) that each anchor point is connected to, e.g.:
          {1, 1}: ((0, 1), (1, 2)}
    """


def generate_patch_centers(img, structure_mask, unknown_mask, sampling_interval):
    """
    Generates the sample set P, which contains the centers of all patches that are within a
      narrow band (1-5 pixels wide) along sample curve C, outside the unknown region, Omega.

    Args:
        img (np.array): dtype=np.float64
        structure_mask (np.array): dtype=np.float64
        unknown_mask (np.array): dtype=np.float64
        sampling_interval (int): Sampling interval (in pixels), which is typically half the
                                   patch size

    Returns:
        set(tuple): Set of center points of patches (whose sample size is on order of size
          hundreds or thousands), e.g. length of output should be on that order
    """


def generate_patch(patch_center, patch_size):
    """
    Generate patch from patch center and patch size.

    Args:
        patch_center (tuple): Patch center
        patch_size (int): Patch size

    Returns:
        np.array: 2-D array containing patch (mini matrix) of shape (patch_size, patch_size)
    """


def apply_patch(img, patch_center, patch_dest, patch_size):
    """
    Applies the patch located at patch_center to the to-be-filled in patch in the unknown
      region, Omega, of shape (patch_size, patch_size).

    Args:
        img (np.array): dtype=np.float64
        patch_center (tuple): Patch center
        patch_dest (tuple): Patch center
        patch_size (int): Patch size

    Returns:
        np.array: Image with patch in unknown region, Omega, filled in
    """


# Energy Minimization


def generate_Es_point(img, source_point, target_point):
    """
    Generates Es point, which encodes the structure similarity between the source patch and the
      structure indicated

    Notes:
        * Es(xi) = d(ci, cxi) + d(cxi, ci)
        * d(ci, cxi) = sum(mag(ci(s), cxi))**2) -> the sum of the shortest distance from point
          ci(s) on segment ci to segment, cxi
        * Es(xi) is further normalized by dividing the total number of points in ci

    Args:
        img (np.array): dtype=np.float64
        source_point (tuple): Center of source patch, P(xi)
        target_point (tuple): Center of target point with the same patch size, centered at
                                anchor point, pi

    Returns:
        np.float64: Energy similarity at xi, Es(xi)
    """
    # TODO: Patch size?


def generate_Ei_point(img, source_point, patch_size):
    """
    Generates Ei point, which constrains the synthesizes patches on the boundary of unknown
      region, Omega, to match well with the known pixels in I - Omega.

    Notes:
        * Ei(xi) is the sum of the normalized squared differences (SSD) calculated in the "red"
            region on boundary patches
        * Ei is set to zero for all other patches inside Omega

    Args:
        img (np.array): dtype=np.float64
        source_point (tuple): Center of source patch, P(xi)
        patch_size (int): Patch size

    Returns:
        np.float64: Ei(xi), sum of the normalized squared differences
    """
    # TODO
    return 0.


def generate_E1_point(Es_point, Ei_point, ks=1.0, ki=1.0):
    """
    Generates E1 point

    Args:
        Es_point (float): Energy structure point
        Ei_point (float): Energy completion point
        ks (float): Relative weight 1
        ki (float): Relative weight 2

    Returns:
        float: E1 point
    """
    return (ks * Es_point) + (ki * Ei_point)


def generate_E2_point(img, source_point, target_point, patch_size):
    """
    Generates E2 point, for the energy coherence constraint

    Args:
        img (np.array): dtype=np.float64
        source_point (tuple): Center of source patch, P(xi)
        target_point (tuple): Target point with the same patch size, centered at
                                anchor point, p
        patch_size (int): Patch size

    Returns:
        float: Normalized SSD between the overlapped regions of the two patches
    """
    # 1) Extract overlapped region as np.array
    # 2) Compute SSD of overlapped region
    # 3) Normalize by dividing by the total number of points in ci
    return 0.

# Structure Propagation


def do_stuff(img, graph, patch_size):
    # Step 1
    messages = {node: 0 for node in graph}
    # Step 2
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            Es_point = generate_Es_point(img, node, neighbor)
            Ei_point = generate_Ei_point(img, node, patch_size)
            E1_point = generate_E1_point(Es_point, Ei_point)
            E2_point = generate_E2_point(img, node, neighbor, patch_size)
            cum_energy = 0
            # TODO: There's a min() around this apparently
            messages[node] = E1_point + E2_point + cum_energy
    # Step 3
