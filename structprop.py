"""
Structure Propagation

Assumptions:
* Ei equation; not provided in paper; assume Ei is 0
* Relative weights, ks & ki will be set to 1
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Preparation


def plot_img(img, savefig="test.png", **kwargs):
    """
    Plots image

    Args:
        img:
        savefig:
        **kwargs:

    Returns:

    """
    plt.figure()
    plt.imshow(img.astype(np.uint8), **kwargs)
    plt.axis("off")
    if savefig:
        cv2.imwrite(savefig, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def read_img(filename, color=cv2.IMREAD_GRAYSCALE):
    """
    Reads mask

    Args:
        filename (str): Filename

    Returns:

    """
    return cv2.imread(filename, color)


def generate_overlap_mask(structure_mask, unknown_mask):
    """
    Generates the mask where the structure mask and unknown mask overlap.

    Args:
        structure_mask (np.array): dtype=np.float64
        unknown_mask (np.array): dtype=np.float64

    Returns:
        set(tuple): Overlapping points, e.g. [(0, 1), (2, 3), ...]
    """
    return np.bitwise_and(structure_mask.astype(np.bool), unknown_mask.astype(np.bool))


def read_masks(structure_filename, unknown_filename):
    """

    Args:
        structure_filename:
        unknown_filename:

    Returns:

    """
    structure_mask = read_img(structure_filename).astype(np.bool)
    unknown_mask = read_img(unknown_filename).astype(np.bool)
    return structure_mask, unknown_mask, generate_overlap_mask(structure_mask, unknown_mask)


# Structure Propagation


def generate_anchor_points(overlap_mask):
    """
    Sparsely samples curve C in the unknown region, Omega, to generate a set of L
    anchor points.

    Args:
        overlap_mask (np.array): dtype=np.bool

    Returns:
       tuple(tuple): Set of anchor points, L, e.g. ((0, 1), (2, 3), ...)
    """
    rows, cols = np.where(overlap_mask)
    return tuple(zip(rows, cols))[::2]  # TODO: Do we need anchor points on edges?


def generate_one_dimensional_graph(anchor_points):
    """
    Generates a one-dimensional graph, G = {V, E}, where V is the set of L nodes (indices)
      corresponding to the anchor points, and E is the set of all edges connecting adjacent
      nodes on C (sample curve). Generates graph (anchor points with information about
      neighboring points)

    Args:
        anchor_points (set(tuple)): Set of anchor points

    Returns:
        dict(dict): Nested dictionary with graph information
    """
    graph = dict()
    for ind, anchor_point in enumerate(anchor_points):
        graph[ind + 1] = dict()
        graph[ind + 1]["pt"] = anchor_point
        graph[ind + 1]["edges"] = list()
        if ind:
            graph[ind + 1]["edges"].append(anchor_points[ind - 1])
        if ind < (len(anchor_points) - 1):
            graph[ind + 1]["edges"].append(anchor_points[ind + 1])
    return graph


def generate_patch_centers(inv_overlap_mask, sampling_interval=3):
    """
    Generates the sample set P, which contains the centers of all patches that are within a
      narrow band (1-5 pixels wide) along sample curve C, outside the unknown region, Omega.

    Args:
        inv_overlap_mask (np.array): dtype=np.bool
        sampling_interval (int): Sampling interval (in pixels), which is typically half the
                                   patch size

    Returns:
        set(tuple): Set of center points of patches (whose sample size is on order of size
          hundreds or thousands), e.g. length of output should be on that order
    """
    rows, cols = np.where(inv_overlap_mask)
    patch_centers = tuple(zip(rows, cols))
    diff = np.diff(patch_centers)
    ind_stop_cont = np.where(np.abs(np.diff(np.reshape(diff, diff.shape[0]))) > 1)[0][0]
    return patch_centers[:ind_stop_cont:sampling_interval]


def generate_patch_mask(img, patch_center, patch_size):
    """
    Generate patch mask

    Args:
        img:
        patch_center:
        patch_size:

    Returns:

    """
    patch_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    row_start = max(0, patch_center[0] - (patch_size // 2))
    row_stop = min(img.shape[0], patch_center[0] + (patch_size // 2) + 1)
    col_start = max(0, patch_center[1] - (patch_size // 2))
    col_stop = min(img.shape[1], patch_center[1] + (patch_size // 2) + 1)
    patch_mask[row_start: row_stop, col_start: col_stop] = True
    return patch_mask


def generate_patch(img, patch_center, patch_size):
    """
    Generate patch from patch center and patch size.

    Args:
        img (np.array): Image
        patch_center (tuple): Patch center
        patch_size (int): Patch size

    Returns:
        np.array: 2-D array containing patch (mini matrix) of shape (patch_size, patch_size)
    """
    return img[generate_patch_mask(img, patch_center, patch_size)]


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
    patch = generate_patch(img, patch_center, patch_size)
    img[generate_patch_mask(img, patch_dest, patch_size)] = patch
    return img


# Energy Minimization


def generate_Es_point(source_point, target_point):
    """
    Generates Es point, which encodes the structure similarity between the source patch and the
      structure indicated

    Notes:
        * Es(xi) = d(ci, cxi) + d(cxi, ci)
        * d(ci, cxi) = sum(mag(ci(s), cxi))**2) -> the sum of the shortest distance from point
          ci(s) on segment ci to segment, cxi
        * Es(xi) is further normalized by dividing the total number of points in ci

    Args:
        source_point (tuple): Center of source patch, P(xi)
        target_point (tuple): Center of target point with the same patch size, centered at
                                anchor point, pi

    Returns:
        np.float64: Energy similarity at xi, Es(xi)
    """
    return (
        np.linalg.norm(np.array(source_point) - np.array(target_point)) ** 2
        + np.linalg.norm(np.array(target_point) - np.array(source_point)) ** 2
    )  # TODO: Normalize later (and ur mom)


def generate_Ei_point(img, patch_center, patch_size, unknown_mask):
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
    Ei = 0.
    patch_mask = generate_patch_mask(img, patch_center, patch_size)
    patch_unknown_overlap_mask = np.bitwise_and(patch_mask, unknown_mask)
    if patch_unknown_overlap_mask.any():
        np.linalg.norm(img[patch_unknown_overlap_mask]) ** 2
    return Ei  # TODO: Normalize this later??


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


def generate_E2_point(img, patch_center1, patch_center2, patch_size):
    """
    Generates E2 point, for the energy coherence constraint

    Args:
        img (np.array): dtype=np.float64
        patch_center1 (tuple): Center of source patch, P(xi)
        patch_center2 (tuple): Target point with the same patch size, centered at
                                anchor point, p
        patch_size (int): Patch size

    Returns:
        float: Normalized SSD between the overlapped regions of the two patches
    """
    import pdb; pdb.set_trace()
    # 1) Extract overlapped region as np.array
    patch_mask1 = generate_patch_mask(img, patch_center1, patch_size)
    patch_mask2 = generate_patch_mask(img, patch_center2, patch_size)
    patch_overlap_mask = np.bitwise_and(patch_mask1, patch_mask2)
    if patch_overlap_mask.any():
        return np.linalg.norm(img[patch_overlap_mask]) ** 2
    print("Uh oh, your patches don't overlap")
    return 0.  # TODO: Normalize


# Structure Propagation


def propagate_structure(img, anchor_points, patch_centers, unknown_mask, patch_size=5):
    """

    Args:
        img:
        anchor_points:
        patch_centers:

    Returns:

    """
    graph = generate_one_dimensional_graph(anchor_points)
    img = img.copy()
    M = np.zeros((len(graph), len(patch_centers)), dtype=np.float64)
    for ind_node in range(2, len(graph)):
        for ind_patch in range(2, len(patch_centers)):
            Es_point = generate_Es_point(patch_centers[ind_patch], graph[ind_node]["pt"])
            Ei_point = generate_Ei_point(img, patch_centers[ind_patch], patch_size, unknown_mask)
            E1_point = generate_E1_point(Es_point, Ei_point)
            E2_point = generate_E2_point(img, patch_centers[ind_patch - 1], patch_centers[ind_patch], patch_size)
            M[ind_node, ind_patch] = E1_point + E2_point + M[ind_node-1, ind_patch-1]
    return img, M
