"""
Structure Propagation

Assumptions:
* Ei equation; not provided in paper; assume Ei is 0
* Relative weights, ks & ki will be set to 1
"""
import matplotlib.pyplot as plt
import numpy as np
import numba
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
    return structure_mask, unknown_mask


def split_structure_mask(structure_mask):
    """
    Split structure mask

    Args:
        structure_mask (np.array): dtype=np.bool

    Returns:
        list(np.array): List of structure masks, one curve for each (dtype=np.bool)
    """
    contours = cv2.findContours(
        structure_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    structure_masks = list()
    for contour in contours:
        contour_mask = np.zeros(structure_mask.shape, dtype=np.uint8)
        structure_masks.append(
            cv2.fillPoly(contour_mask, pts=[contour], color=(255, 255, 255)).astype(np.bool)
        )
    return structure_masks


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
    # diff = np.diff(patch_centers)
    # ind_stop_cont = np.where(np.abs(np.diff(np.reshape(diff, diff.shape[0]))) > 1)[0][0]
    return patch_centers[::sampling_interval]


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
    patch_mask[row_start:row_stop, col_start:col_stop] = True
    # print(row_start, row_stop, col_start, col_stop)
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
    return img[generate_patch_mask(img, patch_center, patch_size)]  # TODO: Edge case cause prob


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


def compute_ssd(*args):
    """
    Computes SSD

    Args:
        *args:

    Returns:

    """
    # TODO: Use cv2.templateMatch
    return 0.0


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
    return np.linalg.norm(np.array(source_point) - np.array(target_point)) + np.linalg.norm(
        np.array(target_point) - np.array(source_point)
    )  # TODO: Normalize by length of curve in patch


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
    # TODO: I think we need to pass in anchor point (see if its on the edge of the unkown region)
    Ei = 0.0
    patch_mask = generate_patch_mask(img, patch_center, patch_size)
    patch_unknown_overlap_mask = np.bitwise_and(patch_mask, unknown_mask)
    if patch_unknown_overlap_mask.any():
        Ei = compute_ssd(patch_unknown_overlap_mask)
    return Ei


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


def generate_E2_point(
    img, patch_center1, patch_center2, anchor_point1, anchor_point2, patch_size
):
    """
    Generates E2 point, for the energy coherence constraint

    Args:
        img (np.array): dtype=np.float64
        patch_center1 (tuple): Center of source patch1, P(xk)
        patch_center2 (tuple): Center of source patch2, P(xj)
        anchor_point1 (tuple): Center of target patch left of anchor_point2 at xi-1
        anchor_point2 (tuple): Center of target patch at xi
        patch_size (int): Patch size

    Returns:
        float: Normalized SSD between the overlapped regions of the two patches
    """
    # TODO: Need node 1 and node 2 (anchor points)
    # TODO: Need to find the overlap of the patch on node 1 and node 2
    # 1) Extract overlapped region in unknown region as np.array
    # patch_mask1 = generate_patch_mask(img, patch_center1, patch_size)
    # patch_mask2 = generate_patch_mask(img, patch_center2, patch_size)
    patch_at_target_mask1 = generate_patch_mask(img, anchor_point1, patch_size)
    patch_at_target_mask2 = generate_patch_mask(img, anchor_point2, patch_size)
    anchor_overlap_mask = np.bitwise_and(patch_at_target_mask1, patch_at_target_mask2)
    if anchor_overlap_mask.any():
        patch1 = generate_patch(img, patch_center1, patch_size)
        patch2 = generate_patch(img, patch_center2, patch_size)
        # TODO: Feed ssd the values of the overlap of patch1 and patch2 in anchor_overlap
        return compute_ssd(anchor_overlap_mask)  # TODO
    print("Uh oh, your patches don't overlap")
    return 0.0  # TODO: Normalize


# Dynamic Programming


def initialize_cost_matrix(img, M, anchor_points, patch_centers, patch_size, unknown_mask):
    """
    Initialize the first row of the cost matrix, M

    (1) from white board drawing

    Args:
        img (np.array): dtype=np.float64
        M (np.array): dtype=np.float64
        anchor_points:
        patch_centers:
        patch_size (int): Patch size
        unknown_mask (np.array): dtype=np.float64

    Returns:
        M (np.array): dtype=np.float64

    """
    target_point = anchor_points[0]
    for i in range(len(patch_centers)):
        source_point = patch_centers[i]
        Es_point = generate_Es_point(source_point, target_point)
        Ei_point = generate_Ei_point(img, source_point, patch_size, unknown_mask)
        M[0][patch_centers[i]] = generate_E1_point(Es_point, Ei_point)
    return M


def fill_cost_matrix(img, M, anchor_points, patch_centers, patch_size, unknown_mask):
    """
    Fill the cost matrix, M, for each node (anchor) with the energy of each sample (patch)

    (2) from white board drawing

    Args:
        img (np.array): dtype=np.float64
        M (np.array): dtype=np.float64
        anchor_points:
        patch_centers:
        patch_size (int): Patch size
        unknown_mask (np.array): dtype=np.float64

    Returns:
        M (np.array): dtype=np.float64
        index_mat_of_min_energy (np.array): dtype=np.uint8

    """
    index_mat_of_min_energy = np.mat(np.ones(M.shape) * np.inf)
    for i in range(1, len(anchor_points)):
        for j in range(len(patch_centers)):
            curr_energy = np.inf
            curr_index_at_min_energy = np.inf
            source_point = patch_centers[j]
            target_point = anchor_points[i]
            Es_point = generate_Es_point(source_point, target_point)
            Ei_point = generate_Ei_point(img, source_point, patch_size, unknown_mask)
            E1_point = generate_E1_point(Es_point, Ei_point)
            for k in range(len(patch_centers)):
                source_point1 = patch_centers[k]
                source_point2 = patch_centers[j]
                target_point1 = anchor_points[i - 1]
                target_point2 = target_point
                E2_point = generate_E2_point(
                    img, source_point1, source_point2, target_point1, target_point2, patch_size
                )
                new_energy = M[i - 1][k] + E2_point
                if new_energy < curr_energy:
                    curr_energy = new_energy
                    curr_index_at_min_energy = k
            M[i][j] = E1_point + curr_energy
            index_mat_of_min_energy[i][j] = curr_index_at_min_energy
    return M, index_mat_of_min_energy.astype(np.uint8)


def nodes_min_energy_index(M, node):
    """
    Determine lowest energy sample index for current node

    (3) from white board drawing

    Args:
        M (np.array): dtype=np.float64
        node (int): Current node

    Returns:
        idx (int): lowest energy sample index for node
    """
    curr_energy = np.inf
    for i in range(M.shape[1]):
        new_energy = M[node][i]
        if new_energy < curr_energy:
            curr_energy = new_energy
            idx = i
    return idx

def determine_optimal_patches(M, index_mat_of_min_energy):
    """
    Determine the optimal patch centers for each node (anchor)

    (4) from white board drawing

    Args:
        M (np.array): dtype=np.float64
        index_mat_of_min_energy (np.array): dtype=np.uint8

    Returns:
        optimal_patch_centers (np.array): dtype=np.uint8

    """
    optimal_patch_centers = list()
    # Backtrace through cost to determine optimal samples
    for i in range(M.shape[0] - 1, -1, -1):
        idx = nodes_min_energy_index(M, i)
        optimal_patch_centers.append(index_mat_of_min_energy[i][idx])
    return np.array(optimal_patch_centers.reverse()).astype(np.uint8)

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
            Ei_point = generate_Ei_point(
                img, patch_centers[ind_patch], patch_size, unknown_mask
            )
            E1_point = generate_E1_point(Es_point, Ei_point)
            E2_point = generate_E2_point(
                img, patch_centers[ind_patch - 1], patch_centers[ind_patch], patch_size
            )
            M[ind_node, ind_patch] = E1_point + E2_point + M[ind_node - 1, ind_patch - 1]
    return img, M


def fake_propagate_structure(img, anchor_points, patch_centers, patch_size):
    """Fake propagate structure"""
    img = img.copy()
    for anchor_point in anchor_points:
        try:
            img = apply_patch(
                img,
                patch_centers[np.random.randint(len(patch_centers))],
                anchor_point,
                patch_size,
            )
        except Exception:
            pass
    return img
