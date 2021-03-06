"""Structure Propagation"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class StructurePropagation:
    """Class to propagate image structure"""

    def __init__(self, source, **kwargs):
        """StructurePropagation constructor"""

        # Applicable for all structure masks
        self.source = source
        self.fast = kwargs.get("fast", False)
        self.savepath = kwargs.get("savepath", os.path.abspath(os.curdir))
        self.savefigs = kwargs.get("savefigs", True)
        self.source_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sample_images", source
        )
        self.filenames = {
            "image": os.path.join(self.source_folder, "{}_image.png".format(source)),
            "structure_mask": os.path.join(
                self.source_folder, "{}_image_structure_mask.png".format(source)
            ),
            "unknown_mask": os.path.join(
                self.source_folder, "{}_image_unknown_mask.png".format(source)
            ),
        }
        self.img = self.read_image(self.filenames["image"])
        self.comb_structure_mask, self.unknown_mask = self.read_masks()
        self.num_curves = kwargs.get("num_curves", None)
        self.down_sample = kwargs.get("down_sample", 2)
        self.downsample(self.down_sample if self.fast else 1)
        self.img_orig = self.img.copy()  # Store copy of original image
        self.img_output = self.img.copy()
        self.img_output[self.unknown_mask] = (0, 0, 0)
        self.img_output_rand = self.img.copy()
        self.img_output_rand[self.unknown_mask] = (0, 0, 0)
        self.rows, self.cols, self.channels = self.img.shape
        self.structure_masks = self.split_structure_mask()
        self.patch_size = kwargs.get("patch_size", int(self.img.shape[0] / 30))
        self.sampling_int = kwargs.get("sampling_int", max(int(self.patch_size // 2), 1))

        # Applicable only to current structure mask
        self.structure_mask = np.zeros_like(self.comb_structure_mask)
        self.overlap_mask = np.zeros_like(self.comb_structure_mask)
        self.inv_overlap_mask = np.zeros_like(self.comb_structure_mask)
        self.anchor_points = list()
        self.patch_centers = list()
        self.source_patch_masks = dict()
        self.source_patches = dict()
        self.target_patches = dict()
        self.target_patch_masks = dict()
        self.cost_matrix = np.array([])
        self.min_energy_index = np.array([])
        self.optimal_patch_centers = list()
        self.updated_structure_mask = self.structure_mask

        # Make sure savepath directory exists
        if self.savefigs:
            if not os.path.isdir(self.savepath):
                os.makedirs(self.savepath, exist_ok=True)

    def __repr__(self):
        """Official string representation"""
        repr_str = "{}\n".format(self.__class__.__name__)
        repr_str += "{}\n".format("-" * len(self.__class__.__name__))
        for key, value in sorted(self.__dict__.items()):
            if not key.startswith("_"):
                if (
                    hasattr(value, "__iter__")
                    and not isinstance(value, str)
                    and len(str(value)) > 100
                ):
                    repr_str += "{}: {} ({} items)\n".format(key, type(value), len(value))
                else:
                    repr_str += "{}: {}\n".format(key, value)
        return repr_str

    def __str__(self):
        """Informal string representation"""
        data = list()
        for key, value in sorted(self.__dict__.items()):
            if not key.startswith("_"):
                if (
                    hasattr(value, "__iter__")
                    and not isinstance(value, str)
                    and len(str(value)) > 100
                ):
                    data.append("{}: {} ({} items)".format(key, type(value), len(value)))
                else:
                    data.append("{}: {}".format(key, value))
        return "{}({})".format(self.__class__.__name__, ", ".join(data))

    @staticmethod
    def read_image(filename, grayscale=False):
        """Read image, optionally in grayscale"""
        # Convert to grayscale
        if grayscale:
            return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return cv2.imread(filename, cv2.IMREAD_COLOR)

    @staticmethod
    def plot_img(img, savefig="test.png", **kwargs):
        """Simple image plotting wrapper"""
        plt.figure()
        if img.ndim > 2:
            plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB), **kwargs)
        else:
            plt.imshow(img.astype(np.uint8), **kwargs)
        plt.axis("off")
        if savefig:
            cv2.imwrite(savefig, img.astype(np.uint8))

    def read_masks(self):
        """Reads structure mask and unknown mask"""
        structure_mask = self.read_image(
            self.filenames["structure_mask"], grayscale=True
        ).astype(np.bool)
        unknown_mask = self.read_image(self.filenames["unknown_mask"], grayscale=True).astype(
            np.bool
        )
        return structure_mask, unknown_mask

    def downsample(self, factor):
        """Down sample input image/masks"""
        self.img = self.img[::factor, ::factor, :] if self.fast else self.img
        self.comb_structure_mask = self.comb_structure_mask[::factor, ::factor]
        self.unknown_mask = self.unknown_mask[::factor, ::factor]

    def split_structure_mask(self):
        """
        Splits combined structure mask into separate structure masks, each containing just
          a single line/structure

        Returns:
            list(np.array): List of structure masks, one curve for each (dtype=np.bool)
        """
        # If user has specified only 1 curve
        if not self.num_curves:
            contours = cv2.findContours(
                self.comb_structure_mask.astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
            structure_masks = list()
            for contour in contours:
                contour_mask = np.zeros(self.comb_structure_mask.shape, dtype=np.uint8)
                structure_masks.append(
                    cv2.fillPoly(contour_mask, pts=[contour], color=(255, 255, 255)).astype(
                        np.bool
                    )
                )
            return structure_masks
        return [self.comb_structure_mask]

    def get_overlap_mask(self):
        """Generates the mask where the structure mask and unknown mask overlap"""
        self.overlap_mask = np.bitwise_and(
            self.structure_mask.astype(np.bool), self.unknown_mask.astype(np.bool)
        )

    def get_inv_overlap_mask(self):
        """Generates the inverted overlap mask"""
        self.inv_overlap_mask = self.structure_mask != self.overlap_mask

    def get_anchor_points(self):
        """
        Sparsely samples curve C in the unknown region, Omega, to generate a set of L
        anchor points.
        """
        rows, cols = np.where(self.overlap_mask)
        self.anchor_points = tuple(zip(rows, cols))[:: self.sampling_int]
        print("# of anchors: {}".format(len(self.anchor_points)))

    def get_patch_centers(self):
        """
        Generates the sample set P, which contains the centers of all patches that are within a
          narrow band (1-5 pixels wide) along sample curve C, outside the unknown region, Omega.
        """
        rows, cols = np.where(self.inv_overlap_mask)
        patch_centers = tuple(zip(rows, cols))
        # diff = np.diff(patch_centers)
        # ind_stop_cont = np.where(np.abs(np.diff(np.reshape(diff, diff.shape[0]))) > 1)[0][0]
        self.patch_centers = patch_centers[:: self.sampling_int // 2]
        print("# of samples: {}".format(len(self.patch_centers)))

    def get_patch_mask(self, patch_center):
        """
        Generates patch mask

        Args:
            patch_center (iter(int)): Patch center

        Returns:
            np.array: Boolean mask
        """
        patch_mask = np.zeros((self.rows, self.cols), dtype=np.bool)
        row_start = max(0, patch_center[0] - (self.patch_size // 2))
        row_stop = min(self.rows, patch_center[0] + (self.patch_size // 2) + 1)
        col_start = max(0, patch_center[1] - (self.patch_size // 2))
        col_stop = min(self.cols, patch_center[1] + (self.patch_size // 2) + 1)
        patch_mask[row_start:row_stop, col_start:col_stop] = True
        return patch_mask

    def get_source_patch_masks(self):
        """Determine source patch masks"""
        self.source_patch_masks = {
            patch_center: self.get_patch_mask(patch_center)
            for patch_center in self.patch_centers
            if not np.bitwise_and(self.get_patch_mask(patch_center), self.unknown_mask).any()
        }
        self.patch_centers = tuple(list(self.source_patch_masks.keys()))

    def get_target_patch_masks(self):
        """Determine target patch masks"""
        self.target_patch_masks = {
            patch_center: self.get_patch_mask(patch_center)
            for patch_center in self.anchor_points
        }
        for patch in self.target_patch_masks:
            self.updated_structure_mask[self.target_patch_masks[patch]] = True

    def get_source_patches(self):
        """Determine source patches"""
        self.source_patches = {
            patch_center: self.img[self.source_patch_masks[patch_center]]
            for patch_center in self.patch_centers
            if patch_center in self.source_patch_masks
        }
        # for patch in self.source_patches:
        #     self.plot_img(self.source_patches[patch].reshape(5, 5, 3))

    def get_target_patches(self):
        """Determine source patches"""
        self.target_patches = {
            patch_center: self.img[self.target_patch_masks[patch_center]]
            for patch_center in self.anchor_points
        }

    def get_patches(self):
        """Determines patch masks and match centers for both source and target patches"""
        self.get_source_patch_masks()
        self.get_target_patch_masks()
        self.get_source_patches()

    @staticmethod
    def get_norm_ssd(image, template):
        """
        Computes SSD

        Args:
            image (np.array): Image
            template (np.array): Template

        Returns:
            float: SSD
        """
        try:
            ssd = cv2.matchTemplate(image, template, method=cv2.TM_SQDIFF_NORMED)
            return ssd[0][0]
        except Exception:
            return 0.0

    def get_Es_point(self, source_point, target_point):
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
        normalize_by = np.bitwise_and(
            self.overlap_mask, self.target_patch_masks[target_point]
        ).sum()
        return (
            np.linalg.norm(np.array(source_point) - np.array(target_point))
            + np.linalg.norm(np.array(target_point) - np.array(source_point)) / normalize_by
        )

    def get_Ei_point(self, source_point, target_point):
        """
        Generates Ei point, which constrains the synthesizes patches on the boundary of unknown
          region, Omega, to match well with the known pixels in I - Omega.

        Notes:
            * Ei(xi) is the sum of the normalized squared differences (SSD) calculated in the "red"
                region on boundary patches
            * Ei is set to zero for all other patches inside Omega

        Args:
           source_point (iter(int)): Source point center
           target_point (iter(int)): Target point center

        Returns:
            np.float64: Ei(xi), sum of the normalized squared differences
        """
        Ei = 0.0
        source_patch_mask = self.source_patch_masks[source_point]
        source_patch = self.source_patches[source_point]
        source_patch_unknown_overlap_mask = np.bitwise_and(source_patch_mask, self.unknown_mask)
        target_patch_mask = self.target_patch_masks[target_point]
        # try:
        if source_patch_unknown_overlap_mask.any():
            test_img = self.img.copy()
            test_img[target_patch_mask] = source_patch
            Ei = self.get_norm_ssd(self.img, source_patch)
        return Ei

    @staticmethod
    def get_E1_point(Es_point, Ei_point, ks=50.0, ki=2.0):
        """
        Generates E1 point

        Args:
            Es_point (float): Energy structure point
            Ei_point (float): Energy completion point
            ks (float): Relative weight 1 (default to value in paper)
            ki (float): Relative weight 2 (default to value in paper)

        Returns:
            float: E1 point
        """
        return (ks * Es_point) + (ki * Ei_point)

    def get_E2_point(self, source_point1, source_point2, target_point1, target_point2):
        """
        Generates E2 point, for the energy coherence constraint

        Args:
            source_point1 (tuple): Center of source patch1, P(xk)
            source_point2 (tuple): Center of source patch2, P(xj)
            target_point1 (tuple): Center of target patch left of anchor_point2 at xi-1
            target_point2 (tuple): Center of target patch at xi

        Returns:
            float: Normalized SSD between the overlapped regions of the two patches
        """
        target_overlap_mask = np.bitwise_and(
            self.target_patch_masks[target_point1], self.target_patch_masks[target_point2]
        )
        if target_overlap_mask.any():
            full1 = np.zeros_like(self.img)
            full2 = np.zeros_like(self.img)
            try:
                full1[self.target_patch_masks[target_point1]] = self.source_patches[
                    source_point1
                ]
                full2[self.target_patch_masks[target_point2]] = self.source_patches[
                    source_point2
                ]
            except Exception:
                return 0.0
            return self.get_norm_ssd(full1[target_overlap_mask], full2[target_overlap_mask])
        return 0.0

    def initialize_cost_matrix(self):
        """
        Initialize the first row of the cost matrix, M

        (1) from white board drawing
        """
        self.cost_matrix = np.zeros(
            (len(self.anchor_points), len(self.patch_centers)), dtype=np.float64
        )
        target_point = self.anchor_points[0]
        for i in range(len(self.patch_centers)):
            source_point = self.patch_centers[i]
            Es_point = self.get_Es_point(source_point, target_point)
            Ei_point = self.get_Ei_point(source_point, target_point)
            self.cost_matrix[0][i] = self.get_E1_point(Es_point, Ei_point)

    def get_cost_matrix(self, process_one=False):
        """
        Fill the cost matrix, M, for each node (anchor) with the energy of each sample (patch)

        (2) from white board drawing
        """
        self.initialize_cost_matrix()
        self.min_energy_index = np.ones(self.cost_matrix.shape) * np.inf
        for i in range(1, len(self.anchor_points)):
            print("Processing anchor point {} out of {}...".format(i, len(self.anchor_points)))
            for j in range(len(self.patch_centers)):
                curr_energy = np.inf
                curr_index_at_min_energy = np.inf
                source_point = self.patch_centers[j]
                target_point = self.anchor_points[i]
                Es_point = self.get_Es_point(source_point, target_point)
                Ei_point = self.get_Ei_point(source_point, target_point)
                E1_point = self.get_E1_point(Es_point, Ei_point)
                for k in range(len(self.patch_centers)):
                    source_point1 = self.patch_centers[j]
                    source_point2 = self.patch_centers[k]
                    target_point1 = self.anchor_points[i - 1]
                    target_point2 = target_point
                    E2_point = self.get_E2_point(
                        source_point1, source_point2, target_point1, target_point2
                    )
                    new_energy = self.cost_matrix[i - 1][k] + E2_point
                    if new_energy < curr_energy:
                        curr_energy = new_energy
                        curr_index_at_min_energy = k
                self.cost_matrix[i][j] = E1_point + curr_energy
                self.min_energy_index[i][j] = curr_index_at_min_energy
            if process_one:
                return

    def nodes_min_energy_index(self, node):
        """
        Determine lowest energy sample index for current node

        (3) from white board drawing

        Args:
            node (int): Current node

        Returns:
            idx (int): lowest energy sample index for node
        """
        idx = -1
        curr_energy = np.inf
        for i in range(self.cost_matrix.shape[1]):
            new_energy = self.cost_matrix[node][i]
            if new_energy < curr_energy:
                curr_energy = new_energy
                idx = i
        return idx

    def get_optimal_patches(self):
        """
        Determine the optimal patch centers for each node (anchor)

        (4) from white board drawing
        """
        self.optimal_patch_centers = list()
        # Backtrace through cost to determine optimal samples
        for i in range(self.cost_matrix.shape[0] - 1, -1, -1):
            idx = self.nodes_min_energy_index(i)
            node = self.min_energy_index[i][idx]
            self.optimal_patch_centers.append(node)
        self.optimal_patch_centers.reverse()
        self.optimal_patch_centers = [
            int(patch) for patch in self.optimal_patch_centers if np.isfinite(patch)
        ]
        optimal_patch_centers = list()
        for patch_center in self.optimal_patch_centers:
            if (
                self.source_patches[self.patch_centers[patch_center]].size
                != self.patch_size * self.patch_size * 3
            ):
                node = patch_center - 1 if patch_center > 1 else patch_center + 1
                optimal_patch_centers.append(node)
        if optimal_patch_centers:
            self.optimal_patch_centers = optimal_patch_centers

    def apply_optimal_patches(self):
        """Apply optimal patches"""
        for ind, patch in enumerate(self.optimal_patch_centers):
            source_patch = self.source_patches[self.patch_centers[int(patch)]]
            target_patch = self.target_patch_masks[self.anchor_points[ind]]
            try:
                self.img_output[target_patch] = source_patch
            except Exception:
                print("Couldn't apply patch")

    def apply_random_patches(self):
        """Apply optimal patches"""
        for ind, patch in enumerate(self.optimal_patch_centers):
            source_patch = self.source_patches[
                self.patch_centers[np.random.randint(len(self.patch_centers))]
            ]
            target_patch = self.target_patch_masks[self.anchor_points[ind]]
            try:
                self.img_output_rand[target_patch] = source_patch
            except Exception:
                print("Couldn't apply patch")

    def run(self):
        """Run structure propagation"""
        print("Patch size: {} pixels".format(self.patch_size))
        print("Sampling interval: {} pixels".format(self.sampling_int))

        self.plot_img(self.comb_structure_mask)

        for ind, structure_mask in enumerate(self.structure_masks):
            print(
                "Processing structure: {} out of {}".format(ind + 1, len(self.structure_masks))
            )
            self.structure_mask = structure_mask
            self.updated_structure_mask = structure_mask
            self.get_overlap_mask()
            self.get_inv_overlap_mask()
            self.get_anchor_points()
            self.get_patch_centers()
            self.get_patches()
            self.get_cost_matrix()
            self.get_optimal_patches()
            self.apply_optimal_patches()
            self.apply_random_patches()

    def debug(self):
        """Temp debug method"""
        self.structure_mask = self.structure_masks[2]
        self.get_overlap_mask()
        self.inv_overlap_mask = self.structure_mask != self.overlap_mask
        self.get_anchor_points()
        self.get_patch_centers()
        self.get_patches()
        self.get_cost_matrix()
        self.get_optimal_patches()
        self.apply_optimal_patches()
        self.plot_img(self.img_output)
