"""
Texture Propagation

Assumptions:

"""
import os

import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np


class TexturePropagation:
    """Class to propagate image texture (after structure has been propagated)"""

    def __init__(self, img, structure_mask, unknown_mask, patch_size, **kwargs):
        """StructurePropagation constructor"""
        self.img = img
        self.mask = self.get_mask(img)
        self.img_output = np.zeros_like(self.img)
        self.img_output_2 = None
        self.img_output_3 = None
        self.img_output_4 = None
        self.img_output_5 = None
        self.savepath = kwargs.get("savepath", os.path.abspath(os.curdir))
        self.savefigs = kwargs.get("savefigs", True)
        self.unknown_mask = unknown_mask.copy()
        self.structure_mask = structure_mask.copy()
        self.unknown_mask[unknown_mask == self.structure_mask] = False

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

    @staticmethod
    def get_mask(img):
        """Get image mask"""
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) == 0).astype(np.uint8)

    def fill_with_source_texture(self):
        """Fill unknown region with specified texture in img"""
        rows, cols = np.where(self.unknown_mask)
        try:
            img_copy = self.img.copy()
            img_copy[(rows, cols)] = self.img[(rows + (rows.max() - rows.min()), cols)]
            self.img_output_2 = img_copy
        except IndexError:
            pass
        try:
            img_copy = self.img.copy()
            img_copy[(rows, cols)] = self.img[(rows - (rows.max() - rows.min()), cols)]
            self.img_output_3 = img_copy
        except IndexError:
            pass
        try:
            img_copy = self.img.copy()
            img_copy[(rows, cols)] = self.img[(rows, cols - (cols.max() - cols.min()))]
            self.img_output_4 = img_copy
        except IndexError:
            pass
        try:
            img_copy = self.img.copy()
            img_copy[(rows, cols)] = self.img[(rows, cols + (cols.max() - cols.min()))]
            self.img_output_5 = img_copy
        except IndexError:
            pass

    def run(self):
        """Run texture propagation"""
        self.img_output = cv2.inpaint(self.img, self.mask, 3, cv2.INPAINT_NS)
        self.fill_with_source_texture()