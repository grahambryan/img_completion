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

    def __init__(self, img, **kwargs):
        """StructurePropagation constructor"""
        self.img = img
        self.mask = self.get_mask(img)
        self.img_output = np.zeros_like(self.img)
        self.savepath = kwargs.get("savepath", os.path.abspath(os.curdir))
        self.savefigs = kwargs.get("savefigs", True)

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

    def run(self):
        """Run texture propagation"""
        self.img_output = cv2.inpaint(self.img, self.mask, 3, cv2.INPAINT_NS)
