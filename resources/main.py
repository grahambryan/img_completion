"""Main entry point script"""
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter

import structprop
import textprop


class ArgumentFormatter(RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    """Add raw description formatting and argument default help formatting to argument parser"""


def get_inputs():
    """
    Extract inputs from command line

    Returns:
        args (argparse.NameSpace): Argparse namespace object
    """
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentFormatter)
    parser.add_argument("-s", "--image-source", help="Image source", default="car")
    parser.add_argument(
        "-i", "--sampling-int", help="Sampling interval", type=int, default=None
    )
    parser.add_argument("-p", "--patch-size", help="Patch size", type=int, default=None)
    parser.add_argument(
        "-c", "--curves", help="Number of curves (structures)", type=int, default=None
    )
    parser.add_argument(
        "-x", "--down-sample", help="Down sample image by", type=int, default=None
    )
    parser.add_argument(
        "-f", "--fast", help="Fast option (downsamples image)", action="store_true"
    )
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    return parser.parse_args()


def main():
    """
    Main entry point for script.

    Returns:
        0 = successful, 1 = unsuccessful
    """
    # TODO: Eventually remove debug opt
    # Extract inputs
    args = get_inputs()
    inputs = dict()
    if args.patch_size:
        inputs["patch_size"] = args.patch_size
    if args.sampling_int:
        inputs["sampling_int"] = args.sampling_int
    if args.sampling_int:
        inputs["num_curves"] = args.curves
    if args.down_sample:
        inputs["down_sample"] = args.down_sample
    struct_prop = structprop.StructurePropagation(args.image_source, fast=args.fast, **inputs)
    if args.debug:
        struct_prop.debug()
        import pdb

        pdb.set_trace()
    struct_prop.run()
    struct_prop.plot_img(
        struct_prop.img_output,
        os.path.join(
            os.path.abspath(os.path.curdir), "{}-struct-prop-only.png".format(args.image_source)
        ),
    )
    text_prop = textprop.TexturePropagation(
        struct_prop.img_output,
        struct_prop.updated_structure_mask,
        struct_prop.unknown_mask,
        struct_prop.patch_size,
    )
    text_prop.run()
    text_prop.plot_img(
        text_prop.img_output,
        os.path.join(
            os.path.abspath(os.path.curdir), "{}-inpaint.png".format(args.image_source)
        ),
    )
    if text_prop.img_output_2 is not None:
        text_prop.plot_img(
            text_prop.img_output_2,
            os.path.join(
                os.path.abspath(os.path.curdir), "{}-txfill-down.png".format(args.image_source)
            ),
        )
    if text_prop.img_output_3 is not None:
        text_prop.plot_img(
            text_prop.img_output_3,
            os.path.join(
                os.path.abspath(os.path.curdir), "{}-txfill-up.png".format(args.image_source)
            ),
        )
    if text_prop.img_output_4 is not None:
        text_prop.plot_img(
            text_prop.img_output_4,
            os.path.join(
                os.path.abspath(os.path.curdir),
                "{}-txfill-col-left.png".format(args.image_source),
            ),
        )
    if text_prop.img_output_5 is not None:
        text_prop.plot_img(
            text_prop.img_output_5,
            os.path.join(
                os.path.abspath(os.path.curdir),
                "{}-txfill-col-right.png".format(args.image_source),
            ),
        )

    return 0


if __name__ == "__main__":
    main()
