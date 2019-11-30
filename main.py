"""
Main entry point script
"""
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
    struct_prop = structprop.StructurePropagation(args.image_source, fast=args.fast)
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
    text_prop = textprop.TexturePropagation(struct_prop.img_output)
    text_prop.run()
    text_prop.plot_img(
        text_prop.img_output,
        os.path.join(os.path.abspath(os.path.curdir), "{}.png".format(args.image_source)),
    )
    return 0


if __name__ == "__main__":
    main()
