#!/usr/bin/env bash
#
# Purpose: Generate all output images for each of the sample image sets under the 'sample_images' directory

# Stop if we hit an exception
set -e

# Barrel
python main.py -s barrel -i 8 -p 7 -x 8 -f  # Best output: inpaint

# Pumpkin
python main.py -s pumpkin -i 15 -p 15 -x 2 -f  # Best output: inpaint

# Car
python main.py -s car -i 3 -p 5 -x 3 -f  # Best output: inpaint

# Concrete
python main.py -s concrete -i 10 -p 15 -x 8 -f  # Best output: col_right

# Stop Sign
python main.py -s stopsign -i 4 -p 14 -x 12 -f  # Best output: inpaint

# Parking
python main.py -s parking -i 5 -p 12 -x 8 -f  # Best output: down

echo "Done!"
