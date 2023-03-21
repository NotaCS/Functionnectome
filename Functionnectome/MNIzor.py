#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program to use on data that are alreaady aligned with the MNI space, but with
a resolution or an orientation incompactible with the Functionnectome.

As a reminder, the Functionnectome expects data that are in 2x2x2mm3 MNI space.
The expected orientation affine (written in the header of the nifti file) should be:
    -2    0    0   90
     0    2    0   -126
     0    0    2   -72
     0    0    0    1

Thus, this program is to be used if your affine is not like that.
ONLY to be used if you are SURE you are in MNI space though!
A good way to check is to load your data in a viewer and see if it's aligned with
a brain template known to be in MNI space.

The program does NOT realign MRI data on a template.
"""

import nibabel as nib
from nibabel.processing import resample_from_to
import argparse
import os
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Required argument
    p.add_argument('-i', '--in_file',
                   required=True,
                   help='Path of the input file (.nii, .nii.gz).')
    p.add_argument('-o', '--out_file',
                   required=True,
                   help='Path to the output file to create (.nii, nii.gz).')
    p.add_argument('-so', '--spline_order',
                   default=3,
                   type=int,
                   help="Order for the spline interpolation computed when resampling. Default is 3.")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    inF = args.in_file
    if not os.path.exists(inF):
        raise FileNotFoundError(f'The input file {inF} does not exist.')
    outF = args.out_file
    if os.path.exists(outF):
        raise FileExistsError('The output file already exists')
    spline = args.spline_order

    inIm = nib.load(inF)
    outShape = (91, 109, 91)
    if len(inIm.shape) == 4:
        outShape += (inIm.shape[-1],)
    outAffine = np.array(
        [[-2.,    0.,    0.,   90.],
         [0.,    2.,    0., -126.],
         [0.,    0.,    2.,  -72.],
         [0.,    0.,    0.,    1.]]
    )

    outIm = resample_from_to(inIm, (outShape, outAffine), order=spline)
    nib.save(outIm, outF)


if __name__ == "__main__":
    main()
