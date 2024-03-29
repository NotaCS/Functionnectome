#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The program reduces the compression of a given priors .h5 file.
This significantly increase the space it takes on the disk, but also
speeds up the Functionnectome process quite a bit.

To use it, you simply need to give it the file priors file you want
to decompress, and the file path (with the file name) where you want
to save the resulting file.

Note that to use the new priors, you will need to input their path in
as "Custom priors" or replace the current default priors with them
(e.g. by ensuring that the file path used by the old priors now points
to the new ones).
"""

import h5py
import argparse
import os


def changeCompPriors(inF, outF, compType='lzf', comptStr=4, thr=None, verbose=True):
    """
    Function used to change the level and/or method of compression of the volumes
    contained in the HDF5 priors.

    Parameters
    ----------
    inF : str
        Path to the input priors.
    outF : str
        Path (including file name) of the output HDF5 file to generate.
    compType : str, optional
        Type of compression algorithm. Can be 'gzip' or 'lzf'. The default is 'lzf'.
    comptStr : int, optional
        Level of compression if using 'gzip'. Can be between 1 and 10. The default is 4.
    thr : float, optional
        Threshold to apply to the connectivity maps. This can be usefull for very big
        (e.g. whole-brain) priors. 0-ing the low values can drammatically reduce the
        size of the file, while having little (but still some) impact on the final results.
        The default is None.
    verbose : bool, optional
        If True, print the number of maps left every once in a while. Default is True.

    """
    if compType == 'lzf':
        comptStr = None

    with h5py.File(outF, "a") as h5fout:
        with h5py.File(inF, "r") as h5fin:
            h5fin.copy(h5fin['mask_region'], h5fout['/'], 'mask_region')
            h5fin.copy(h5fin['template'], h5fout['/'], 'template')
            grp_reg = h5fout.create_group('tract_region')
            regs = h5fin['tract_region'].keys()
            if len(regs):
                grp_reg.attrs['header'] = h5fin['tract_region'].attrs['header']
                if verbose:
                    print('Regionwise priors')
                for ii, reg in enumerate(regs):
                    if verbose and ii % 5 == 0:
                        print(f'{ii+1}/{len(regs)}')
                    vol = h5fin['tract_region'][reg][:]
                    if thr:
                        vol[vol <= thr] = 0
                    grp_reg.create_dataset(reg, data=vol, maxshape=vol.shape,
                                           compression=compType, compression_opts=comptStr,
                                           chunks=vol.shape)
            grp_vox = h5fout.create_group('tract_voxel')
            voxs = h5fin['tract_voxel'].keys()
            if len(voxs):
                grp_vox.attrs['header'] = h5fin['tract_voxel'].attrs['header']
                if verbose and len(voxs) > 1:
                    print('Voxelwise priors')
                for ii, vox in enumerate(voxs):
                    if ii % 1000 == 0 and verbose:
                        print(f'{ii//1000}k/{len(voxs)//1000}k')
                    vol = h5fin['tract_voxel'][vox][:]
                    if thr:
                        vol[vol <= thr] = 0
                    grp_vox.create_dataset(vox, data=vol, maxshape=vol.shape,
                                           compression=compType, compression_opts=comptStr, chunks=vol.shape)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Positional argument (obligatory)
    p.add_argument('inPriors',
                   help='Path of the input priors to decompress.')
    p.add_argument('outPriors',
                   help=(
                       'Path of the output (decompressed) priors '
                       'to generate. Must end in ".h5"'
                   )
                   )

    # Optional arguments
    p.add_argument('-s', '--silent',
                   action='store_true',
                   help=(
                       'Silences the program (no prints of advances)'
                   )
                   )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not os.path.isfile(args.inPriors):
        parser.error(
            f"The input file '{args.inPriors}' does not exist. Check and correct the path, then retry."
        )
    if os.path.exists(args.outPriors):
        parser.error(
            f"The input file '{args.outPriors}' already exists. Check and correct the path, then retry."
        )
    verbose = not args.silent
    changeCompPriors(args.inPriors, args.outPriors, verbose=verbose)


if __name__ == '__main__':
    main()
