#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates regionwise priors, stored in a HDF5 (.h5) file.
Leverage voxelwise priors and the QuickDisco algorithm
to estimate the connectivity of each regions in an atlas.

It can be used to complete an existing priors file, or to
make a new one purely with regionwise priors (see the 'mode'
option).

"""

import nibabel as nib
import numpy as np
import os
import h5py
from Functionnectome.quickDisco import probaMap_fromROI
from Functionnectome.functionnectome import checkOrient_load
from Functionnectome.makeHDF5prior import header2string
import argparse
import sys
from Functionnectome.quickDisco import checkH5


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def _build_arg_parser(h5Labels):
    p = MyParser(formatter_class=argparse.RawTextHelpFormatter,
                 description=__doc__)
    p.add_argument('-pi', '--priorsIn',
                   help=("Label of the connectivity priors, or full path\n"
                         "to the priors file, to use as source of connectivity\n"
                         "data to use in the creation of the regionwise priors\n"
                         "(available labels:\n" + "\n".join(h5Labels.keys())),
                   required=True, metavar="PRIORS_LABEL/PATH")
    p.add_argument('-po', '--priorsOut',
                   help=("File path to the new priors to create when using\n"
                         "the 'create' mode."),
                   default='')
    p.add_argument('-a', '--atlas',
                   help=("File path to the atlas (nifty file) containing all then\n"
                         "regions (or parcels) for the creation of the regionwise\n"
                         "priors. Must be a 3D volume, with each different region\n"
                         "defined by a non-zero integer."),
                   required=True)
    p.add_argument('-m', '--mode',
                   help=("Writting mode. Can be 'create', 'append', or 'replace'.\n"
                         "'create' will create a new h5 file with the regionwise\n"
                         "maps. It requires the 'priorsOut' option.\n"
                         "'append' will add the regionwise priors to the input file,\n"
                         "but only if there is no regionwise priors already there.\n"
                         "'replace' will overwrite the regionwise priors of the\n"
                         "input file and replace them with the new ones."),
                   choices=['create', 'append', 'replace'],
                   required=True
                   )
    p.add_argument('-p', '--proc',
                   help='Number of processes to run in parallel.',
                   default=1, type=int)
    return p


def getPriors(priorsLoc, h5Labels, priors_paths):
    if priorsLoc in h5Labels.keys():
        plabel = h5Labels[priorsLoc]
        ppath = priors_paths[plabel]
    else:
        ppath = priorsLoc
    return ppath


def prepAtlas(atlasF, priorsIn):
    atlasI = nib.load(atlasF)
    atlasName = os.path.basename(atlasF)
    atlasName = atlasName[: atlasName.find('.nii')]
    with h5py.File(priorsIn, "r") as h5fin:
        template_vol = h5fin["template"][:]
        templShape = template_vol.shape
        hdr = h5fin["tract_voxel"].attrs["header"]
        hdr3D = eval(hdr)
        header3D = nib.Nifti1Header()
        for key in header3D.keys():
            header3D[key] = hdr3D[key]
        affine3D = header3D.get_sform()

    atlasI = checkOrient_load(atlasI, templShape, affine3D, 0, True)
    atlasHdr = header2string(atlasI.header)
    atlasV = atlasI.get_fdata()
    atlasV = np.nan_to_num(np.round(atlasV)).astype(int)
    return atlasV, atlasName, atlasHdr


def mkRegionwise(priorsF, priorsFout, atlasV, atlasName, atlasHdr, writeType, proc=1):

    if not priorsFout:
        priorsFout = priorsF

    regList = list(np.unique(atlasV))
    regList.remove(0)

    with h5py.File(priorsFout, "a") as h5fout:
        if writeType == 'create':
            with h5py.File(priorsF, "r") as h5fin:
                h5fin.copy(h5fin['template'], h5fout['/'], 'template')
            h5fout.create_group('tract_voxel')
        elif writeType == 'append':
            grp_reg = h5fout['tract_region']
            grp_mreg = h5fout['mask_region']
            if len(grp_reg.keys()) > 1 or len(grp_mreg.keys()) > 1:
                raise FileExistsError('There already is data in the regionwise portion of '
                                      'the priors file. Use the "create" writting mode to '
                                      'create a new file, or the "replace" mode to overwrite '
                                      'existing data.')
            else:
                del h5fout['tract_region']
                del h5fout['mask_region']
        elif writeType == 'replace':
            del h5fout['tract_region']
            del h5fout['mask_region']
        grp_reg = h5fout.create_group('tract_region')
        grp_reg.attrs['header'] = atlasHdr
        grp_mreg = h5fout.create_group('mask_region')
        grp_mreg.attrs['header'] = atlasHdr

    for ireg, reg in enumerate(regList):
        print(f'Processing region labeled {reg} ({ireg+1}/{len(regList)})')
        regName = f'{atlasName}_{reg}'
        regVol = np.where(atlasV == reg, 1, 0).astype('int32')
        pmap = probaMap_fromROI(regVol, priorsF, atlasV.shape, proc=proc).astype('f')
        with h5py.File(priorsFout, "a") as h5fout:
            h5fout['tract_region'].create_dataset(regName, data=pmap, chunks=regVol.shape,
                                                  compression='gzip')
            h5fout['mask_region'].create_dataset(regName, data=regVol, chunks=regVol.shape,
                                                 compression='gzip')


def main():
    priorsOK, txtH5, h5Labels, priors_paths = checkH5()
    parser = _build_arg_parser(h5Labels)
    args = parser.parse_args()

    if args.mode == 'create' and args.priorsOut:
        raise parser.error('The ouput file already exists.')
    if args.mode in ['replace', 'append'] and args.priorsOut:
        raise parser.error("The 'replace' and 'append' options work directly on the "
                           "input priors file, and do not accept a 'priorsOut' output file path.")

    priorsIn = getPriors(args.priorsIn, h5Labels, priors_paths)
    if not os.path.isfile(priorsIn):
        raise parser.error(f"'priorsIn' file not found ({priorsIn})")

    atlasV, atlasName, atlasHdr = prepAtlas(args.atlas, priorsIn)

    mkRegionwise(priorsIn, args.priorsOut, atlasV, atlasName, atlasHdr, args.mode, args.proc)


if __name__ == '__main__':
    main()
