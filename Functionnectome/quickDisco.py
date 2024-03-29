#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:13:13 2022

@author: nozais

Generates a disconnectome based on a lesion files (3D nifti files) and a set of Functionnectome priors
"""


import nibabel as nib
import h5py
import numpy as np
import os
import sys
import json
import argparse
import fnmatch
import multiprocessing

import Functionnectome.functionnectome as fun
from Functionnectome.functionnectome import PRIORS_H5

# %%


class MyParser(argparse.ArgumentParser):  # Used to diplay the help if no argment is given
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def init_worker(ptype, ploc, tshape, mval):
    global priors_type, priorsLoc, templShape, maxVal
    priors_type = ptype
    priorsLoc = ploc
    templShape = tshape
    maxVal = mval


def probaMapMulti(batchVox):
    outMap = np.zeros(templShape)
    if priors_type == "h5":
        with h5py.File(priorsLoc, "r") as h5fout:
            for indvox in batchVox:
                try:
                    priorMap = h5fout["tract_voxel"][f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"][:]
                except KeyError:  # The voxel is not part of the current priors. May happen with split priors
                    continue
                if maxVal:
                    outMap = np.max(np.stack((outMap, priorMap)), 0)
                else:
                    outMap += priorMap
    else:
        for indvox in batchVox:
            try:
                mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz"
                vox_pmap_img = nib.load(os.path.join(priorsLoc, mapf))
            except FileNotFoundError:
                mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii"
                vox_pmap_img = nib.load(os.path.join(priorsLoc, mapf))
            except KeyError:  # The voxel is not part of the current priors. May happen with split priors
                continue
            priorMap = vox_pmap_img.get_fdata()
            if maxVal:
                outMap = np.max(np.stack((outMap, priorMap)), 0)
            else:
                outMap += priorMap
    return outMap


def probaMap_fromROI(roiVol, priorsLoc, templShape, priors_type='h5', proc=1, maxVal=True):
    """
    See probaMap_fromMask description for help on the arguments
    """

    listVox = np.argwhere(roiVol)
    outMap = np.zeros(templShape)
    if not fun.testPriorsH5(priorsLoc, 'voxel'):
        raise Exception('The selected priors do not have voxelwise data.')

    if proc == 1:
        if priors_type == "h5":
            with h5py.File(priorsLoc, "r") as h5fout:
                for indvox in listVox:
                    try:
                        priorMap = h5fout["tract_voxel"][f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"][:]
                    except KeyError:  # The voxel is not part of the current priors. May happen with split priors
                        continue
                    if maxVal:
                        outMap = np.max(np.stack((outMap, priorMap)), 0)
                    else:
                        outMap += priorMap
        else:
            for indvox in listVox:
                try:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz"
                    vox_pmap_img = nib.load(os.path.join(priorsLoc, mapf))
                except FileNotFoundError:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii"
                    vox_pmap_img = nib.load(os.path.join(priorsLoc, mapf))
                except KeyError:  # The voxel is not part of the current priors. May happen with split priors
                    continue
                priorMap = vox_pmap_img.get_fdata()
                if maxVal:
                    outMap = np.max(np.stack((outMap, priorMap)), 0)
                else:
                    outMap += priorMap
    elif proc > 1:
        batchVox = np.array_split(listVox, proc)
        with multiprocessing.Pool(
            processes=proc,
            initializer=init_worker,
            initargs=(priors_type, priorsLoc, templShape, maxVal),
        ) as pool:
            out_batch_disco = pool.map_async(probaMapMulti, batchVox)
            out_batch_disco = out_batch_disco.get()
        if maxVal:
            outMap = np.max(np.stack(out_batch_disco), 0)
        else:
            outMap = np.sum(np.stack(out_batch_disco), 0)
    return outMap


def probaMap_fromMask(roiFile, priorsLoc, priors_type,
                      outFile=None, proc=1, maxVal=False, templateFile=None):
    """
    Read the ROI and the associated maps, then create and save the new map

    Parameters
    ----------
    roiFile : str
        Full path to the ROI nifti file.
    priorsLoc : str
        Path to either the HDF5 file or the folder containing all the nifti maps.
    priors_type : str
        Type of priors input, either 'h5' or 'nii'.
    outFile : str
        Full file path for the new map to be created.
    proc : int
        Number of processes used. If > 1, will run parallel processing
    maxVal : bool
        If true, will return the max value of the priors for each voxel. The output will thus be between 0 and 1.
        Otherwise, returns the sum of the priors.
    templateFile : str, optional
        Path to the template file defining the shape and orientation. Only needed for 'nii' priors.
    Raises
    ------
    ValueError
        Bad orientation of the ROI (bad affine).

    Returns
    -------
    None.

    """

    roiIm = nib.load(roiFile)

    if priors_type == "h5":
        # First get the shape and the affine of the template used in the priors
        with h5py.File(priorsLoc, "r") as h5fout:
            template_vol = h5fout["template"][:]
            templShape = template_vol.shape
            hdr = h5fout["tract_voxel"].attrs["header"]
            hdr3D = eval(hdr)
            header3D = nib.Nifti1Header()
            for key in header3D.keys():
                header3D[key] = hdr3D[key]
            affine3D = header3D.get_sform()
    else:
        templIm = nib.load(templateFile)
        templShape = templIm.shape
        affine3D = templIm.affine

    roiIm = fun.checkOrient_load(roiIm, templShape, affine3D, 0, True)
    roiVol = roiIm.get_fdata().astype(bool)

    outMap = probaMap_fromROI(roiVol, priorsLoc, templShape, priors_type, proc, maxVal)

    outIm = nib.Nifti1Image(outMap, affine3D)
    if outFile:
        nib.save(outIm, outFile)
        return
    else:
        return outIm


def checkH5():
    """
    Check and list the available Functionnectome priors to be used for quickDisco
    """
    pkgPath = os.path.dirname(__file__)
    jsonPath = os.path.join(pkgPath, "priors_paths.json")
    if os.path.exists(jsonPath):
        with open(jsonPath, "r") as jsonP:
            priors_paths = json.load(jsonP)
        priors_paths = fun.updateOldJson(jsonPath, priors_paths)
        fun.check_old_V2_priors(priors_paths)
        h5Priors = fnmatch.filter(priors_paths.keys(), 'V* - *')  # h5 labels should follow this pattern
        if len(h5Priors) and priors_paths[h5Priors[0]]:
            priorsOK = True
            h5Labels = {h5[:h5.find(' - ')]: h5 for h5 in h5Priors}
            txtH5 = (
                "Currently available (downloaded) priors are:\n"
                + "\n".join([f"{h5l.replace(' - ', ' (') + ')'}" for h5l in h5Labels.values()])
            )
            if len(h5Priors) < len(PRIORS_H5):
                txtH5 += "\n\n(More priors are available for download on the Functionnectome GUI)"
        else:
            priorsOK = False
    else:
        priorsOK = False
        txtH5 = ("/!\\ No connectivity priors found. Please download them using the Functionnectome"
                 " GUI or give the full path to a .h5 priors file.")
        h5Labels = {}
        priors_paths = {}
    return priorsOK, txtH5, h5Labels, priors_paths


def main():
    # First, checking the h5 priors paths in the json
    priorsOK, txtH5, h5Labels, priors_paths = checkH5()

    # Creating the parser (with indication of the available priors)
    parser = MyParser(
        description=(
            "Generate a quick disconnectome using a lesion mask as input. Instead of running a tractography "
            "for each diffusion volume of the normative dataset using the lesion as the seed, it uses the "
            "pre-computed connectivity maps from the Functionnectome priors to obtain the disconnectome map.\n"
            "As for the classical Disconnectome, the lesion should be in the MNI space.\n\n"
            + txtH5
            + "\n\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--inFiles", help="Path to the input lesion files.",
                        required=True, nargs='+')
    parser.add_argument("-p", "--priors", help=("Label of the connectivity priors, "
                                                "or full path to the priors file, to use for the analysis\n"
                                                "Available labels:\n" + "\n".join(h5Labels.keys())),
                        required=True, metavar="PRIORS_LABEL/PATH")
    parser.add_argument("-o", "--outDir", help="Path to the directory (or folder) "
                        "where the disconnectomes will be stored. If none is given, save them in the same "
                        "folder as their corresponding input (with 'qdisco_' as prefix).")
    parser.add_argument('-pp', '--parallel_proc',
                        default=1,
                        help='Number of processes to launch, > 1 will parallelize the task.')

    args = parser.parse_args()
    # Convert relative path to absolute path (might be necessary)
    args.inFiles = [os.path.abspath(fp) for fp in args.inFiles]

    priors_type = 'h5'
    maxVal = True

    lesionFiles = args.inFiles
    h5Loc = args.priors
    outDir = args.outDir
    pproc = int(args.parallel_proc)
    if pproc < 1:
        raise ValueError('The number of processes should be equal or more than 1')
    if h5Loc in list(h5Labels.keys()):
        priorsLoc = priors_paths[h5Labels[h5Loc]]
    elif os.path.splitext(h5Loc)[-1] == ".h5" and os.path.exists(h5Loc):
        priorsLoc = h5Loc
    else:
        raise parser.error('No correct priors label or path were given')

    for lnum, lesF in enumerate(lesionFiles):
        print(f'{lnum + 1}/{len(lesionFiles)}')
        inDir = os.path.dirname(lesF)
        lesName = os.path.basename(lesF)
        if args.outDir:
            outDir = args.outDir
        else:
            outDir = inDir

        if os.path.isdir(outDir):
            outF = os.path.join(outDir, 'qdisco_' + lesName)
        else:
            raise FileNotFoundError(f'The output ({outDir}) directory does not exists.')

        if not os.path.exists(outF):
            probaMap_fromMask(lesF, priorsLoc, priors_type, outF, pproc, maxVal=maxVal)
        else:
            print(f'{outF} already exists. Skipping.')


# %%
if __name__ == "__main__":
    main()
