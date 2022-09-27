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

try:
    import Functionnectome.functionnectome as fun
    from Functionnectome.functionnectome import PRIORS_H5  # , PRIORS_URL, PRIORS_ZIP
except ModuleNotFoundError:
    print(
        "The Functionnectome module was not found (probably not installed via pip)."
        " Importing functions from the folder where the current script was saved..."
    )
    import functionnectome as fun
    from functionnectome import PRIORS_H5  # , PRIORS_URL, PRIORS_ZIP

# %%


class MyParser(argparse.ArgumentParser):  # Used to diplay the help if no argment is given
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def probaMap_fromROI(roiFile, priorsLoc, priors_type, outFile, maxVal=False, templateFile=None):
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
    maxVal : bool
        If true, will return the max value of the priors for each voxel. The output will thus be between 0 and 1.
        Otherwise, returns the sum of the priors.
    templateFile : str, optional
        DESCRIPTION. Path to the template file defining the shape and orientation. Only needed for 'nii' priors.

    Raises
    ------
    ValueError
        Bad orientation of the ROI (bad affine).

    Returns
    -------
    None.

    """

    roiIm = nib.load(roiFile)
    roiVol = roiIm.get_fdata().astype(bool)

    h5file = mapDir = priorsLoc

    if priors_type == "h5":
        # First get the shape and the affine of the template used in the priors
        with h5py.File(h5file, "r") as h5fout:
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

    # Check if the ROI is in the correct orientation
    if not (roiIm.affine == affine3D).all():
        flipLFaffine = roiIm.affine.copy()
        flipLFaffine[0] = -flipLFaffine[0]
        if (flipLFaffine == affine3D).all():  # The ROI need to be flipped
            roiVol = np.flip(roiVol, 0)
            print("Warning: the ROI was flipped Left/Right")
        else:
            raise ValueError(
                "The orientations of the ROI and of the priors are not compatible"
            )

    listVox = np.argwhere(roiVol)
    outMap = np.zeros(templShape)

    if priors_type == "h5":
        with h5py.File(h5file, "r") as h5fout:
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
                vox_pmap_img = nib.load(os.path.join(mapDir, mapf))
            except FileNotFoundError:
                mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii"
                vox_pmap_img = nib.load(os.path.join(mapDir, mapf))
            except KeyError:  # The voxel is not part of the current priors. May happen with split priors
                continue
            mapVol = vox_pmap_img.get_fdata()
            outMap += mapVol

    outIm = nib.Nifti1Image(outMap, affine3D)
    nib.save(outIm, outFile)
    return


def main():
    # First, checking the h5 priors paths in the json
    pkgPath = os.path.dirname(__file__)
    jsonPath = os.path.join(pkgPath, "priors_paths.json")
    if os.path.exists(jsonPath):
        with open(jsonPath, "r") as jsonP:
            priors_paths = json.load(jsonP)
        priors_paths = fun.updateOldJson(jsonPath, priors_paths)
        h5Priors = fnmatch.filter(priors_paths.keys(), 'V* - *')  # h5 labels should follow this pattern
        if len(h5Priors) and priors_paths[h5Priors[0]]:
            priorsOK = True
            h5Labels = {h5[:h5.find(' - ')]: h5 for h5 in h5Priors}
            txtH5 = (
                "Currently available (downloaded) priors are:\n" +
                "\n".join([f"{h5l.replace(' - ', ' (') + ')'}" for h5l in h5Labels.values()])
            )
            if len(h5Priors) < len(PRIORS_H5):
                txtH5 += "\n\n(More priors are available for download on the Functionnectome GUI)"
        else:
            priorsOK = False
    else:
        priorsOK = False

    if not priorsOK:
        txtH5 = "/!\\ No connectivity priors found. Please download them using the Functionnectome GUI."
        h5Labels = {}

    # Creating the parser (with indication of the available priors)
    parser = MyParser(
        description=(
            "Generate a quick disconnectome using a lesion mask as input. Instead of running a tractography "
            "for each diffusion volume of the normative dataset using the lesion as the seed, it uses the "
            "pre-computed connectivity maps from the Functionnectome priors to obtain the disconnectome map.\n"
            "As for the classical Disconnectome, the lesion should be in the MNI space.\n\n" +
            txtH5 +
            "\n\n"
            "/!\\ Be aware however that this method hasn't been published yet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--inFiles", help="Path to the input lesion files.",
                        required=True, nargs='+')
    parser.add_argument("-p", "--priors", help="Label of the connectivity priors to use for the analysis"
                        " (available labels: " + " ".join(h5Labels.keys()) + ").",
                        required=True, choices=list(h5Labels.keys()), metavar="PRIORS_LABEL")
    parser.add_argument("-o", "--outDir", help="Path to the directory (or folder) "
                        "where the disconnectomes will be stored. If none is given, save them in the same "
                        "folder as their corresponding input (with 'qdisco_' as prefix).")

    args = parser.parse_args()
    # Convert relative path to absolute path (might be necessary)
    args.inFiles = [os.path.abspath(fp) for fp in args.inFiles]

    priors_type = 'h5'
    maxVal = True

    lesionFiles = args.inFiles
    h5Label = args.priors
    outDir = args.outDir
    if h5Label:
        h5file = priors_paths[h5Labels[h5Label]]
    else:
        raise args.error('No priors label were given')

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
            probaMap_fromROI(lesF, h5file, priors_type, outF, maxVal)
        else:
            print(f'{outF} already exists. Skipping.')


# %%
if __name__ == "__main__":
    main()
