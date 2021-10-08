#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Explore the white matter of RSNs, and determine the potential impact of white matter lesions to RSNs

Options:
    - Print or no print pictures:
        - pie chart
        - Brain cross-section  # TODO maybe
        - RSN 3D picture  # TODO maybe
    - Witout z-score threshold:
        - Output presence raw value / proportion
        - What threshold for the outpout
    - With z-score threshold
        - Output presence raw value / prortion
        - Output binarised share
        - Output proprotion of the total RSN map?
    - Save output in txt file

TODO Add option to input multiple ROI (4D vol)


    ROI_f = ('/beegfs_data/scratch/nozais-functionnectome/colab_marc/analyse_300_sujets/MICCA_HCP/'
              'apply_funtome/ROI_centrum_semiovale_atlasBCBLAB_thr0p5_2mm.nii.gz')
    atlas_f = ('/beegfs_data/scratch/nozais-functionnectome/colab_marc/analyse_300_sujets/MICCA_HCP/'
                'apply_funtome/group_zmaps_best_masked.nii.gz')
    RSNlabels_f = ('/beegfs_data/scratch/nozais-functionnectome/colab_marc/analyse_300_sujets/MICCA_HCP/'
                    'labels_31RSN.csv')
    zThresh = 0

"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# %%


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Positional argument (obligatory)
    p.add_argument('in_ROI',
                   help='Path of the ROI file (.nii, .nii.gz).')
    p.add_argument('atlas_maps',
                   help='Path of the RSN atlas (.nii, nii.gz).')
    p.add_argument('atlas_labels',
                   help='Path to the atlas labels identifying the RSNs.')

    # Optional arguments
    p.add_argument('--out_table',
                   help='Path to save the results (.csv).')
    p.add_argument('--Z_thresh',
                   default=7,
                   help='Threshold to apply to the atlas z-maps (default z>7).')
    p.add_argument('--binarize',
                   action='store_true',
                   help='Binarize the maps after thresholding.')
    p.add_argument('--out_pie',
                   help='Path to save the a pie-chart figure of the results (.png).')
    p.add_argument('--thr_low_pie',
                   default=3,
                   help='Presence % under which the RSNs are grouped on the pie-chart.')

    return p


def checkOutFile(parser, path):
    path = os.path.abspath(path)
    pathDir = os.path.dirname(path)
    nameFile = os.path.basename(path)
    if not os.path.isdir(pathDir):
        parser.error(
            f"The directory of the output file '{nameFile}' does not exist."
            f"Please change the output directory or create it ({pathDir})."
            )
    if os.path.isfile(path):
        parser.error(
            f"The output file '{path}' already exists. Change the name or delete it."
            )


def checkInFile(parser, path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        parser.error(
            f"The input file '{path}' does not exists. Check and correct the path, then retry."
            )


def computPresence(ROI_f, atlas_f, RSNlabels_f, zThresh=7, binarize=False):
    '''
    Compute the presence score

    Parameters
    ----------
    ROI_f : str
        Filepath of the input ROI file (3D nifti in MNI 2mm space).
    atlas_f : str
        Filepath of the RSN white matter atlas (4D nifti in MNI 2mm space).
    RSNlabels_f : str
        Filepath of the text file with the labels and names of each RSN.
    options : dict
        Dictionnary containing the values of all the options.
            zThresh (float)
            rsnThresh (if zThresh=0) in %
            bin (if zThresh>0)
            propRSN (if zThresh>0)
            figs

    Returns
    -------
    A Pandas Dataframe with the presence score of each involved RSN

    '''
    ROI_im = nib.load(ROI_f)  # TODO Add a resampling to MNI if necessary
    ROI = ROI_im.get_fdata().astype(bool)
    atlas_im = nib.load(atlas_f)
    atlas = atlas_im.get_fdata()
    RSNlabels = pd.read_csv(RSNlabels_f)

    if len(RSNlabels) != atlas.shape[3]:
        raise IndexError('Number of RSNs in the atlas and the label file not the same')

    if RSNlabels.iloc[-1]['RSN name'] == 'Cerebellum':  # Removing the cerebellum RSN from the analysis
        RSNlabels = RSNlabels.drop(RSNlabels.index[-1])
        atlas = atlas[:, :, :, :-1]
    atlas = np.nan_to_num(atlas)
    totalPresRSN = [atlas[..., i].sum() for i in range(atlas.shape[-1])]
    resPresence = pd.DataFrame(columns=('RSN label', 'RSN name',
                                        'Presence (%)', 'Presence (raw)', 'Presence/RSN (%)',
                                        'Coverage (%)'))

    ROIinAtlas = atlas[ROI]  # 2D array ROIxRSN
    ROIinAtlas[ROIinAtlas < zThresh] = 0
    if binarize:  # Sould only be used if zThresh > 0, but I don't enforce it
        ROIinAtlas[ROIinAtlas > 0] = 1
        binMaps = ROIinAtlas
    else:
        binMaps = ROIinAtlas.copy()
        binMaps[binMaps > 0] = 1

    presence = ROIinAtlas.sum(0)
    presenceProp = 100*presence/presence.sum()
    presenceRSNprop = 100*presence/totalPresRSN
    coverage = 100*binMaps.sum(0)/ROI.sum()

    indRSNpresent = np.argwhere(presence).T[0]
    for i in indRSNpresent:
        resPresence.loc[i] = [RSNlabels.loc[i, 'RSN label'],  # RSN label
                              RSNlabels.loc[i, 'RSN name'],  # RSN name
                              presenceProp[i],  # Presence (%)
                              presence[i],  # Presence (raw)
                              presenceRSNprop[i],  # Presence (RSN)
                              coverage[i]]  # Coverage
    return resPresence


def plot_pie(res, outFile, thresh_percent):
    res = res.sort_values('Presence (%)').reset_index()
    lowrsn = (res['Presence (%)'] < thresh_percent)
    lowrsnNb = lowrsn.sum()
    res_thr = res.copy()
    sumLow = 0
    if lowrsnNb > 1:
        for i in range(lowrsnNb):
            sumLow += res.loc[i, 'Presence (raw)']
            res_thr = res_thr.drop(i)
        res_thr = pd.concat([pd.DataFrame({'RSN label': [f'< 5% ({lowrsnNb})'], 'Presence (raw)': [sumLow]}), res_thr],
                            ignore_index=True)
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, len(res_thr))]
        expl = [0 for i in range(len(res_thr))]  # To put the low value appart in the pie-chart
        expl[0] = 0.1
        plt.figure(figsize=(8, 8), dpi=80)
        patches, texts, autotexts = plt.pie(res_thr['Presence (raw)'],
                                            labels=res_thr['RSN label'],
                                            autopct='%1.1f%%',
                                            shadow=False,
                                            colors=colors,
                                            explode=expl,
                                            pctdistance=0.8)
        for autotxt in autotexts:
            autotxt.set_fontsize(15)
        plt.savefig(outFile)
    else:
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, len(res))]
        plt.figure(figsize=(8, 8), dpi=80)
        patches, texts, autotexts = plt.pie(res['Presence (raw)'], labels=res['RSN label'],
                                            autopct='%1.1f%%', shadow=False, colors=colors, pctdistance=0.8)
        for autotxt in autotexts:
            autotxt.set_fontsize(15)
        plt.savefig(outFile)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ROI_f = args.in_ROI
    atlas_f = args.atlas_maps
    RSNlabels_f = args.atlas_labels
    zThresh = args.Z_thresh
    binarize = args.binarize

    checkInFile(parser, ROI_f)
    checkInFile(parser, atlas_f)
    checkInFile(parser, RSNlabels_f)

    if args.out_table:
        checkOutFile(parser, args.out_table)
    if args.out_pie:
        checkOutFile(parser, args.out_pie)

    res = computPresence(ROI_f, atlas_f, RSNlabels_f, zThresh, binarize)

    if args.out_table:
        res.to_csv(args.out_table)
    else:
        print(res)

    if args.out_pie:
        plot_pie(res, args.out_pie, args.thr_low_pie)


if __name__ == "__main__":
    main()
