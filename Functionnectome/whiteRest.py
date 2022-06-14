#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WhiteRest module

Explore the white matter of RSNs from the WhiteRest Atlas, and determine
the potential impact of white matter lesions on RSNs.

The minimal inputs should be (in that order):
    - The ROI you wish to explore (as a nifti file in the MNI space)
    - The white matter maps of WhiteRest atlas (input as one 4D nifti file)
    - The RSN labels information from the atlas (input as a .txt file)

The atlas and label files can be downloaded from:
    https://www.dropbox.com/s/mo4zs159rqhqopv/WhiteRest.zip?dl=0
or, if there is a problem with the link, uppon request to:
    victor.nozais@gmail.com

WhiteRest will output a table with the Presence score for each RSN in the ROI,
both in % and raw score, as well as a few other metrics.
If no output file is given (with the "-o" option), the table will be printed in
the terminal. Otherwise, it will be saved as a text file, which can be imported
to a spreadsheet software (such as Excel) for further processing.

The software also gives the possibility to save a pie-chart summary of the results.

"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


pd.set_option("display.max_rows", None, "display.max_columns", None)

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
    p.add_argument('-o', '--out_table',
                   help='Path to save the results (.txt or .csv).')
    p.add_argument('-z', '--Z_thresh',
                   default=7,
                   help='Threshold to apply to the atlas z-maps (default z>7).')
    p.add_argument('-b', '--binarize',
                   action='store_true',
                   help='Binarize the maps after thresholding.')
    p.add_argument('-p', '--out_pie',
                   help='Path to save a pie-chart figure of the results (.png).')
    p.add_argument('-pt', '--thr_low_pie',
                   default=5,
                   help='Presence %% under which the RSNs are grouped on the pie-chart (default <5%%).')

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
    ROI_im = nib.load(ROI_f)
    ROI = ROI_im.get_fdata().astype(bool)
    ROI_affine = ROI_im.affine.copy()
    atlas_im = nib.load(atlas_f)
    atlas = atlas_im.get_fdata()
    RSNlabels = pd.read_csv(RSNlabels_f, sep='\t')

    if not (ROI_im.affine == atlas_im.affine).all():
        ROI_affine[0] = -ROI_affine[0]
        if (ROI_affine == atlas_im.affine).all():
            ROI = np.flip(ROI)
        else:
            print("Orientation (affine) of the input ROI volume not recognized. "
                  "Expected orientation matrix:")
            print(atlas_im.affine)
            print('Affine of the input ROI:')
            print(ROI_im.affine)
            raise ValueError('Wrong data orientation, or not in MNI152 space.')

    if len(RSNlabels) != atlas.shape[3]:
        raise IndexError('Number of RSNs in the atlas and the label file not the same')

    if RSNlabels.iloc[-1]['RSN name'] == 'Cerebellum':  # Removing the cerebellum RSN from the analysis
        RSNlabels = RSNlabels.drop(RSNlabels.index[-1])
        atlas = atlas[:, :, :, :-1]
    atlas = np.nan_to_num(atlas)
    totalPresRSN = [atlas[..., i].sum() for i in range(atlas.shape[-1])]
    resPresence = pd.DataFrame(columns=('RSN number', 'RSN name',
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
    presenceRSNnorm = 100 * presence / totalPresRSN
    presenceProp = 100 * np.divide(presenceRSNnorm,
                                   presenceRSNnorm.sum(),
                                   out=np.zeros_like(presenceRSNnorm),
                                   where=presenceRSNnorm.sum() != 0)
    coverage = 100 * binMaps.sum(0) / ROI.sum()

    indRSNpresent = np.argwhere(presence).T[0]
    for i in indRSNpresent:
        resPresence.loc[i] = [RSNlabels.loc[i, 'RSN number'],  # RSN number
                              RSNlabels.loc[i, 'RSN name'],  # RSN name
                              presenceProp[i],  # Presence (%)
                              presence[i],  # Presence (raw)
                              presenceRSNnorm[i],  # Presence (RSN)
                              coverage[i]]  # Coverage
    resPresence.sort_values('Presence (%)', ascending=False, inplace=True, ignore_index=True)
    return resPresence


def make_fun_autopct(res):
    def fun_autopct(pct):
        res_array = res[['Presence (%)', 'Coverage (%)']].values
        ind_pct = (np.abs(res_array[:, 0] - pct)).argmin()  # Ugly but should work (in most cases)
        pct_cov = res_array[ind_pct, 1]
        return "{:.1f}%\n({:.1f}%)".format(pct, pct_cov)
    return fun_autopct


def make_fun_autopct_withThr(res):
    def fun_autopct(pct):
        res_array = res[['Presence (%)', 'Coverage (%)']].values
        ind_pct = (np.abs(res_array[:, 0] - pct)).argmin()  # Ugly but should work (in most cases)
        if ind_pct == 0:
            return "{:.1f}%".format(pct)
        else:
            pct_cov = res_array[ind_pct, 1]
            return "{:.1f}%\n({:.1f}%)".format(pct, pct_cov)
    return fun_autopct


def plot_pie(res, outFile, thresh_percent):
    lowrsn = (res['Presence (%)'] < thresh_percent)
    lowrsnNb = lowrsn.sum()
    res_thr = res.copy()
    sumLow = 0
    if lowrsnNb > 1:
        for i in range(lowrsnNb):
            sumLow += res.loc[i, 'Presence (raw)']
            res_thr = res_thr.drop(i)
        perc_sumLow = 100 * sumLow / res['Presence (raw)'].sum()
        res_thr = pd.concat([pd.DataFrame({'RSN number': [f'< {thresh_percent}% ({lowrsnNb} RSN)'],
                                           'Presence (raw)': [sumLow],
                                           'Presence (%)': perc_sumLow}), res_thr],
                            ignore_index=True)
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, len(res_thr))]
        expl = [0 for i in range(len(res_thr))]  # To put the low value appart in the pie-chart
        expl[0] = 0.1
        plt.figure(figsize=(10, 8), dpi=120)
        patches, texts, autotexts = plt.pie(res_thr['Presence (raw)'],
                                            labels=res_thr['RSN number'],
                                            textprops={'fontsize': 12, 'font': 'Arial'},
                                            autopct=make_fun_autopct_withThr(res_thr),  # '%1.1f%%',
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
        plt.figure(figsize=(10, 8), dpi=120)
        patches, texts, autotexts = plt.pie(res['Presence (raw)'],
                                            labels=res['RSN number'],
                                            textprops={'fontsize': 12, 'font': 'Arial'},
                                            autopct=make_fun_autopct(res),
                                            shadow=False,
                                            colors=colors,
                                            pctdistance=0.8)
        for autotxt in autotexts:
            autotxt.set_fontsize(15)
        plt.savefig(outFile)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ROI_f = args.in_ROI
    atlas_f = args.atlas_maps
    RSNlabels_f = args.atlas_labels
    zThresh = float(args.Z_thresh)
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
        res.to_csv(args.out_table, sep='\t')
    else:
        print(res)

    if args.out_pie:
        res = res.sort_values('Presence (%)').reset_index()
        plot_pie(res, args.out_pie, float(args.thr_low_pie))


# %%
if __name__ == "__main__":
    main()
