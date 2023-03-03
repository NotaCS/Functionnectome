#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:57:49 2023

@author: Victor Nozais

Transform a functionnectome 4D volume into a dynamical functional connectivity
functionnectome 4D volume (comparable to TW-dFC)
"""

import nibabel as nib
import numpy as np
import h5py
from scipy.spatial.distance import cdist
import multiprocessing
import argparse


# %%

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Required argument
    p.add_argument('-i', '--in_functionnectome',
                   required=True,
                   help='Path of the input functionnectome 4D file (.nii, .nii.gz).')
    p.add_argument('-gm', '--gm_mask',
                   required=True,
                   help='Path of the grey matter mask used in the analysis (.nii, nii.gz).')
    p.add_argument('-wp', 'wm_priors',
                   required=True,
                   help='Path to the priors file (.h5) used in the analysis.')
    p.add_argument('-w', '--window_size',
                   type=int,
                   required=True,
                   help='Number of volumes (i.e. TR) per window for the sliding window.')
    p.add_argument('-o', '--output_file',
                   required=True,
                   help='Path of the output file to save (.nii, .nii.gz)')

    # Optional arguments
    p.add_argument('-p', '--process',
                   default=1,
                   type=int,
                   help="Number of (parallel) processes to launch")
    return p


def run_dFC_funtome(vox_list):
    len_dFC = funtome_vol.shape[-1] + 1 - windowSize  # Nb of time-points in the output
    ts_out = np.zeros((len(vox_list), len_dFC))
    with h5py.File(priorsLoc, "r") as h5fout:
        for ind_num, indvox in enumerate(vox_list):
            ts_funtome = funtome_vol[indvox]
            try:
                priorMap = h5fout["tract_voxel"][f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"][:]
            except KeyError:
                continue
            weights = priorMap[gm_mask]
            for step in range(len_dFC):
                ts_funtome_w = ts_funtome[step: step + windowSize]
                ts_gm_w = funtome_vol[gm_mask, step: step + windowSize]
                fc_w = 1 - cdist(ts_funtome_w.reshape(1, -1), ts_gm_w, metric='correlation')
                z_fc_w = np.arctanh(fc_w)  # Fisher transform
                fc_average = (weights * z_fc_w).sum() / len(weights)
                ts_out[ind_num, step] = fc_average
    return ts_out


def init_worker(gm, bold, bold_shape, winSize, pLoc):
    global funtome_vol, gm_mask, windowSize, priorsLoc
    funtome_vol = np.frombuffer(bold, "f").reshape(bold_shape)
    gm_mask = gm
    windowSize = winSize
    priorsLoc = pLoc


def main():
    global gm_mask, funtome_vol, windowSize, priorsLoc

    parser = _build_arg_parser()
    args = parser.parse_args()

    funtome_f = args.in_functionnectome
    gm_f = args.gm_mask
    priorsLoc = args.wm_priors
    windowSize = args.window_size
    out_f = args.output_file
    proc = args.process

    gm_mask = nib.load(gm_f).get_fdata().astype(bool)
    funtome_im = nib.load(funtome_f)
    print('Loading functionnectome file')
    funtome_vol = funtome_im.get_fdata(caching='unchanged', dtype='f')
    funtome_shape = funtome_vol.shape

    if windowSize > funtome_im.shape[-1]:
        raise ValueError('The sliding window is bigger that the total volume.')
    if windowSize < 1:
        raise ValueError('The sliding window size is lower than 1.')

    dFC_funtome = np.zeros(funtome_shape)
    with h5py.File(priorsLoc, "r") as h5fout:
        template_vol = h5fout["template"][:].astype(bool)
    listVox = np.argwhere(template_vol)

    print('Starting process')
    if proc == 1:
        listInd = tuple(listVox.T)  # like np.where
        listVox = [tuple(ind) for ind in listVox]  # tuple-ing
        ts_dFC = run_dFC_funtome(listVox)
        dFC_funtome = np.zeros(funtome_shape)
        # dFC_funtome[np.where(template_vol)] = ts_dFC
        dFC_funtome[listInd] = ts_dFC
    elif proc > 1:
        vox_batches = np.array_split(listVox, proc)
        vox_batches = [[tuple(ind) for ind in ind_arr] for ind_arr in vox_batches]  # tuple-ing
        # Creating shared memory variables accessed by the parrallel processes:
        bold_shared = multiprocessing.RawArray("f", int(np.prod(funtome_shape)))
        # Manipulate the RawArray as a numpy array
        bold_shared_np = np.frombuffer(bold_shared, "f").reshape(funtome_shape)
        np.copyto(bold_shared_np, funtome_vol)  # Filling the RawArray
        with multiprocessing.Pool(
            processes=proc,
            initializer=init_worker,
            initargs=(gm_mask, bold_shared, funtome_shape, windowSize, priorsLoc),
        ) as pool:
            poolRes = pool.map_async(
                run_dFC_funtome, vox_batches
            )
            res_batches = poolRes.get()
            for ind_tupled_batch, batch_res in zip(vox_batches, res_batches):
                ind_arg_batch = np.array(ind_tupled_batch)  # like np.argwhere
                ind_vox_batch = tuple(ind_arg_batch.T)  # line np.where
                dFC_funtome[ind_vox_batch] = batch_res
    else:
        raise ValueError("Number of parallel processing should be 1 or higher.")

    out_im = nib.Nifti1Image(dFC_funtome, funtome_im.affine)
    out_im.header['pixdim'] = funtome_im.header['pixdim']
    nib.save(out_im, out_f)


if __name__ == "__main__":
    main()
