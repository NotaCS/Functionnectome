#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:57:49 2023

@author: Victor Nozais

Transform a functionnectome 4D volume into a dynamical functional connectivity
functionnectome 4D volume (comparable to TW-dFC).

A bit faster now...

"""

import nibabel as nib
import numpy as np
import h5py
import multiprocessing
import argparse
import time
import os
from Functionnectome.functionnectome import checkOrient_load
from Functionnectome.quickDisco import checkH5
# from scipy.spatial.distance import cdist
# from numba import njit
# import numexpr as ne


# %%

def _build_arg_parser(h5L):
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Required argument
    p.add_argument('-i', '--in_functionnectome',
                   required=True,
                   help='Path of the input functionnectome 4D file (.nii, .nii.gz).')
    p.add_argument('-gm', '--gm_mask',
                   required=True,
                   help='Path of the grey matter mask used in the analysis (.nii, nii.gz).')
    p.add_argument('-wp', '--wm_priors',
                   required=True,
                   help=(
                       'Path to the priors file (.h5) used in the analysis, or label to the '
                       'corresponding priors (if set-up for the Functionnectome).\n'
                       '(available labels: ' + ' '.join(h5L.keys()) + ").")
                   )
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
    p.add_argument('-wm', '--wm_mask',
                   help='Path of a white matter mask (output voxels) to speed-up computation (.nii, nii.gz).')
    p.add_argument('-v', '--verbose',
                   action='store_true',
                   help='Print the advancement evey 100 voxel processed')
    return p


def correl_all(A, B):
    '''
    Correlate each time-window beetween A and B. Much faster (X5) than using cdist.
    But may be more RAM hungry.

    A = (window, time)
    B = (voxel, window, time)

    C = (voxel, window)

    '''
    A = np.expand_dims(A, 0)
    Aresiduals = A - A.mean(-1, keepdims=True)
    Bresiduals = B - B.mean(-1, keepdims=True)

    Aresidual_sqsums = (Aresiduals**2).sum(-1, keepdims=True)
    Bresidual_sqsums = (Bresiduals**2).sum(-1, keepdims=True)

    covs = (Aresiduals * Bresiduals).sum(-1)

    norm_prods = np.sqrt((Aresidual_sqsums * Bresidual_sqsums).sum(-1))

    C = covs / norm_prods
    return C


# @njit
# def correlJIT(a, b):
#     '''
#     Correlate each time-window beetween A and B
#     Should be faster than the non-numexpr function, but doesn't appear to be...
#     a = (wind, time)
#     b = (wind, voxel, time)

#     c.T = (voxel, wind)

#     '''
#     c = np.zeros((b.shape[0:2]), dtype=np.float32)
#     for w in range(len(a)):
#         B = b[w]
#         A = a[w]

#         Bmean = np.array([B[i, :].mean() for i in range(B.shape[1])])
#         Amean = np.mean(A)
#         A = A.reshape(1, -1)

#         Aresiduals = A - Amean
#         Bresiduals = B - Bmean

#         Aresidual_sqsums = np.expand_dims(np.sum(Aresiduals**2, 1), 1)
#         Bresidual_sqsums = np.expand_dims(np.sum(Bresiduals**2, 1), 1)

#         covs = np.dot(Aresiduals, Bresiduals.T)

#         norm_prods = np.sqrt(np.dot(Aresidual_sqsums, Bresidual_sqsums.T))

#         C = (covs / norm_prods).reshape(-1)
#         c[w] = C
#     return c.T


# def correl_all_numexpr(A, B):
#     A = np.expand_dims(A, 0)
#     Amean = A.mean(-1, keepdims=True)
#     Bmean = B.mean(-1, keepdims=True)
#     Aresiduals = ne.evaluate('A - Amean')
#     Bresiduals = ne.evaluate('B - Bmean')

#     Aresidual_sqsums = np.expand_dims(ne.evaluate('sum(Aresiduals**2, 2)'), -1)
#     Bresidual_sqsums = np.expand_dims(ne.evaluate('sum(Bresiduals**2, 2)'), -1)

#     covs = ne.evaluate('sum(Aresiduals * Bresiduals, 2)')

#     norm_prods = ne.evaluate('sum(Aresidual_sqsums * Bresidual_sqsums, 2)')

#     C = ne.evaluate('covs / sqrt(norm_prods)')
#     return C


def correl(AB):
    '''
    Correlate each time-window beetween A and B
    Should be faster than the non-numexpr function, but doesn't appear to be...
    A = (time)
    B = (voxel, time)

    C = (voxel,)

    I tried using numexpr, numba, cdist, but for some reason
    this is still the fastest method in cobination with map()...
    '''
    A, B = AB
    A = A.reshape(1, -1)
    Aresiduals = A - A.mean(-1, keepdims=True)
    Bresiduals = B - B.mean(-1, keepdims=True)

    Aresidual_sqsums = (Aresiduals**2).sum(-1, keepdims=True)
    Bresidual_sqsums = (Bresiduals**2).sum(-1, keepdims=True)

    covs = np.dot(Aresiduals, Bresiduals.T)

    norm_prods = np.sqrt(np.dot(Aresidual_sqsums, Bresidual_sqsums.T))

    C = np.squeeze(covs / norm_prods)
    return C


def run_dFC_funtome_all(vox_list):
    ''' Vectorise the sliding window operation and the correlation computation '''
    len_dFC = funtome_vol.shape[-1] + 1 - windowSize  # Nb of time-points in the output
    ts_out = np.zeros((len(vox_list), len_dFC))
    # Create a matrice of indexes with the time indexes for each time window per line
    mat_windows = np.array(list(range(windowSize)) * len_dFC).reshape(len_dFC, windowSize)
    mat_windows += np.arange(len_dFC).reshape((len_dFC, 1))  # incrementing 1 index per line
    gm_ts = funtome_vol[gm_mask]  # Time-series of all gm voxels
    mat_win_gm_ts = gm_ts[:, mat_windows]
    mat_win_gm_ts = np.swapaxes(mat_win_gm_ts, 0, 1)  # test
    tlast = time.time()
    with h5py.File(priorsLoc, "r") as h5fout:
        for ind_num, indvox in enumerate(vox_list):
            if verb and not ind_num % 100:
                tnow = time.time()
                tdiff = int(tnow - tlast)
                print(f'voxel {ind_num} of {len(vox_list)} ({tdiff} sec since previous print)')
                tlast = tnow
            ts_funtome = funtome_vol[indvox]
            try:
                priorMap = h5fout["tract_voxel"][f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"][:]
            except KeyError:
                continue
            mat_win_ts = ts_funtome[mat_windows]  # ts of all the time windows in 1 matrix
            # mat_fc = correl_all(mat_win_ts, mat_win_gm_ts)
            winzip = zip(mat_win_ts, mat_win_gm_ts)  # test
            mat_fc = np.array(list(map(correl, winzip))).T  # test
            # mat_fc = correlJIT(mat_win_ts, mat_win_gm_ts)  # test2
            weights = priorMap[gm_mask].reshape(-1, 1)
            ts_out[ind_num] = (mat_fc * weights).sum(0) / len(weights)
    return ts_out


# def run_dFC_funtome(vox_list):
#     len_dFC = funtome_vol.shape[-1] + 1 - windowSize  # Nb of time-points in the output
#     ts_out = np.zeros((len(vox_list), len_dFC))
#     tlast = time.time()
#     with h5py.File(priorsLoc, "r") as h5fout:
#         for ind_num, indvox in enumerate(vox_list):
#             if verb and not ind_num % 100:
#                 tnow = time.time()
#                 tdiff = int(tnow - tlast)
#                 print(f'voxel {ind_num} of {len(vox_list)} ({tdiff} sec since previous print)')
#                 tlast = tnow
#             ts_funtome = funtome_vol[indvox]
#             try:
#                 priorMap = h5fout["tract_voxel"][f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"][:]
#             except KeyError:
#                 continue
#             weights = priorMap[gm_mask]
#             for step in range(len_dFC):
#                 ts_funtome_w = ts_funtome[step: step + windowSize]
#                 ts_gm_w = funtome_vol[gm_mask, step: step + windowSize]
#                 fc_w = 1 - cdist(ts_funtome_w.reshape(1, -1), ts_gm_w, metric='correlation')
#                 # z_fc_w = np.arctanh(fc_w)  # Fisher transform -> Bad results! :(
#                 # z_fc_average = (weights * z_fc_w).sum() / len(weights)
#                 # fc_average = np.tanh(z_fc_average)  # back to "correlation" values
#                 fc_average = (weights * fc_w).sum() / len(weights)
#                 ts_out[ind_num, step] = fc_average
#     return ts_out


def init_worker(gm, bold, bold_shape, winSize, pLoc, v):
    global funtome_vol, gm_mask, windowSize, priorsLoc, verb
    funtome_vol = np.frombuffer(bold, "f").reshape(bold_shape)
    gm_mask = gm
    windowSize = winSize
    priorsLoc = pLoc
    verb = v


def main():
    global gm_mask, funtome_vol, windowSize, priorsLoc, verb

    st = time.time()

    _, _, h5Labels, priors_paths = checkH5()
    parser = _build_arg_parser(h5Labels)
    # inArgs = ['-i',
    #           '/Users/victor/Dropbox (GIN)/Thèse/Collab_Cacciola/files/functionnectome100.nii.gz',
    #           '-gm',
    #           '/Users/victor/Dropbox (GIN)/Thèse/Collab_Cacciola/files/left_functionnectome_mask.nii.gz',
    #           '-wp',
    #           '/Users/victor/Dropbox (GIN)/Thèse/Collab_Cacciola/files/priors_arcuate_L.h5',
    #           '-w',
    #           '10',
    #           '-o',
    #           '/Users/victor/Desktop/test funtome/dFC_small_funtome_arcuate.nii.gz',]
    # inArgs += ['-p', '6']
    # args = parser.parse_args(inArgs)
    args = parser.parse_args()

    funtome_f = args.in_functionnectome
    gm_f = args.gm_mask
    h5Loc = args.wm_priors
    if h5Loc in list(h5Labels.keys()):
        priorsLoc = priors_paths[h5Labels[h5Loc]]
    elif os.path.splitext(h5Loc)[-1] == ".h5" and os.path.exists(h5Loc):
        priorsLoc = h5Loc
    else:
        raise parser.error('No correct priors label or path were given')
    windowSize = args.window_size
    out_f = args.output_file
    proc = args.process
    verb = args.verbose
    wm_f = args.wm_mask

    if os.path.exists(out_f):
        raise FileExistsError('Output file already exists.')

    # Get the template info
    with h5py.File(h5Loc, "r") as h5fout:
        template_vol = h5fout["template"][:].astype(bool)
        templShape = template_vol.shape
        hdr = h5fout["tract_voxel"].attrs["header"]
        hdr3D = eval(hdr)
        affine3D = np.stack([hdr3D['srow_x'],
                             hdr3D['srow_y'],
                             hdr3D['srow_z'],
                             [0, 0, 0, 1]])

    gm_mask_im = nib.load(gm_f)
    gm_mask_im = checkOrient_load(gm_mask_im, templShape, affine3D, 0)
    gm_mask = gm_mask_im.get_fdata().astype(bool)

    wm_mask = None
    if wm_f is not None:
        if not os.path.exists(wm_f):
            raise FileNotFoundError(f'The white matter mask file was not found ("{wm_f}" does not exist).')
        else:
            wm_mask_im = nib.load(wm_f)
            wm_mask_im = checkOrient_load(wm_mask_im, templShape, affine3D, 0)
            wm_mask = wm_mask_im.get_fdata().astype(bool)

    funtome_im = nib.load(funtome_f)
    if verb:
        print('Loading functionnectome file')
    funtome_shape = templShape + (funtome_im.shape[-1],)
    funtome_im = checkOrient_load(funtome_im, funtome_shape, affine3D)
    funtome_vol = funtome_im.get_fdata(caching='unchanged', dtype='f')
    funtome_signal = np.invert(np.all(np.equal(funtome_vol[..., 0, None], funtome_vol), -1))  # Find non-constant voxels
    # funtome_signal = funtome_vol.std(-1).astype(bool)  # Eqyuivalent to the above line, but slower
    gm_mask = gm_mask * funtome_signal  # !!! Can change absolute values in the weighted average between subjects
    if not gm_mask.sum():  # No remaining voxels in the mask
        raise ValueError('No functional signal ')
    len_dFC = funtome_shape[-1] + 1 - windowSize  # Nb of time-points in the output
    dFC_funtome = np.zeros(funtome_shape[:-1] + (len_dFC,))

    if windowSize > funtome_shape[-1]:
        raise ValueError('The sliding window is bigger that the total volume.')
    if windowSize < 1:
        raise ValueError('The sliding window size is lower than 1.')

    if wm_mask is None:
        fun_mask = template_vol * funtome_signal
    else:
        fun_mask = wm_mask * funtome_signal
    listVox = np.argwhere(fun_mask)

    if verb:
        print('Starting process')
    if proc == 1:
        listInd = tuple(listVox.T)  # like np.where
        listVox = [tuple(ind) for ind in listVox]  # tuple-ing
        ts_dFC = run_dFC_funtome_all(listVox)
        # dFC_funtome[np.where(fun_mask)] = ts_dFC
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
            initargs=(gm_mask, bold_shared, funtome_shape, windowSize, priorsLoc, verb),
        ) as pool:
            poolRes = pool.map_async(
                run_dFC_funtome_all, vox_batches
            )
            res_batches = poolRes.get()
            for ind_tupled_batch, batch_res in zip(vox_batches, res_batches):
                ind_arg_batch = np.array(ind_tupled_batch)  # like np.argwhere
                ind_vox_batch = tuple(ind_arg_batch.T)  # like np.where
                dFC_funtome[ind_vox_batch] = batch_res
    else:
        raise ValueError("Number of parallel processing should be 1 or higher.")

    out_im = nib.Nifti1Image(dFC_funtome, funtome_im.affine)
    out_im.header['pixdim'] = funtome_im.header['pixdim']
    nib.save(out_im, out_f)
    et = int(time.time() - st)
    et_h = et // 3600
    et_min = et % 3600 // 60
    et_sec = et % 3600 % 60
    print(f'Process over.\nElapsed time: {et_h}h {et_min}min {et_sec}sec (= {et} sec)')

# %%


if __name__ == "__main__":
    main()
