#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:43:05 2020

@author: nozais
"""

import os
import glob
import h5py
import nibabel as nib
import sys
import argparse


class MyParser(argparse.ArgumentParser):  # Used to diplay the help if no argment is given
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def header2string(header):
    '''
    Transform the header (extracted from a .nii file by nibabel) into a dict,
    then to a string and clean it so that it can be transformed back to a dict
    simply usig eval(string_header), and then put back in a new header for re-creation
    of the nifti file from h5 data.

    This transformation to string ease the storage of the header in the hdf5
    '''
    hdrD = dict(header)
    hdr = str(hdrD)
    hdr_cleaned = hdr.replace("\n", "")
    hdr_cleaned = hdr_cleaned.replace("=float32", "='float32'")
    hdr_cleaned = hdr_cleaned.replace("=int16", "='int16'")
    hdr_cleaned = hdr_cleaned.replace("=int32", "='int32'")
    hdr_cleaned = hdr_cleaned.replace("=uint8", "='uint8'")
    hdr_cleaned = hdr_cleaned.replace("nan", "np.nan")
    hdr_cleaned = hdr_cleaned.replace("array", "np.array")
    return hdr_cleaned


def run_h5maker(outFile, templatePath, dirVoxel, dirRegion, maskRegion):
    '''
    Create a new HDF5 priors file using the voxel-wise and region-wise probabily maps provided.

    Parameters
    ----------
    outFile : string
        Full path of the output HDF5 file, e.g /myPath/functionnectome_priors.h5
    templatePath : string
        Full path of the NIfTI brain template used to delineate the brain
    dirVoxel : string
        Path to the directory containing the voxel-wise probability maps
    dirRegion : string
        Path to the directory containing the region-wise probability maps
    maskRegion : string
        Path to the directory containing the region masks (1 file per region)

    Returns
    -------
    None.

    '''
    listPmapVoxel = [os.path.basename(x) for x in glob.glob(dirVoxel+'/*.nii*')]
    listPmapRegion = [os.path.basename(x) for x in glob.glob(dirRegion+'/*.nii*')]
    listMaskRegion = [os.path.basename(x) for x in glob.glob(maskRegion+'/*.nii*')]

    listPmapRegion.sort()
    listMaskRegion.sort()
    if not listPmapRegion == listMaskRegion:
        print('Problem with the region files')
        sys.exit()

    template_img = nib.load(templatePath)
    shape3D = template_img.shape

    with h5py.File(outFile, "a") as h5fout:
        # Writting the template
        template = h5fout.require_dataset('template', template_img.shape, template_img.get_data_dtype(),
                                          compression="gzip", chunks=shape3D)
        try:  # Try to load the volume in the original data type
            template[:] = template_img.get_fdata(dtype=template_img.get_data_dtype())
        except ValueError:
            # if the data type is not valid (for example, integers are not valid with Nibabel),
            # load as float32 then convert
            template[:] = template_img.get_fdata().astype(template_img.get_data_dtype())
        # Saving the nifti header in the attributes
        template.attrs['header'] = header2string(template_img.header)

        # Creation of the groups containing the white matter probability maps region-wise and voxel-wise
        grp_vox = h5fout.create_group('tract_voxel')
        vox0_img = nib.load(os.path.join(dirVoxel, listPmapVoxel[0]))
        grp_vox.attrs['header'] = header2string(vox0_img.header)

        grp_reg = h5fout.create_group('tract_region')
        reg0_img = nib.load(os.path.join(dirRegion, listPmapRegion[0]))
        grp_reg.attrs['header'] = header2string(reg0_img.header)

        grp_mreg = h5fout.create_group('mask_region')
        reg0m_img = nib.load(os.path.join(maskRegion, listMaskRegion[0]))
        grp_mreg.attrs['header'] = header2string(reg0m_img.header)

        # Checking if all the files have the correct shape
        if not all([img3D.shape == shape3D for img3D in [vox0_img, reg0_img, reg0m_img]]):
            raise Exception(f'Some of the input files do not have the proper shape {shape3D}')
        # Filling the groups with the data
        for ii, pmapF in enumerate(listPmapRegion):
            print(f'Region {ii+1} on {len(listPmapRegion)}')
            indvox = pmapF.find('.nii')
            reg_ID = pmapF[:indvox]
            reg_img = nib.load(os.path.join(dirRegion, pmapF))
            reg = grp_reg.create_dataset(reg_ID, reg_img.shape, reg_img.get_data_dtype(),
                                         compression="gzip", chunks=shape3D)
            try:
                reg[:] = reg_img.get_fdata(dtype=reg_img.get_data_dtype())
            except ValueError:
                reg[:] = reg_img.get_fdata().astype(reg_img.get_data_dtype())

            regm_img = nib.load(os.path.join(maskRegion, pmapF))
            regm = grp_mreg.create_dataset(reg_ID, regm_img.shape, regm_img.get_data_dtype(),
                                           compression="gzip", chunks=shape3D)
            try:
                regm[:] = regm_img.get_fdata(dtype=regm_img.get_data_dtype())
            except ValueError:
                regm[:] = regm_img.get_fdata().astype(regm_img.get_data_dtype())

        for ii, pmapF in enumerate(listPmapVoxel):
            print(f'Voxel {ii+1} on {len(listPmapVoxel)}')
            indvox1 = pmapF.find('_') + 1
            indvox2 = pmapF.find('.nii')
            vox_ID = pmapF[indvox1:indvox2]  # Keeping only the voxel coordinates
            vox_img = nib.load(os.path.join(dirVoxel, pmapF))
            vox = grp_vox.create_dataset(vox_ID, vox_img.shape, vox_img.get_data_dtype(),
                                         compression="gzip", chunks=shape3D)
            try:
                vox[:] = vox_img.get_fdata(dtype=vox_img.get_data_dtype())
            except ValueError:
                vox[:] = vox_img.get_fdata().astype(vox_img.get_data_dtype())


# %%
def main():
    '''
    Load the arguments from the command line call and launch the script
    '''
    parser = MyParser(description=(
        "Generates an HDF5 file containing the anatomical priors (probability maps) using the"
        "NIfTI files of those priors. The files must be properly named and in separate folders.\n"
        'The voxel-wise probability maps files must all be named with the following pattern:\n'
        'something_xx_yy_zz.nii.gz where "something" can be any string of character, but must not'
        'contain a "_". "xx", "yy", and "zz" are the spatial coordinates of the specific voxel linked'
        "to the given probability map.\n"
        "The region-wise probability maps and their associated region masks must have the same name")
        )
    parser.add_argument("-o", "--H5file", help="Path to the generated HDF5 file (output)."
                        "Should end with the '.h5' extension", required=True)
    parser.add_argument("-tpl", "--templateFile", help="Path to the brain template used for masking",
                        required=True)
    parser.add_argument("-vxl", "--voxelsDir", help="Path to the directory (or folder) "
                        "containing the voxel-wise priors", required=True)
    parser.add_argument("-reg", "--regionsDir", help="Path to the directory (or folder) "
                        "containing the region-wise priors", required=True)
    parser.add_argument("-rmas", "--regionMasksDir", help="Path to the directory (or folder) "
                        "containing the masks of the regions corresponding to the region-wise priors",
                        required=True)

    args = parser.parse_args()

    # Convert relative path to absolute path (might be necessary)
    for iarg in vars(args):
        setattr(args, iarg, os.path.abspath(getattr(args, iarg)))

    run_h5maker(args.H5file, args.templateFile, args.voxelsDir, args.regionsDir, args.regionMasksDir)


if __name__ == '__main__':
    main()
