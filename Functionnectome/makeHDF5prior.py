#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:43:05 2020

@author: nozais
"""

import os, glob, h5py
import nibabel as nib
import sys 
def header2string(header):
    '''
    Transform the header into a dict, then to a string and clean it so that it 
    can be transformed back to a dict simply usig eval(string_header), and then 
    put back in a new header for re-creation of the nifti file from h5 data.
    
    This transformation to string ease the storage of the header in the hdf5
    '''
    hdrD = dict(header)
    hdr = str(hdrD)
    hdr_cleaned = hdr.replace("\n","")
    hdr_cleaned = hdr_cleaned.replace("=float32","='float32'")
    hdr_cleaned = hdr_cleaned.replace("=int16","='int16'")
    hdr_cleaned = hdr_cleaned.replace("=int32","='int32'")
    hdr_cleaned = hdr_cleaned.replace("=uint8","='uint8'")
    hdr_cleaned = hdr_cleaned.replace("nan","np.nan")
    hdr_cleaned = hdr_cleaned.replace("array","np.array")
    
    return hdr_cleaned

outFile = '/myPath/functionnectome_priors.h5'
templatePath = '/myPath/template_MNI_2mm/MNI152_T1_2mm_brain.nii.gz'
dirVoxel = '/myPath/probaMaps_voxels'
dirRegion = '/myPath/priors_region/probaMaps_regions'
maskRegion = '/myPath/priors_region/masks_regions'

listPmapVoxel = [os.path.basename(x) for x in glob.glob(dirVoxel+'/*.nii*')]
listPmapRegion = [os.path.basename(x) for x in glob.glob(dirRegion+'/*.nii*')]
listMaskRegion = [os.path.basename(x) for x in glob.glob(maskRegion+'/*.nii*')]

listPmapRegion.sort()
listMaskRegion.sort()
if not listPmapRegion==listMaskRegion:
    print('Problem with the region files')
    sys.exit()

template_img = nib.load(templatePath)

with h5py.File(outFile, "a") as h5fout:
    
    # Writting the template
    template = h5fout.create_dataset('template', template_img.shape, template_img.get_data_dtype(), compression="gzip")
    try: #Try to load the volume in the original data type
        template[:] = template_img.get_fdata(dtype=template_img.get_data_dtype())
    except: # if the data type is not valid (for example, integers are not valid with Nibabel), load as float32 then convert
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
    
    # Filling the groups with the data
    for ii,pmapF in enumerate(listPmapRegion):
        print(f'Region {ii+1} on {len(listPmapRegion)}')
        indvox = pmapF.find('.nii')
        reg_ID = pmapF[:indvox]
        reg_img = nib.load(os.path.join(dirRegion,pmapF))
        reg = grp_reg.create_dataset(reg_ID, reg_img.shape, reg_img.get_data_dtype(),compression="gzip")
        try:
            reg[:] = reg_img.get_fdata(dtype=reg_img.get_data_dtype())
        except:
            reg[:] = reg_img.get_fdata().astype(reg_img.get_data_dtype())
        
        regm_img = nib.load(os.path.join(maskRegion,pmapF))
        regm = grp_mreg.create_dataset(reg_ID, regm_img.shape, regm_img.get_data_dtype(),compression="gzip")
        try:
            regm[:] = regm_img.get_fdata(dtype=regm_img.get_data_dtype())
        except:
            regm[:] = regm_img.get_fdata().astype(regm_img.get_data_dtype())
    
    
    for ii,pmapF in enumerate(listPmapVoxel):
        print(f'Region {ii+1} on {len(listPmapVoxel)}')
        indvox1 = pmapF.find('_') + 1
        indvox2 = pmapF.find('.nii')
        vox_ID = pmapF[indvox1:indvox2] #Keeping only the voxel coordinates
        vox_img = nib.load(os.path.join(dirVoxel,pmapF))
        vox = grp_vox.create_dataset(vox_ID, vox_img.shape, vox_img.get_data_dtype(),compression="gzip")
        try:
            vox[:] = vox_img.get_fdata(dtype=vox_img.get_data_dtype())
        except:
            vox[:] = vox_img.get_fdata().astype(vox_img.get_data_dtype())
    
    
    
    
    
    






