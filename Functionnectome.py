#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:43:08 2020

@author: nozais

Main script used for the computation of functionnectomes.

"pmap" = probability map (= group level visitation map)

TODO : Add an ETA.
Regionwise and voxelwise analyses started quite differently, but they kind of 
converged on the same algorithm, so I might have to fuse them together later...
"""


import os, sys, glob, time, warnings, multiprocessing, json
from shutil import copyfile
import h5py
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd

#%%
def LogDiplayPercent(logDir,previous_percent=0): # Don't forget to put a 'print("\n100%")' or equivalent after the function
    '''
    Check the logs in logDir and display the progress in percent (look at the 
    last line of each log).
    '''
    logList = glob.glob(os.path.join(logDir,'log_*.txt'))
    if not logList:
        print('Process starting...')
        return
    currentLen = []
    maxETA=0 #Store the longest ETA  among the parallel processes
    for logf in logList:
        with open(logf, "r") as lf:
            logtxt = lf.readlines() #Shouldn't be too big
        lastline = logtxt[-1]
        spiltLastLine = lastline.split(' ')
        procLen = int(spiltLastLine[1])
        currentLen.append(procLen)
        totalLen = int(spiltLastLine[3]) # Should be the same for all the logs
        if len(logtxt)>1:
            prevline = logtxt[-2]
            spiltPrevLine = prevline.split(' ')
            timestep = int(spiltLastLine[5])*60 + int(spiltLastLine[8]) - (int(spiltPrevLine[5])*60 + int(spiltPrevLine[8]))
            stepLength = int(spiltLastLine[1]) - int(spiltPrevLine[1])
            procETA = round(timestep*(totalLen-procLen)/stepLength)
            if procETA > maxETA:
                maxETA = procETA
    meanProgress = sum(currentLen)/len(currentLen)
    percentProgress = round(100*meanProgress/totalLen,2)
    if maxETA:
        hours = maxETA//3600
        minutes = maxETA%3600//60
        if minutes+hours>0:
            ETA = f'{hours}h and {minutes}min'
        else:
            ETA = '< 1 min'
    else:
        ETA = '...'
    if not percentProgress == previous_percent:
        sys.stdout.write(f"\rProgress of the current process: {percentProgress}% ETA: {ETA}      ")
        sys.stdout.flush()
    return percentProgress

def init_worker_sumPmaps(templShape,pmapStore,prior,outDir):
    '''
    Declares the constants used commonly by the workers of the multiproc pool 
    during the computation of the sum of all selected probability maps. Stores
    them in a dictionary ('dict_var').

    Parameters
    ----------
    templShape : tuple
        Shape (3D) of the brain template.
    pmapStore : string
        Path to the file(s) containing the probability maps.
    prior : string
        Type of prior file used ('.nii' or '.h5').
    outDir : string
        Path to the output directory

    Returns
    -------
    None. Because declares global variables.

    '''
    global dict_var
    dict_var={'templateShape': templShape,
              'pmapStore': pmapStore,
              'prior_type': prior,
              'outDir':outDir}


def Sum_regionwise_pmaps(regions_batch):
    '''
    Function given to the pool of workers in parallel.
    Compute the sum of all regions' probability maps.
    '''
    current = multiprocessing.current_process()
    startTime = time.time()
    logFile = os.path.join(dict_var['outDir'],f'log_{current._identity[0]}.txt')
    if dict_var['prior_type']=='nii':
        for ii,reg in enumerate(regions_batch):
            if not ii%5: # Log to follow the progress every few iteration
                ctime = time.time()-startTime
                logtxt = f'Region {ii} in {len(regions_batch)} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                with open(logFile, "a") as log:
                    log.write(logtxt)
            region_pmap_img = nib.load(os.path.join(dict_var['pmapStore'],f'{reg}.nii.gz'))
            if ii==0:
                sum_pmap=np.zeros(dict_var['templateShape'],dtype=region_pmap_img.get_data_dtype())
            try:
                sum_pmap += region_pmap_img.get_fdata(dtype=region_pmap_img.get_data_dtype())
            except:
                sum_pmap += region_pmap_img.get_fdata().astype(region_pmap_img.get_data_dtype())
    elif dict_var['prior_type']=='h5':
        with h5py.File(dict_var['pmapStore'], "r") as h5fout:
            for ii,reg in enumerate(regions_batch):
                if not ii%5: # Log to follow the progress every few iteration
                    ctime = time.time()-startTime
                    logtxt = f'Region {ii} in {len(regions_batch)} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                if ii==0 : sum_pmap=np.zeros(dict_var['templateShape'],dtype=h5fout['tract_region'][reg].dtype)
                sum_pmap += h5fout['tract_region'][reg][:]
    return sum_pmap

def Sum_voxelwise_pmaps(ind_voxels_batch):
    '''
    Function given to the pool of workers in parallel.
    Compute the sum of all selected voxels' probability maps.
    '''
    current = multiprocessing.current_process()
    startTime = time.time()
    logFile = os.path.join(dict_var['outDir'],f'log_{current._identity[0]}.txt')
    if dict_var['prior_type']=='nii':
        for ii,indvox in enumerate(ind_voxels_batch):
            if ii%100==0:
                ctime = time.time()-startTime
                logtxt = f'Voxel {ii} in {len(ind_voxels_batch)} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                with open(logFile, "a") as log:
                    log.write(logtxt)
            vox_pmap_img = nib.load(os.path.join(dict_var['pmapStore'],f'probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz'))
            if ii==0:
                sum_pmap = np.zeros(dict_var['templateShape'], dtype=vox_pmap_img.get_data_dtype())
            sum_pmap += vox_pmap_img.get_fdata(dtype=vox_pmap_img.get_data_dtype())
    elif dict_var['prior_type']=='h5':
        with h5py.File(dict_var['pmapStore'], "r") as h5fout:
            for ii,indvox in enumerate(ind_voxels_batch):
                if not ii%100:
                    ctime = time.time()-startTime
                    logtxt = f'Voxel {ii} in {len(ind_voxels_batch)} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                if ii==0:
                    sum_pmap=np.zeros(dict_var['templateShape'],dtype=h5fout['tract_voxel'][f'{indvox[0]}_{indvox[1]}_{indvox[2]}_vox'].dtype)
                sum_pmap += h5fout['tract_voxel'][f'{indvox[0]}_{indvox[1]}_{indvox[2]}_vox'][:]
    return sum_pmap

def init_worker_regionwise(shared4D,sharedDF,outShape,boldDFshape,bold_ctype,regionDF,nb_of_batchs,pmapStore,prior,outDir):
    '''
    Initialize the process of the current pool worker with the variables 
    commonly used across the different workers.
    '''
    global dict_var
    dict_var={'fun4D': shared4D,
              'funDF': sharedDF,
              'boldShape': outShape,
              'bold_DF_Shape': boldDFshape,
              'bold_ctype': bold_ctype,
              'regionDF': regionDF,
              'nb_batch': nb_of_batchs,
              'pmapStore': pmapStore,
              'prior_type': prior,
              'outDir':outDir}

def Regionwise_functionnectome(batch_num):
    '''
    Computation of the regionwise functionnectome. Used in a pool of workers for 
    multiprocessing.
    Parameters shared between the processes are defined grobaly with the 
    initiator function "init_worker_regionwise".
    They are stored in the "dict_var" dictionary (as a gobal variable for the
    new processes spawned for the multiprocessing).
    
    Parameters
    ----------
    batch_num : int
        Number of the current batch being processed.

    Returns
    -------
    None. The results are in shared memory (in "fun_4D_shared" in the main process)

    '''
    startTime = time.time()
    
    # Loading the 4D array from shared memory, and selecting the batch
    # Empty, will be filled with the functionnectome data for the current batch
    share4D_np = np.frombuffer(dict_var['fun4D'],dict_var['bold_ctype']).reshape(dict_var['boldShape'])
    split_share4D_np = np.array_split(share4D_np, dict_var['nb_batch'],0)
    current_split_out = split_share4D_np[batch_num]
    
    # Loading the dataframe from shared memory, and selecting the batch
    # Contains the functional timeseries of each region
    sharedDF_np = np.frombuffer(dict_var['funDF'],dict_var['bold_ctype']).reshape(dict_var['bold_DF_Shape'])
    sharedDF = pd.DataFrame(sharedDF_np, columns=dict_var['regionDF'])
    split_sharedDF = np.array_split(sharedDF, dict_var['nb_batch'],0)
    current_split_in = split_sharedDF[batch_num]
    
    logFile = os.path.join(dict_var['outDir'],f'log_{batch_num}.txt')
    if dict_var['prior_type']=='nii':
        for ii,reg in enumerate(current_split_in):
            if not ii%10: # Write a line in the log every 10 region
                ctime = time.time()-startTime
                logtxt = f'Region {ii} in {len(list(current_split_in))} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                with open(logFile, "a") as log:
                    log.write(logtxt)
            # Load proba map of the current region and divide it by the sum of all proba map
            region_map_img = nib.load(os.path.join(dict_var['pmapStore'],f'{reg}.nii.gz'))
            region_map = region_map_img.get_fdata(dtype=region_map_img.get_data_dtype())
            current_split_out +=  np.expand_dims(region_map,0)*current_split_in[reg].values.reshape((len(current_split_in[reg]),1,1,1))
    elif dict_var['prior_type']=='h5':
        with h5py.File(dict_var['pmapStore'], "r") as h5fout:
            for ii,reg in enumerate(current_split_in):
                if not ii%10: # Write a line in the log every 10 region
                    ctime = time.time()-startTime
                    logtxt = f'Region {ii} in {len(list(current_split_in))} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                region_map = h5fout['tract_region'][reg][:]
                current_split_out +=  np.expand_dims(region_map,0)*current_split_in[reg].values.reshape((len(current_split_in[reg]),1,1,1))

def init_worker_voxelwise(shared4Dout,bold_ctype,reShape,nb_of_batchs,sharedBold,prior,indvox_shared,pmapStore,outDir):
    '''
    Initialize the process of the current pool worker with the variables 
    commonly used across the different workers.
    '''
    global dict_var
    dict_var={'fun4Dout': shared4Dout,
              'bold_ctype': bold_ctype,
              'boldReshape': reShape,
              'nb_batch': nb_of_batchs,
              'bold': sharedBold,
              'prior_type': prior,
              'voxel_ind': indvox_shared,
              'pmapStore': pmapStore,
              'outDir':outDir}
    
def Voxelwise_functionnectome(batch_num):
    '''
    Computation of the voxelwise functionnectome. Used in a pool of workers for 
    multiprocessing.
    Parameters shared between the processes are defined grobaly with the 
    initiator function "init_worker_voxelwise".
    They are stored in the "dict_var" dictionary (as a gobal variable for the
    new processes spawned for the multiprocessing).
    
    Parameters
    ----------
    batch_num : int
        Number of the current batch being processed.

    Returns
    -------
    None. The results are in shared memory (in "fun_4D_shared" in the main process)

    '''
    startTime = time.time()
    nbTR = dict_var['boldReshape'][0] # Number of volumes (TR) in the 4D data
    
    # Loading the 4D output data from shared memory and selecting the part to
    # fill in the current process (should be empty)
    shared4Dout_np = np.frombuffer(dict_var['fun4Dout'],dict_var['bold_ctype']).reshape(dict_var['boldReshape'])
    split_shared4Dout_np = np.array_split(shared4Dout_np, dict_var['nb_batch'],0)
    batch_4D_out = split_shared4Dout_np[batch_num]
    
    # Loading the input 2D array (temporal x flattened spatial) from shared memory
    shared_2D_bold_np = np.frombuffer(dict_var['bold'],dict_var['bold_ctype']).reshape((nbTR,-1))
    split_shared_2D_bold_np = np.array_split(shared_2D_bold_np, dict_var['nb_batch'],0)
    batch_bold = split_shared_2D_bold_np[batch_num].T # put back the time dim at the end
    nbTRbatch = batch_bold.shape[-1]
    
    # Load the voxels' index (for the results to go back from 2D to 4D)
    shared_vox_ind_np = np.frombuffer(dict_var['voxel_ind'],'i').reshape((-1,3))
    
    logFile = os.path.join(dict_var['outDir'],f'log_{batch_num}.txt')
    
    if dict_var['prior_type']=='nii':
        for ii,indvox in enumerate(shared_vox_ind_np):
            if ii%100==0: # Check the progress every 100 steps
                ctime = time.time()-startTime
                logtxt = f'Voxel {ii} in {len(shared_vox_ind_np)} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                with open(logFile, "a") as log:
                    log.write(logtxt)
            # Load the probability map of the current voxel, combine it with the functional signal, and add it to the results
            vox_pmap_img = nib.load(os.path.join(dict_var['pmapStore'],f'probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz'))
            vox_pmap = vox_pmap_img.get_fdata(dtype=vox_pmap_img.get_data_dtype())
            batch_4D_out += np.expand_dims(vox_pmap,0)*batch_bold[ii].reshape((nbTRbatch,1,1,1))
            
    elif dict_var['prior_type']=='h5':
        with h5py.File(dict_var['pmapStore'], "r") as h5fout:
            for ii,indvox in enumerate(shared_vox_ind_np):
                if ii%100==0:# Check the progress every 100 steps
                    ctime = time.time()-startTime
                    logtxt = f'Voxel {ii} in {len(shared_vox_ind_np)} : {int(ctime//60)} min and {int(ctime%60)} sec\n'
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                # Load the probability map of the current voxel, combine it with the functional signal, and add it to the results
                vox_pmap = h5fout['tract_voxel'][f'{indvox[0]}_{indvox[1]}_{indvox[2]}_vox'][:]
                batch_4D_out += np.expand_dims(vox_pmap,0)*batch_bold[ii].reshape((nbTRbatch,1,1,1))



#%%
def run_functionnectome(settingFilePath):
    '''
    Main functionction. Run the computation and call the other functions.
    
    Parameters
    ----------
    settingFilePath : string
        Path to the settings file (.fcntm)
    '''
    print('Process starting...')
    st=time.time()
    
    print('Loading settings')
    #Read the setting file given to set the input variables
    with open(settingFilePath,'r') as f:
        settings = f.read().split('\n')
        settings = list(map(lambda s: s.strip('\t'), settings))
        # Test if the settings file is properly organized (see the output of the GUI for an example)
        orgTest = [settings[0]=='Output folder:',
                   settings[2]=="Analysis ('voxel' or 'region'):",
                   settings[4]=='Number of parallel processes:',
                   settings[6]=="Priors stored as ('h5' or 'nii'):",
                   settings[8]=='Position of the subjects ID in their path:',
                   settings[10]=='Mask the output:',
                   settings[12]=='Number of subjects:',
                   settings[14]=='Number of masks:',
                   settings[16]=="Subject's BOLD paths:",
                   settings[16+int(settings[13])+2]=="Masks for voxelwise analysis:"
                   ]
        if not all(orgTest):
            print('Settings file with bad internal organization')
            sys.exit()
        # Filling the input variables with the settings from the file
        results_dir_root = settings[1]
        anatype = settings[3]
        nb_of_batchs = int(settings[5])
        prior_type = settings[7]
        subIDpos = int(settings[9])
        maskOutput = int(settings[11])
        subNb = int(settings[13])
        bold_paths = []
        for isub in range(17, 17+subNb):
            bold_paths.append(settings[isub])
        if anatype=='voxel':
            mask_nb = int(settings[15])
            masks_vox = []
            startline = 19 + subNb
            for imask in range(startline, startline+mask_nb):
                masks_vox.append(settings[imask])
        
        # Optional variables at the end of the file, to change the priors paths written here
        while not settings[-1]: # removing empty lines at the end of the file 
            settings.pop()
        if settings[-1]=='###': # "###" marks the presence of the optional settings
            opt_h5_loc = settings[-10]
            opt_template_path = settings[-8]
            opt_pmap_vox_loc = settings[-6]
            opt_pmap_region_loc = settings[-4]
            opt_regions_loc = settings[-2]
    
    # Checking for the existence of priors and asking what to do if none found 
    # TODO: Code the part where the script ask for missing priors
    priors_paths={'h5_priors':'/put/your/path/here/functionnectome_7Tpriors.h5'} # TODO: Temporary fix. To be removed in v0.1.0
    if not settings[-1]=='###': # If non default priors are used, override this
        pkgPath = os.path.dirname(__file__)
        priorPath = os.path.join(pkgPath,'priors_paths.json')
        if os.path.exists(priorPath):
            with open(priorPath,'r') as pP:
                priors_paths = json.load(pP)
            if prior_type=='h5' and priors_paths['h5_priors']:
                if not os.path.exists(priors_paths['h5_priors']):
                    raise Exception("Using HDF5 priors, but the file was not found")
                else:
                    # TODO: Ask for the missing path to the user and set it in the file
                    pass 
            if prior_type=='nii':
                if priors_paths['template']:
                    if not os.path.exists(priors_paths['template']):
                        raise Exception("Using NifTI priors, but the brain template file was not found")
                else:
                    # TODO: Ask for the missing path to the user and set it in the file
                    pass 
                
                if anatype=='region' and priors_paths['regions'] and priors_paths['region_pmap']:
                    if not (os.path.exists(priors_paths['regions']) and os.path.exists(priors_paths['region_pmap'])):
                        raise Exception("Using NifTI priors with regionwise analysis,"
                                        " but at least some of the files/folders were not found")
                    else:
                        # TODO: Ask for the missing path to the user and set it in the file
                        pass 
                if anatype=='voxel' and not os.path.exists(priors_paths['voxel_pmap']):
                    raise Exception("Using NifTI priors with voxelwise analysis,"
                                    " but the folder containing the probability maps was not found")
        else:
            # TODO: Choose the files location or download them
            pass
    
    # Association of the priors path with their variables
    if prior_type=='nii':
        if settings[-1]=='###':
            if not (opt_template_path and (opt_pmap_vox_loc or (opt_pmap_region_loc and opt_regions_loc))):
                raise Exception("Using optional settings, but some are not filled.")
            template_path = opt_template_path
            if anatype=='region':
                pmap_loc = opt_pmap_region_loc
                regions_loc = opt_regions_loc
            elif anatype=='voxel':
                pmap_loc = opt_pmap_vox_loc
            else:
                raise Exception('Bad type of analysis (not "voxel" nor "region")')
        else:
                
            template_path = priors_paths['template']
            if anatype=='region':
                pmap_loc = priors_paths['region_pmap']
                regions_loc = priors_paths['regions']
                    
            elif anatype=='voxel':
                pmap_loc = priors_paths['voxel_pmap']
            else:
                raise Exception('Bad type of analysis (not "voxel" nor "region")')
    elif prior_type=='h5':
        if settings[-1]=='###': # Two cases: a path to an external HDF5 file, or a path to a .nii template (i.e we use the default .h5 for everything but the template)
            if not any((opt_h5_loc,opt_template_path)): raise Exception("Using optional settings, but no path given.")
            if opt_h5_loc:
                h5_loc = opt_h5_loc
                template_path=''
            if opt_template_path:
                print('Using priors from the HDF5 file except for the template, imported from a nifti file.')
                template_path = opt_template_path
                if opt_h5_loc:
                    h5_loc = opt_h5_loc
                else:
                    priors_paths['h5_priors']
        else:
            h5_loc = priors_paths['h5_priors']
            template_path=''
        pmap_loc = h5_loc
        regions_loc = h5_loc
    else:
        raise Exception('Bad type of priors (not "nii" nor "h5")')
    
    
    Path(results_dir_root).mkdir(parents=True, exist_ok=True) # Create results_dir_root if needed
    if not os.path.dirname(settingFilePath)==results_dir_root: # If the setting file is imported from somewhere else
        n=1
        fpath = os.path.join(results_dir_root,'settings.fcntm') # Default name
        while os.path.exists(fpath):
            fpath = os.path.join(results_dir_root,f'settings{n}.fcntm')
            n+=1
        copyfile(settingFilePath,fpath) # Save the settings into the result directory
    
    # Get the basic info about the input (shape, file header, list of regions, ...) 
    if prior_type=='nii':
        # MNI template
        template_img = nib.load(template_path)
        template_vol = template_img.get_fdata().astype(bool)# binarized
        if anatype=='region':
            listRegionsFiles = glob.glob(os.path.join(regions_loc,'*.nii.gz'))
            listRegions = [os.path.basename(reg).rsplit('.', 2)[0] for reg in listRegionsFiles]
            listRegions.sort()
            regions_batchs = np.array_split(listRegions,nb_of_batchs)
            base_region_img = nib.load(listRegionsFiles[0])
            affine3D = base_region_img.affine
            header3D = base_region_img.header
            
        elif anatype=='voxel':
            listVoxFiles = glob.glob(os.path.join(pmap_loc,'*.nii.gz'))
            base_vox_img = nib.load(listVoxFiles[0])
            affine3D = base_vox_img.affine
            header3D = base_vox_img.header
            listVoxFiles=None
    elif prior_type=='h5':
        with h5py.File(pmap_loc, "r") as h5fout:
            if template_path:
                template_img = nib.load(template_path)
                template_vol = template_img.get_fdata().astype(bool)# binarized
            else:
                template_vol = h5fout['template'][:]
                template_vol = template_vol.astype(bool)
            
            if  anatype=='region':
                hdr = h5fout['tract_region'].attrs['header']
                hdr3D = eval(hdr)
                header3D = nib.Nifti1Header()
                for key in header3D.keys():
                    header3D[key] = hdr3D[key]
                affine3D = header3D.get_sform()
                listRegions = list(h5fout['tract_region'].keys())
                listRegions.sort() #Should already be sorted anyways
                regions_batchs = np.array_split(listRegions,nb_of_batchs)
            elif anatype=='voxel':
                hdr = h5fout['tract_voxel'].attrs['header']
                hdr3D = eval(hdr)
                header3D = nib.Nifti1Header()
                for key in header3D.keys():
                    header3D[key] = hdr3D[key]
                affine3D = header3D.get_sform()
                
    # Retrieves all subjects' unique IDs from their path
    IDs=[]
    for boldf in bold_paths:
        decomposedPath = tuple(filter(None, os.path.normpath(boldf).split(os.path.sep))) # cf. the GUI
        if (subIDpos==len(decomposedPath)-1) or subIDpos==-1:# case where subject ID is the BOLD file's name
            subID = decomposedPath[subIDpos].replace('.gz','').replace('.nii','') # REmove extension
        else:
            subID = decomposedPath[subIDpos]
        IDs.append(subID)
    #%% Start the loop over all the input files (i.e. "subjects")
    for isub,subID,boldf in zip(range(len(bold_paths)),IDs,bold_paths): 
        
        print(f'Processing subject {subID} in {anatype}wise analysis')
        results_dir = os.path.join(results_dir_root, anatype + 'wise_analysis',subID)
        finalOutPath = os.path.join(results_dir,'functionnectome.nii.gz')
        if os.path.exists(finalOutPath):
            print(f'Output file already exists for subject {subID}. Skiping...')
            continue
        Path(results_dir).mkdir(parents=True, exist_ok=True) #Create results_dir if needed
        
        # If there are some old and older log files (there shouldn't though...)
        old_logs = glob.glob(os.path.join(results_dir,'log_*.txt'))
        very_old_logs = glob.glob(os.path.join(results_dir,'old_log_*.txt'))
        for veryoldlog in very_old_logs: # Delete the older logs
            os.remove(veryoldlog)
        for oldlog in old_logs: # And save (for now) the old (but more recent) logs
            logname = 'old_' + os.path.basename(oldlog)
            os.rename(oldlog,os.path.join(results_dir,logname))
        
        #%% Run the regionwise analysis
        if anatype=='region':
            
            # Launching parallel processing for the sum all the regions' probability maps for later normalization
            sumpath = os.path.join(results_dir_root,'sum_probaMaps_regions.nii.gz')
            if not os.path.exists(sumpath):
                with multiprocessing.Pool(processes=nb_of_batchs,
                                          initializer=init_worker_sumPmaps,
                                          initargs=(template_vol.shape,
                                                    pmap_loc,
                                                    prior_type,
                                                    results_dir)
                                          ) as pool:
                    print('Launching parallel computation: Sum of probability maps')
                    out_batch_sum = pool.map_async(Sum_regionwise_pmaps,regions_batchs)
                    # Diplay the progress
                    percent = None # to keep track of the preivous
                    while not out_batch_sum.ready():
                        percent = LogDiplayPercent(results_dir,percent)
                        time.sleep(1)
                    sys.stdout.write("\rProgress of the current process: 100%    \n")
                    sys.stdout.flush()
                    out_batch_sum = out_batch_sum.get()
                    logfiles = glob.glob(os.path.join(results_dir,'log_*.txt'))
                    for logf in logfiles:
                        os.remove(logf)
                
                print('Multiprocessing done. Sum & save now.')
                sum_pmap_all = np.sum(out_batch_sum,0)
                out_batch_sum=None # Release the RAM
                sum_pmap_img = nib.Nifti1Image(sum_pmap_all,affine3D,header3D)
                nib.save(sum_pmap_img,sumpath)
            else:
                alreadyThereMsg = ('Sum of probability maps already computed previously. Reloading it.\n'
                                   'WARNING: If you changed the regions for the analysis, '
                                   'stop the process, delete the "sum_probaMaps_regions.nii.gz" file '
                                   'and relaunch the analysis.')
                print(alreadyThereMsg)
                sum_pmap_img = nib.load(sumpath)
                sum_pmap_all = sum_pmap_img.get_fdata(dtype=sum_pmap_img.get_data_dtype())
                
            # Loading the 4D BOLD file
            print('Loading 4D data...')
            bold_img = nib.load(boldf)
            bold_dtype = bold_img.get_data_dtype().name
            bold_vol = bold_img.get_fdata(dtype=bold_dtype)
            bold_header = bold_img.header
            bold_affine = bold_img.affine
            bold_shape = bold_img.shape
            
            # Computing the DataFrame containing the median timeseries of all the regions
            print('Computation of BOLD median for each region')
            regions_BOLD_median = pd.DataFrame(columns=listRegions)
            if prior_type=='nii':
                for reg in listRegions:
                    region_img = nib.load(os.path.join(regions_loc,f'{reg}.nii.gz'))
                    region_vol = region_img.get_fdata(dtype=region_img.get_data_dtype())
                    region_vol *= template_vol # masking
                    if region_vol.sum(): # if region not empty
                        region_BOLD = bold_vol[np.where(region_vol)]
                        regions_BOLD_median[reg] = np.median(region_BOLD,0)
            elif prior_type=='h5':
                with h5py.File(pmap_loc, "r") as h5fout:
                    grp_reg = h5fout['mask_region']
                    for reg in listRegions:
                        region_vol = grp_reg[reg][:]
                        region_vol *= template_vol # masking
                        if region_vol.sum(): # if region not empty
                            region_BOLD = bold_vol[np.where(region_vol)]
                            regions_BOLD_median[reg] = np.median(region_BOLD,0)
            
            # Release the RAM
            bold_vol=bold_img=None
                  
            # Launching parallel processing for the functionnectome computation proper
            if np.dtype(bold_dtype) is np.dtype('float32'):
                bold_ctype = 'f'
            elif np.dtype(bold_dtype) is np.dtype('float64'):
                bold_ctype = 'd'
            else:
                badTypeMsg = (f"Bold data-type ({bold_dtype}) is not float32 (a.k.a 'float') "
                              "nor float64 (a.k.a 'double'), which is not normal and is not "
                              "currently supported")
                warnings.warn(badTypeMsg)
                break
            
            # Create a shared RawArray containing the data from the BOLD regionwise DataFrame
            boldDFshape = regions_BOLD_median.shape
            bold_DF_shared = multiprocessing.RawArray(bold_ctype, int(np.prod(boldDFshape)))
            bold_DF_shared_np = np.frombuffer(bold_DF_shared,bold_ctype).reshape(boldDFshape) # To manipulate the RawArray as a numpy array
            np.copyto(bold_DF_shared_np, regions_BOLD_median.values) # Filling the RawArray
            regions_BOLD_median = None
            
            # Create a shared RawArray that will contain the results
            bold_reshape = (bold_shape[-1], *bold_shape[:-1]) #puting the time dim first (for contiguous data array, necessary to avoid copy with shared memory access)
            fun_4D_shared = multiprocessing.RawArray(bold_ctype, int(np.prod(bold_reshape)))
            
            with multiprocessing.Pool(processes=nb_of_batchs,
                                      initializer=init_worker_regionwise,
                                      initargs=(fun_4D_shared,
                                                bold_DF_shared,
                                                bold_reshape,
                                                boldDFshape,
                                                bold_ctype,
                                                listRegions,
                                                nb_of_batchs,
                                                pmap_loc,
                                                prior_type,
                                                results_dir)
                                      ) as pool:
                poolCheck = pool.map_async(Regionwise_functionnectome,range(nb_of_batchs))
                # Diplay the progress
                percent = None # to keep track of the preivous
                while not poolCheck.ready():
                    percent = LogDiplayPercent(results_dir,percent)
                    time.sleep(1)
                sys.stdout.write("\rProgress of the current process: 100%    \n")
                sys.stdout.flush()
                logfiles = glob.glob(os.path.join(results_dir,'log_*.txt'))
                for logf in logfiles:
                    os.remove(logf)
            print('Multiprocessing done. Application of the proportionality and saving results.')
            sum_pmap4D_all = np.frombuffer(fun_4D_shared,bold_ctype).reshape(bold_reshape)
            # Applying proportionality
            sum_pmap4D_all = np.divide(sum_pmap4D_all, 
                                       np.expand_dims(sum_pmap_all,0),
                                       out=sum_pmap4D_all,
                                       where=np.expand_dims(sum_pmap_all,0)!=0) 
            sum_pmap4D_all = np.moveaxis(sum_pmap4D_all, 0, -1)
            # Masking the output with the template
            if maskOutput:
                sum_pmap4D_all *= np.expand_dims(template_vol,-1)
            # Saving the results
            sum_pmap4D_img = nib.Nifti1Image(sum_pmap4D_all,bold_affine,bold_header)
            nib.save(sum_pmap4D_img,finalOutPath)
            time.sleep(5) # Waiting a bit, just in case...
            sum_pmap4D_img=sum_pmap4D_all=None
        #%% Run the voxelwise analysis
        elif anatype=='voxel':
            
            # Loading subject's mask, restriction to voxels inside the template, and getting the list of voxels' indexes
            if len(masks_vox)>1:
                mask_path = masks_vox[isub]
            elif len(masks_vox)==1:
                mask_path = masks_vox[0]
            else:
                mask_path=''
            
            if mask_path:
                voxel_mask_img = nib.load(mask_path)
                voxel_mask = voxel_mask_img.get_fdata().astype(bool)
                voxel_mask *= template_vol
            else:
                print("No mask given => using all voxels... Not a super good idea.")
                voxel_mask=template_vol
            ind_mask1 = np.nonzero(voxel_mask)
            ind_mask2 = np.transpose(ind_mask1).astype('i')
            voxel_mask_img=voxel_mask=None
            
            # Division of arg_voxels in batches used by the grid 
            split_ind = np.array_split(ind_mask2,nb_of_batchs)
            
            # Summing all the proba maps, for later "normalization"
            if len(masks_vox)==1:
                firstSubRes = os.path.join(results_dir_root, anatype + 'wise_analysis',IDs[0])
                sumpath = os.path.join(firstSubRes, 'sum_probaMaps_voxel.nii.gz')
            else:
                sumpath = os.path.join(results_dir, 'sum_probaMaps_voxel.nii.gz')
                
            if not os.path.exists(sumpath):
                
                print('Launching parallel computation: Sum of Probability maps')
                with multiprocessing.Pool(processes=nb_of_batchs,
                                          initializer=init_worker_sumPmaps,
                                          initargs=(template_vol.shape,
                                                    pmap_loc,
                                                    prior_type,
                                                    results_dir)
                                          ) as pool:
                    out_batch_sum = pool.map_async(Sum_voxelwise_pmaps,split_ind)
                    # Diplay the progress
                    percent = None # to keep track of the preivous
                    while not out_batch_sum.ready():
                        percent = LogDiplayPercent(results_dir,percent)
                        time.sleep(1)
                    sys.stdout.write("\rProgress of the current process: 100%    \n")
                    sys.stdout.flush()
                    out_batch_sum=out_batch_sum.get()
                    logfiles = glob.glob(os.path.join(results_dir,'log_*.txt'))
                    for logf in logfiles:
                        os.remove(logf)
                sum_pmap_all = np.sum(out_batch_sum,0)
                out_batch_sum=None # Release the RAM
                sum_pmap_img = nib.Nifti1Image(sum_pmap_all,affine3D,header3D)
                nib.save(sum_pmap_img,sumpath)
            else:
                alreadyThereMsg = ('Sum of probability maps already computed previously. Reloading it.\n'
                                   'WARNING: If you changed the mask for the analysis, '
                                   'stop the process, delete the "sum_probaMaps_voxel.nii.gz" file '
                                   'and relaunch the analysis.')
                # print(alreadyThereMsg)
                warnings.warn(alreadyThereMsg)
                sum_pmap_img = nib.load(sumpath)
                sum_pmap_all = sum_pmap_img.get_fdata(dtype=sum_pmap_img.get_data_dtype())
            
            # Loading the 4D BOLD file
            print('Loading 4D data...')
            bold_img = nib.load(boldf)
            bold_dtype = bold_img.get_data_dtype().name
            bold_vol = bold_img.get_fdata(dtype=bold_dtype)
            bold_header = bold_img.header
            bold_affine = bold_img.affine
            bold_shape = bold_img.shape
            
            if np.dtype(bold_dtype) is np.dtype('float32'):
                bold_ctype = 'f'
            elif np.dtype(bold_dtype) is np.dtype('float64'):
                bold_ctype = 'd'
            else:
                badTypeMsg = (f"Bold data-type ({bold_dtype}) is not float32 (a.k.a 'float') "
                              "nor float64 (a.k.a 'double'), which is not normal and is not "
                              "currently supported")
                warnings.warn(badTypeMsg)
                break
            
            #Creating shared memory variables accessed by the parrallel processes
            bold_shape = bold_vol.shape
            bold_vol_2D = bold_vol[ind_mask1] #Select the voxels from the mask => makes a 2D array (flattened sptial x temporal)
            bold_vol=bold_img=ind_mask1=None # Release the RAM
            bold_2D_shared = multiprocessing.RawArray(bold_ctype, int(np.prod(bold_vol_2D.shape)))
            bold_2D_shared_np = np.frombuffer(bold_2D_shared,bold_ctype).reshape(bold_vol_2D.T.shape) # To manipulate the RawArray as a numpy array
            np.copyto(bold_2D_shared_np, bold_vol_2D.T) # Filling the RawArray
            bold_vol_2D=None
            
            # Create a shared RawArray containing the index of each used voxel
            ind_mask_shared = multiprocessing.RawArray('i',int(np.prod(ind_mask2.shape)))
            ind_mask_shared_np = np.frombuffer(ind_mask_shared, 'i').reshape(ind_mask2.shape)
            np.copyto(ind_mask_shared_np, ind_mask2)
            ind_mask2=None
            
            # Create a shared RawArray that will contain the results
            bold_reshape = (bold_shape[-1],)+(bold_shape[:-1])
            fun_4D_shared = multiprocessing.RawArray(bold_ctype, int(np.prod(bold_reshape)))
            
            with multiprocessing.Pool(processes=nb_of_batchs,
                                      initializer=init_worker_voxelwise,
                                      initargs=(fun_4D_shared,
                                                bold_ctype,
                                                bold_reshape,
                                                nb_of_batchs,
                                                bold_2D_shared,
                                                prior_type,
                                                ind_mask_shared,
                                                pmap_loc,
                                                results_dir)
                                    ) as pool:
                poolCheck = pool.map_async(Voxelwise_functionnectome,range(nb_of_batchs))
                percent = None # to keep track of the preivous
                # Diplay the progress
                while not poolCheck.ready():
                    percent = LogDiplayPercent(results_dir,percent)
                    time.sleep(1)
                sys.stdout.write("\rProgress of the current process: 100%    \n")
                sys.stdout.flush()
                logfiles = glob.glob(os.path.join(results_dir,'log_*.txt'))
                for logf in logfiles:
                    os.remove(logf)
            bold_2D_shared=bold_2D_shared_np=None
            
            print('Multiprocessing done. Application of the proportionality and saving results.')
            sum_pmap4D_all = np.frombuffer(fun_4D_shared,bold_ctype).reshape(bold_reshape)
            # Applying proportionality
            sum_pmap4D_all = np.divide(sum_pmap4D_all, 
                                        np.expand_dims(sum_pmap_all,0),
                                        out=sum_pmap4D_all,
                                        where=np.expand_dims(sum_pmap_all,0)!=0) 
            sum_pmap4D_all = np.moveaxis(sum_pmap4D_all, 0, -1)
            # Masking out the stray voxels out of the brain
            if maskOutput:
                sum_pmap4D_all *= np.expand_dims(template_vol,-1)
            # Saving the results
            sum_pmap4D_img = nib.Nifti1Image(sum_pmap4D_all,bold_affine,bold_header)
            nib.save(sum_pmap4D_img,finalOutPath)
            time.sleep(5)
            sum_pmap4D_img=sum_pmap4D_all=None
        
    print(f'Total run time : {time.time()-st}')

#%% Run the code if the script is called directly, the path to the setting file must be given as argument
if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3. And probably Python 3.6 (or superior).")
    if sys.version_info[1] < 6:
        warnings.warn("Python version < 3.6 |nIt might not work. Consider updating.")
    settingFilePath = sys.argv[1]
    run_functionnectome(settingFilePath)







