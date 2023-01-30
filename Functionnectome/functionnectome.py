#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
"""
Created on Wed Jun 10 15:43:08 2020

@author: nozais

Main script used for the computation of functionnectomes.

"pmap" = probability map (= group level visitation map)

Data is imported as float32 for all volumes, to simplify and speed up computation

TODO:Regionwise and voxelwise analyses started quite differently, but they kind of
converged on the same algorithm, so I might have to fuse them together later...
"""


import pandas as pd
from pathlib import Path
import nibabel as nib
import numpy as np
import h5py
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
from urllib.request import urlopen
import threading
import zipfile
import shutil
import json
import multiprocessing
import warnings
import time
import glob
import sys
import os

# Setting the number of threads for numpy (before importing numpy)
threads = 1
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

# %% Information about the priors, shared as gloaba variable. Update here.

PRIORS_INFO = (  # TODO : Fill the URL when available
    ('V2.D.WB - Whole brain, Deterministic',
     "https://www.dropbox.com/s/aj84fl3la3yymv7/priors_full_deter3T.h5.zip?dl=1",
     "priors_full_deter3T.h5.zip",
     ),
    ('V2.D.Asso - Association, Deterministic',
     "https://www.dropbox.com/s/r824b5vk5apvp6o/priors_asso_deter3T.h5.zip?dl=1",
     "priors_asso_deter3T.h5.zip",
     ),
    ('V2.D.Proj - Projection, Deterministic',
     "https://www.dropbox.com/s/ln8rdh3oyftvrid/priors_proj_deter3T.h5.zip?dl=1",
     "priors_proj_deter3T.h5.zip",
     ),
    ('V2.D.Comm - Commissural, Deterministic',
     "https://www.dropbox.com/s/xunfqlzaracfp06/priors_comm_deter3T.h5.zip?dl=1",
     "priors_comm_deter3T.h5.zip",
     ),
    ('V2.D.Cereb - Cerebellar, Deterministic',
     "https://www.dropbox.com/s/do5ou7datoxv7ya/priors_cereb_deter3T.h5.zip?dl=1",
     "priors_cereb_deter3T.h5.zip",
     ),
    ('V2.P.WB - Whole brain, Probabilistic',
     "https://www.dropbox.com/s/j5a6sh3w7bltuez/priors_full_proba3T.h5.zip?dl=1",
     "priors_full_proba3T.h5.zip",
     ),
    ('V2.P.Asso - Association, Probabilistic',
     "https://www.dropbox.com/s/yxcs6sjp34io9f1/priors_asso_proba3T.h5.zip?dl=1",
     "priors_asso_proba3T.h5.zip",
     ),
    ('V2.P.Proj - Projection, Probabilistic',
     "https://www.dropbox.com/s/v71j6q5y7l48qeq/priors_proj_proba3T.h5.zip?dl=1",
     "priors_proj_proba3T.h5.zip",
     ),
    ('V2.P.Comm - Commissural, Probabilistic',
     "https://www.dropbox.com/s/0ajvcei25j2zf72/priors_comm_proba3T.h5.zip?dl=1",
     "priors_comm_proba3T.h5.zip",
     ),
    ('V2.P.Cereb - Cerebellar, Probabilistic',
     "https://www.dropbox.com/s/nfq9gh09u4whw72/priors_cereb_proba3T.h5.zip?dl=1",
     "priors_cereb_proba3T.h5.zip",
     ),
    ('V1.D.WB - Whole brain, Deterministic (legacy)',
     "https://www.dropbox.com/s/22vix4krs2zgtnt/functionnectome_7TpriorsH5.zip?dl=1",
     "functionnectome_7TpriorsH5.zip",
     ),
)
PRIORS_H5 = tuple(pinfo[0] for pinfo in PRIORS_INFO)
PRIORS_URL = {pinfo[0]: pinfo[1] for pinfo in PRIORS_INFO}
PRIORS_ZIP = {pinfo[0]: pinfo[2] for pinfo in PRIORS_INFO}


# %%  Additionnal GUI and interactive functions
def check_DLprogress(datasize, zipname):
    """Check the progress of the DL"""
    percentProgress = 0
    prog_thread = threading.currentThread()
    while getattr(prog_thread, "do_run", True):
        if os.path.exists(zipname):
            dl_size = os.path.getsize(zipname)
            percentProgress = round(100 * dl_size / datasize)
            sys.stdout.write(f"\rDownloading progress: {percentProgress}%       ")
            sys.stdout.flush()
            if percentProgress >= 100:
                break
        time.sleep(1)


def Download_H5(prior_dirpath_h5, priorsName):
    '''
    Downloads the HDF5 priors from the internet.
    Takes as input:
        the folder where to save the files
        the priors label indicated which prior to DL
        (opt.) The tkinter instance of the GUI
    Returns the path of the DLed priors after unzipping.
    '''
    dl_url = PRIORS_URL[priorsName]
    fname = PRIORS_ZIP[priorsName]
    zipname = os.path.join(prior_dirpath_h5, fname)
    if os.path.exists(zipname):  # If a previous file was left because the process was interrupted
        os.remove(zipname)
    print("Downloading the priors...")
    with urlopen(dl_url) as response, open(zipname, "wb") as out_file:
        datasize = int(response.getheader("content-length"))
        prog_thread = threading.Thread(
            target=check_DLprogress, args=(datasize, zipname), daemon=True
        )
        prog_thread.start()
        shutil.copyfileobj(response, out_file)
        prog_thread.do_run = False
        prog_thread.join()
        sys.stdout.write("\rDownloading progress: 100%           \n")
        sys.stdout.flush()
    print("Unzipping...")
    with zipfile.ZipFile(zipname, "r") as zip_ref:
        filename = zip_ref.namelist()[0]
        outPath = zip_ref.extract(filename, prior_dirpath_h5)
    os.remove(zipname)
    print("Done")
    return outPath


def find_missingH5(dictPaths):
    checked_dictPaths = {k: dictPaths[k] for k in dictPaths.keys() if os.path.exists(dictPaths[k])}
    missingH5 = PRIORS_URL.keys() - checked_dictPaths.keys()
    return missingH5


def DL_missingH5(dictPaths, h5Dir, currentH5=None):
    ''' Download all h5 priors available but still missing in the json'''
    missingH5 = find_missingH5(dictPaths) - {currentH5}
    for n, priorname in enumerate(missingH5):
        print(f"Downloading file {n + 1}/{len(missingH5)} ({priorname})")
        missH5P = Download_H5(h5Dir, priorname)
        if os.path.exists(missH5P):
            dictPaths[priorname] = missH5P
    return dictPaths


class Ask_hdf5_path(tk.Tk):
    """
    Ask of the path to the priors in HDF5 file, or offer to download them
    Works with a GUI if the Functionnecomte has been launched from the GUI, with command lines otherwise

    Returns the path to the HDF5 priors file, or False if the action was canceled

    """

    def __init__(self, pName, dictH5):
        super().__init__()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        size = tuple(int(_) for _ in self.geometry().split("+")[0].split("x"))
        x = screen_width / 2 - size[0] / 2
        y = screen_height / 2 - size[1] / 2
        self.geometry("+%d+%d" % (x, y))
        #
        self.title("Acquisition of HDF5 file for priors")
        self.prior_path_h5 = ""  # Where the file path to the priors will be stored
        self.home = os.path.expanduser("~")
        self.dictH5 = dictH5
        priorFiles = [f for f in self.dictH5.values() if os.path.exists(f)]
        if len(priorFiles):
            self.home = os.path.dirname(priorFiles[-1])
        msg = (
            "No HDF5 priors file was found. If you already downloaded it, "
            "select select the file. Otherwise download it."
        )
        self.lbl = tk.Label(self, text=msg)
        self.lbl.grid(column=0, row=0, columnspan=3)
        #
        self.priorsName = pName
        self.btnDL = tk.Button(self, text="Download priors", command=self.Btn_DL_priors)
        self.btnDL.grid(column=0, row=1)
        #
        self.btnSelect = tk.Button(
            self, text="Select priors file", command=self.Btn_select_file
        )
        self.btnSelect.grid(column=1, row=1)
        #
        self.btnCancel = tk.Button(self, text="Cancel", command=self.destroy)
        self.btnCancel.grid(column=2, row=1)

    def Btn_DL_priors(self):
        self.DLall = tk.messagebox.askyesno('Download all priors?', 'Also download the full suit of priors?')
        prior_dirpath_h5 = filedialog.askdirectory(
            initialdir=self.home, parent=self, title="Choose where to save the priors"
        )
        if prior_dirpath_h5:
            # Should be the final priors path
            self.withdraw()
            self.prior_path_h5 = Download_H5(prior_dirpath_h5, self.priorsName)
            if self.DLall:
                self.dictH5 = DL_missingH5(self.dictH5, prior_dirpath_h5, self.prior_path_h5)
            self.destroy()

    def Btn_select_file(self):
        self.prior_path_h5 = filedialog.askopenfilename(
            parent=self,
            initialdir=self.home,
            title="Select the HDF5 priors file",
            filetypes=[("HDF5 files", ".h5 .hdf5")],
        )
        if self.prior_path_h5:
            self.withdraw()
            self.destroy()


# %% Main functions

def testInputType(inVal, inLab, dtypeLab, opt, forGUI):
    '''
    Check if if retrieved setting correspond to the type it is supposed to.
    '''
    if isinstance(dtypeLab, tuple):  # The tuple should contain all possible inVal
        if inVal not in dtypeLab:
            error = ValueError(f"Wrong input value for '{inLab}'. Was expecting"
                               f" a value among {dtypeLab} but got '{inVal}'.")
            if forGUI:
                return error
            else:
                raise error
    elif isinstance(dtypeLab, type):
        if dtypeLab is int:
            try:
                inVal = int(inVal)
            except ValueError:
                error = ValueError(f"Wrong input value for '{inLab}'. Was expecting"
                                   f" an integer but got '{inVal}'.")
                if forGUI:
                    return error
                else:
                    raise error
        elif isinstance(inVal, dtypeLab):
            pass
        else:
            raise TypeError(f"Expecting the value to be '{dtypeLab}' but is '{type(inVal)}', "
                            'which is unexpected...')
    elif opt and inVal is None:
        pass
    else:
        raise ValueError(f"Something went wrong with the retrieval of the setting '{inLab}': "
                         f"The problematic parsed value is '{inVal}'")
    return inVal


def getInputValues(inLabel, sett, simpleLabels, optLabels, multLabels, newLabels, nbItems=0, forGUI=False):
    """
    Find the input from the settings to associate to a given variable

    Parameters
    ----------
    inLabel : str
        Label of the input to get the value to return.
    sett : list
        List of all the content of the settings file.
    simpleLabels : dict
        List of necessary labels (in the keys) with one input, and their data type
    optLabels : dict
        List of optional labels (in the keys) for custom priors (after ###)
    multLabels : dict
        List of necessary labels (in the keys) with multiple inputs (paths)
    newLabs : dict
        List of 'new' labels (not in V1) to take into account for retrocompatibility
    nbItems : int
        Number of input expected if inLabel in multilabels
    forGUI : bool
        Define if the function is called from the GUI or not, to handle errors differently

    Returns
    -------
    inValue : str or list
        Values to put in the variables associated with the input.

    """
    all_labels = {**simpleLabels, **multLabels, **optLabels}
    labels = list(all_labels.keys())
    labels.append("###")  # Alternative stopping condition for the while loop

    optLabel = inLabel in optLabels.keys()

    if inLabel not in sett:
        if optLabel:
            inValue = None  # Missing optional label
        elif inLabel in newLabels.keys():
            inValue = newLabels[inLabel]  # Missing newer label (with defaut value available)
        else:
            error = ValueError(f'"{inLabel}" is missing in the settings file.')
            if forGUI:
                return error
            else:
                raise error
    else:
        simpleLabels.update(optLabels)
        if inLabel in simpleLabels.keys():
            indvalue = sett.index(inLabel) + 1
            if sett[indvalue] not in labels:
                inValue = sett[indvalue]
            else:
                inValue = None  # Case of optional labels (or empty line in general)
        elif inLabel in multLabels.keys():
            inValue = []
            labelInd = sett.index(inLabel)
            lastItemInd = labelInd + 1
            while lastItemInd < len(sett) and sett[lastItemInd] and sett[lastItemInd] not in labels:
                if os.path.exists(sett[lastItemInd]):
                    inValue.append(sett[lastItemInd])
                else:
                    error = FileNotFoundError(f"'{sett[lastItemInd]}' does not exist.")
                    if forGUI:
                        pass  # It's OK to have non existing files in the GUI
                    else:  # It's not when actually running the Functionnectome
                        raise error
                lastItemInd += 1
            if not len(inValue) == nbItems:
                error = ValueError(
                    f"Number of items ({nbItems}) for '{inLabel}' not matching the "
                    f"number of files found in the settings file ({len(inValue)})"
                )
                if forGUI:
                    return error
                else:
                    raise error
        else:
            error = ValueError("Wrong input value when parsing the settings file")
            if forGUI:
                return error
            else:
                raise error

    # Test the data type of the retrieved input
    dtypeLabel = all_labels[inLabel]
    inValue = testInputType(inValue, inLabel, dtypeLabel, optLabel, forGUI)
    return inValue


def readSettings(settingF, forGUI=False):
    """
    Read the settings file and store the settings in a dict that can be used to
    initialize the variables needed in the script.
    Also check if all the intems are present.

    When adding or removing settings in the file, change 'uniqueLabels',
    'multiLabels', and 'optionalLabels' accordingly. There, each element in the
    dict must be a key informing the label to search in the file and the type
    of data it should yield: ('label': type). 'label' should be a string, and 'type'
    should be a type (e.g. int, string, list) or a tuple. If it is a tuple,
    it should contain all the possible allowed items the label should yield.

    When adding new labels, add it to the newLabels variable too, to ensure
    retrocompatibility

    Parameters
    ----------
    settingF : str
        Path to the settings file.

    Raises
    ------
    ValueError
        Settings file does not have all the expected values.

    Returns
    -------
    varDict : dict
        Contains all the settings to be exported into variables in the main function.

    """
    with open(settingF, "r") as f:
        settings = f.read().split("\n")
    settings = list(map(lambda s: s.strip(), settings))
    varDict = {}
    # Necessary items in the settings
    uniqueLabels = {  # With 1 input
        "Output folder:": str,
        "Analysis ('voxel' or 'region'):": (
            'voxel',
            'region',
        ),
        "Number of parallel processes:": int,
        "Priors stored as ('h5' or 'nii'):": (
            'h5',
            'nii',
        ),
        "Position of the subjects ID in their path:": int,
        "Mask the output:": int,
        "Number of subjects:": int,
        "Number of masks:": int,
        "HDF5 priors:": PRIORS_H5
    }
    multiLabels = {  # With multiple inputs
        "Subject's BOLD paths:": list,
        "Masks for voxelwise analysis:": list
    }
    # Optional items for custom priors in the settings (must only have 1 input each at most)
    optionalLabels = {
        "HDF5 path:": str,
        "Template path:": str,
        "Probability maps (voxel) path:": str,
        "Probability maps (region) path:": str,
        "Region masks path:": str
    }
    # Labels recently added, with potential retrocompatibility issues
    # The value of the dict should be the default value to add if the label
    # is absent from the file
    newLabels = {"HDF5 priors:": 'V1.D.WB - Whole brain, Deterministic (legacy)'}

    # Filling the input variables with the settings from the file
    sameInput = {'sett': settings,
                 'simpleLabels': uniqueLabels,
                 'optLabels': optionalLabels,
                 'multLabels': multiLabels,
                 'newLabels': newLabels,
                 'forGUI': forGUI,
                 }
    varDict['results_dir_root'] = getInputValues("Output folder:", **sameInput)
    varDict['anatype'] = getInputValues("Analysis ('voxel' or 'region'):", **sameInput)
    varDict['nb_of_batchs'] = int(getInputValues("Number of parallel processes:", **sameInput))
    varDict['prior_type'] = getInputValues("Priors stored as ('h5' or 'nii'):", **sameInput)
    varDict['priorsH5'] = getInputValues('HDF5 priors:', **sameInput)
    varDict['subIDpos'] = int(getInputValues("Position of the subjects ID in their path:", **sameInput))
    varDict['maskOutput'] = int(getInputValues("Mask the output:", **sameInput))
    varDict['subNb'] = int(getInputValues("Number of subjects:", **sameInput))
    varDict['bold_paths'] = getInputValues("Subject's BOLD paths:", **sameInput, nbItems=varDict['subNb'])
    varDict['mask_nb'] = int(getInputValues("Number of masks:", **sameInput))
    varDict['masks_vox'] = getInputValues(
        "Masks for voxelwise analysis:",
        **sameInput,
        nbItems=varDict['mask_nb']
    )

    # Optional variables at the end of the file, to change the priors paths written here
    if "###" in settings:
        # "###" marks the presence of the optional settings
        varDict['optSett'] = True
        varDict['opt_h5_loc'] = getInputValues("HDF5 path:", **sameInput)
        varDict['opt_template_path'] = getInputValues("Template path:", **sameInput)
        varDict['opt_pmap_vox_loc'] = getInputValues("Probability maps (voxel) path:", **sameInput)
        varDict['opt_pmap_region_loc'] = getInputValues("Probability maps (region) path:", **sameInput)
        varDict['opt_regions_loc'] = getInputValues("Region masks path:", **sameInput)
    else:
        varDict['optSett'] = False
    if forGUI:
        for val in varDict.values():
            if isinstance(val, Exception):
                return val
    return varDict


def updateOldJson(jsonPath, priorsVal):
    '''
    Check if the json file requires updating, and edit it to fit the current way
    of savin the priors paths.
    '''
    if "h5_priors" in priorsVal.keys():
        priorsVal[PRIORS_H5[-1]] = priorsVal["h5_priors"]
        del priorsVal["h5_priors"]
        with open(jsonPath, "w") as jsonP:
            json.dump(priorsVal, jsonP)
    return priorsVal


def getUniqueIDs(bold_paths, subIDpos):
    """
    Generate a unique ID for each file based on the subject ID
    (and the runs if there are multiple files per subject).
    These IDs will be used to create the output folder for each functionnectome file.

    Parameters
    ----------
    bold_paths : list of str
        List of all the inut functional files (full path).
    subIDpos : int
        Index / position of the subject ID in the path.

    Raises
    ------
    ValueError
        When the file paths given are not correct (multiple times the same, or inhomogeneous organisation).

    Returns
    -------
    IDs : List of str
        List of the unique IDs for each file.

    """

    def uniqOrdered(seq):
        seen = set()
        listres = []
        for i in seq:
            if i not in seen:
                listres.append(i)
                seen.add(i)
        return listres

    IDs = []
    list_dPath = []
    for boldf in bold_paths:
        decomposedPath = tuple(
            filter(None, os.path.normpath(boldf).split(os.path.sep))
        )  # cf. the GUI
        list_dPath.append(decomposedPath)
        if (
            subIDpos == len(decomposedPath) - 1
        ) or subIDpos == -1:  # case where subject ID is the BOLD file's name
            subID = (
                decomposedPath[subIDpos].replace(".gz", "").replace(".nii", "")
            )  # Remove extension
            if subID in IDs:
                print(f"'{subID}' already in {IDs}")
                raise ValueError(
                    "Multiple identical filepath or bad ID position given."
                )
        else:
            subID = decomposedPath[subIDpos]
        IDs.append(subID)

    IDs2 = []
    for subID in uniqOrdered(IDs):
        if (
            IDs.count(subID) > 1
        ):  # If there are multiple identical ID (e.g., due to multiple runs by subjects)
            sub_list_dPath = [
                dpath[subIDpos:] for dpath in list_dPath if subID in dpath
            ]
            minLenPath = min([len(subp) for subp in sub_list_dPath])
            subRunPos = 1
            subRuns = [subp[subRunPos] for subp in sub_list_dPath]
            while not len(set(subRuns)) == len(subRuns) and subRunPos + 1 < minLenPath:
                subRunPos += 1
                subRuns = [subp[subRunPos] for subp in sub_list_dPath]

            if (
                len(set(subRuns)) == len(subRuns) and subRunPos + 1 < minLenPath
            ):  # subRuns not last postition in path
                for subRun in subRuns:
                    ID2 = os.path.join(subID, subRun)
                    IDs2.append(ID2)
            elif (
                len(set(subRuns)) == len(subRuns) and subRunPos + 1 == minLenPath
            ):  # The different runs are the files
                for subRun in subRuns:
                    subRun2 = subRun.replace(".gz", "").replace(".nii", "")
                    ID2 = os.path.join(subID, subRun2)
                    IDs2.append(ID2)
            else:
                raise ValueError(
                    f"Problem with the file paths of subject {subID}. "
                    "Probably inhomogeneous organisation of the files tree of the subject."
                )
        else:  # subId is unique, no multiple runs in that subject
            IDs2.append(subID)
    return IDs2


def LogDiplayPercent(logDir, previous_percent=0):
    """
    Check the logs in logDir and display the progress in percent (look at the
    last line of each log).
    """
    logList = glob.glob(os.path.join(logDir, "log_*.txt"))
    if not logList:
        print("Process starting...")
        return
    currentLen = []
    maxETA = 0  # Store the longest ETA  among the parallel processes
    for logf in logList:
        with open(logf, "r") as lf:
            logtxt = lf.readlines()  # Shouldn't be too big
        if not logtxt:
            time.sleep(
                0.5
            )  # In case trying to read at the same time the file is created
            with open(logf, "r") as lf:
                logtxt = lf.readlines()
        lastline = logtxt[-1]
        spiltLastLine = lastline.split(" ")
        procLen = int(spiltLastLine[1])
        currentLen.append(procLen)
        totalLen = int(spiltLastLine[3])  # Should be the same for all the logs
        if len(logtxt) > 1:
            prevStep = -2
            while (
                prevStep > -len(logtxt) and prevStep > -6
            ):  # Compute the ETA using up to 5 step removed
                prevStep -= 1
            prevline = logtxt[prevStep]
            spiltPrevLine = prevline.split(" ")
            timestep = (
                int(spiltLastLine[5]) * 60
                + int(spiltLastLine[8])
                - (int(spiltPrevLine[5]) * 60 + int(spiltPrevLine[8]))
            )
            stepLength = int(spiltLastLine[1]) - int(spiltPrevLine[1])
            if stepLength:
                procETA = round(timestep * (totalLen - procLen) / stepLength)
            else:
                procETA = 0
            if procETA > maxETA:
                maxETA = procETA
    meanProgress = sum(currentLen) / len(currentLen)
    percentProgress = round(100 * meanProgress / totalLen, 3)
    if maxETA and not percentProgress == previous_percent:
        hours = maxETA // 3600
        minutes = maxETA % 3600 // 60
        if minutes + hours > 0:
            ETA = f"{hours}h and {minutes}min"
        else:
            ETA = "< 1 min"
        sys.stdout.write(
            f"\rProgress of the current process: {percentProgress}% ETA: {ETA}      "
        )
        sys.stdout.flush()
    return percentProgress


def init_worker_sumPmaps(templShape, pmapStore, prior, outDir):
    """
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

    """
    global dict_var
    dict_var = {
        "templateShape": templShape,
        "pmapStore": pmapStore,
        "prior_type": prior,
        "outDir": outDir,
    }


def Sum_regionwise_pmaps(regions_batch):
    """
    Function given to the pool of workers in parallel.
    Compute the sum of all regions' probability maps.
    Also checks if all the necessary files are there
    """
    current = multiprocessing.current_process()
    startTime = time.time()
    logFile = os.path.join(dict_var["outDir"], f"log_{current._identity[0]}.txt")
    if dict_var["prior_type"] == "nii":
        for ii, reg in enumerate(regions_batch):
            if not ii % 5:  # Log to follow the progress every few iteration
                ctime = time.time() - startTime
                logtxt = f"Region {ii} in {len(regions_batch)} : {int(ctime//60)} min and {int(ctime%60)} sec\n"
                with open(logFile, "a") as log:
                    log.write(logtxt)
            try:
                region_pmap_img = nib.load(
                    os.path.join(dict_var["pmapStore"], f"{reg}.nii.gz")
                )
            except FileNotFoundError:
                try:
                    region_pmap_img = nib.load(
                        os.path.join(dict_var["pmapStore"], f"{reg}.nii")
                    )
                except FileNotFoundError:
                    logtxt = f"Region {reg} does not have an associated pmap. Closing the process...\n"
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                    return None
            if ii == 0:
                sum_pmap = np.zeros(
                    dict_var["templateShape"], dtype=region_pmap_img.get_data_dtype()
                )
            sum_pmap += region_pmap_img.get_fdata(dtype="float32")
    elif dict_var["prior_type"] == "h5":
        with h5py.File(dict_var["pmapStore"], "r") as h5fout:
            for ii, reg in enumerate(regions_batch):
                if not ii % 5:  # Log to follow the progress every few iteration
                    ctime = time.time() - startTime
                    logtxt = (
                        f"Region {ii} in {len(regions_batch)} : "
                        f"{int(ctime//60)} min and {int(ctime%60)} sec\n"
                    )
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                if ii == 0:
                    sum_pmap = np.zeros(
                        dict_var["templateShape"],
                        dtype=h5fout["tract_region"][reg].dtype,
                    )
                try:
                    sum_pmap += h5fout["tract_region"][reg][:]
                except KeyError:
                    logtxt = f"Region {reg} does not have an associated pmap. Closing the process...\n"
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                    return None
    return sum_pmap


def Sum_voxelwise_pmaps(ind_voxels_batch):
    """
    Function given to the pool of workers in parallel.
    Compute the sum of all selected voxels' probability maps.
    Also checks if all the necessary files are there
    """
    current = multiprocessing.current_process()
    startTime = time.time()
    logFile = os.path.join(dict_var["outDir"], f"log_{current._identity[0]}.txt")
    if dict_var["prior_type"] == "nii":
        for ii, indvox in enumerate(ind_voxels_batch):
            if ii % 100 == 0:
                ctime = time.time() - startTime
                logtxt = (
                    f"Voxel {ii} in {len(ind_voxels_batch)} : "
                    f"{int(ctime//60)} min and {int(ctime%60)} sec\n"
                )
                with open(logFile, "a") as log:
                    log.write(logtxt)

            pmapFound = True
            try:
                mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz"
                vox_pmap_img = nib.load(os.path.join(dict_var["pmapStore"], mapf))
            except FileNotFoundError:
                try:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii"
                    vox_pmap_img = nib.load(os.path.join(dict_var["pmapStore"], mapf))
                except FileNotFoundError:
                    logtxt = (
                        f"Voxel {indvox[0]}_{indvox[1]}_{indvox[2]} does not "
                        "have an associated pmap.\n"
                    )
                    pmapFound = False
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                    # return None
            if ii == 0:
                sum_pmap = np.zeros(dict_var["templateShape"], dtype="float32")
            if pmapFound:
                sum_pmap += vox_pmap_img.get_fdata(dtype="float32")
    elif dict_var["prior_type"] == "h5":
        with h5py.File(dict_var["pmapStore"], "r") as h5fout:
            for ii, indvox in enumerate(ind_voxels_batch):
                if not ii % 100:
                    ctime = time.time() - startTime
                    logtxt = (
                        f"Voxel {ii} in {len(ind_voxels_batch)} : "
                        f"{int(ctime//60)} min and {int(ctime%60)} sec\n"
                    )
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                if ii == 0:
                    sum_pmap = np.zeros(
                        dict_var["templateShape"],
                        dtype='f',
                    )
                try:
                    sum_pmap += h5fout["tract_voxel"][
                        f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"
                    ][:]
                except KeyError:
                    logtxt = (
                        f"Voxel {indvox[0]}_{indvox[1]}_{indvox[2]} does not "
                        "have an associated pmap.\n"
                    )
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                    continue
                    # return None
    return sum_pmap


def init_worker_regionwise(
    shared4D,
    sharedDF,
    outShape,
    boldDFshape,
    regionDF,
    nb_of_batchs,
    pmapStore,
    prior,
    outDir,
):
    """
    Initialize the process of the current pool worker with the variables
    commonly used across the different workers.
    """
    global dict_var
    dict_var = {
        "fun4D": shared4D,
        "funDF": sharedDF,
        "boldShape": outShape,
        "bold_DF_Shape": boldDFshape,
        "regionDF": regionDF,
        "nb_batch": nb_of_batchs,
        "pmapStore": pmapStore,
        "prior_type": prior,
        "outDir": outDir,
    }


def Regionwise_functionnectome(batch_num):
    """
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

    """
    startTime = time.time()

    # Loading the 4D array from shared memory, and selecting the batch
    # Empty, will be filled with the functionnectome data for the current batch
    share4D_np = np.frombuffer(dict_var["fun4D"], "f").reshape(dict_var["boldShape"])
    split_share4D_np = np.array_split(share4D_np, dict_var["nb_batch"], 0)
    current_split_out = split_share4D_np[batch_num]

    # Loading the dataframe from shared memory, and selecting the batch
    # Contains the functional timeseries of each region
    sharedDF_np = np.frombuffer(dict_var["funDF"], "f").reshape(
        dict_var["bold_DF_Shape"]
    )
    sharedDF = pd.DataFrame(sharedDF_np, columns=dict_var["regionDF"])
    split_sharedDF = np.array_split(sharedDF, dict_var["nb_batch"], 0)
    current_split_in = split_sharedDF[batch_num]

    logFile = os.path.join(dict_var["outDir"], f"log_{batch_num}.txt")
    if dict_var["prior_type"] == "nii":
        for ii, reg in enumerate(current_split_in):
            if not ii % 10:  # Write a line in the log every 10 region
                ctime = time.time() - startTime
                logtxt = (
                    f"Region {ii} in {len(list(current_split_in))} : "
                    f"{int(ctime//60)} min and {int(ctime%60)} sec\n"
                )
                with open(logFile, "a") as log:
                    log.write(logtxt)
            # Load proba map of the current region and divide it by the sum of all proba map
            try:
                region_map_img = nib.load(
                    os.path.join(dict_var["pmapStore"], f"{reg}.nii.gz")
                )
            except FileNotFoundError:
                region_map_img = nib.load(
                    os.path.join(dict_var["pmapStore"], f"{reg}.nii")
                )
            region_map = region_map_img.get_fdata(dtype="float32")
            current_shape = (len(current_split_in[reg]), 1, 1, 1)
            current_split_out += np.expand_dims(region_map, 0) * current_split_in[
                reg
            ].values.reshape(current_shape)
    elif dict_var["prior_type"] == "h5":
        with h5py.File(dict_var["pmapStore"], "r") as h5fout:
            for ii, reg in enumerate(current_split_in):
                if not ii % 10:  # Write a line in the log every 10 region
                    ctime = time.time() - startTime
                    logtxt = (
                        f"Region {ii} in {len(list(current_split_in))} : "
                        f"{int(ctime//60)} min and {int(ctime%60)} sec\n"
                    )
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                region_map = h5fout["tract_region"][reg][:]
                current_shape = (len(current_split_in[reg]), 1, 1, 1)
                current_split_out += np.expand_dims(region_map, 0) * current_split_in[
                    reg
                ].values.reshape(current_shape)


def init_worker_voxelwise2(
    shared4Dout,
    reShape,
    nb_of_batchs,
    sharedBold,
    prior,
    indvox_shared,
    gm_mask,
    pmapStore,
    outDir,
    preComput,
):
    """
    Initialize the process of the current pool worker with the variables
    commonly used across the different workers.
    """
    global dict_var
    dict_var = {
        "fun4Dout": shared4Dout,
        "boldReshape": reShape,
        "nb_batch": nb_of_batchs,
        "bold": sharedBold,
        "prior_type": prior,
        "voxel_ind": indvox_shared,
        "GM mask": gm_mask,
        "pmapStore": pmapStore,
        "outDir": outDir,
        "preComput": preComput,
    }


def Voxelwise_functionnectome2(batch_num):
    """
    Iterating over white matter voxels instead of grey matter voxels.
    Parallelisation over the voxels and not the time.
    Reduce I/O operations and replace for loops with dot product

    Data from shared4Dout_np must be masked beforehand
    selecting only the voxels in the mask for both 4D and pmap

    Parameters
    ----------
    batch_num : int
        Number of the current batch being processed.

    Returns
    -------
    None. The results are in shared memory (in "fun_4D_shared" in the main process)

    """
    # os.sched_setaffinity(0, {batch_num})
    startTime = time.time()
    # nbTR = dict_var['boldReshape'][0]  # Number of volumes (TR) in the 4D data
    nbTR = dict_var["boldReshape"][-1]  # Number of volumes (TR) in the 4D data

    # Loading the 4D output data from shared memory and selecting the part to
    # fill in the current process (should be empty)
    shared4Dout_np = np.frombuffer(dict_var["fun4Dout"], "f").reshape(
        dict_var["boldReshape"]
    )

    # Load the voxels' index of the template and select the vexels for the process
    shared_vox_ind_np = np.frombuffer(dict_var["voxel_ind"], "i").reshape((-1, 3))
    batch_voxels_list = np.array_split(shared_vox_ind_np, dict_var["nb_batch"], 0)
    batch_vox = batch_voxels_list[batch_num]
    mask = dict_var[
        "GM mask"
    ]  # MAkes a copy for each process. But it's only a boolean array

    # Loading the input 2D array (temporal x flattened spatial) from shared memory
    shared_2D_bold_np = np.frombuffer(dict_var["bold"], "f").reshape((nbTR, -1))

    logFile = os.path.join(dict_var["outDir"], f"log_{batch_num}.txt")

    if dict_var["preComput"]:  # Sum of pmaps was done previously in a different step
        if dict_var["prior_type"] == "h5":
            with h5py.File(dict_var["pmapStore"], "r") as h5fout:
                for ii, indvox in enumerate(batch_vox):
                    if ii % 100 == 0:  # Check the progress every 10 steps
                        ctime = time.time() - startTime
                        logtxt = (
                            f"Voxel {ii} in {len(batch_vox)} : {int(ctime//60)} min and {int(ctime%60)} sec\n"
                        )
                        with open(logFile, "a") as log:
                            log.write(logtxt)
                    try:
                        vox_pmap = h5fout["tract_voxel"][
                            f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"
                        ][:]
                        shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = np.dot(
                            shared_2D_bold_np, vox_pmap[mask]
                        )
                    except KeyError:
                        shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = np.zeros(nbTR, dtype='f')
        elif dict_var["prior_type"] == "nii":
            for ii, indvox in enumerate(batch_vox):
                if ii % 100 == 0:  # Check the progress every 100 steps
                    ctime = time.time() - startTime
                    logtxt = f"Voxel {ii} in {len(batch_vox)} : {int(ctime//60)} min and {int(ctime%60)} sec\n"
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                # Load the probability map of the current voxel, combine it with the functional signal,
                # and add it to the results
                pmapFound = True
                try:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz"
                    vox_pmap_img = nib.load(os.path.join(dict_var["pmapStore"], mapf))
                except FileNotFoundError:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii"
                    vox_pmap_img = nib.load(os.path.join(dict_var["pmapStore"], mapf))
                except FileNotFoundError:
                    pmapFound = False
                if pmapFound:
                    vox_pmap = vox_pmap_img.get_fdata(dtype="float32")
                    shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = np.dot(
                        shared_2D_bold_np, vox_pmap[mask]
                    )
                else:
                    shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = np.zeros(
                        nbTR, dtype="f"
                    )
    else:  # sum of pmaps was not done beforehand, so it will be done here and directly applied
        if dict_var["prior_type"] == "h5":
            with h5py.File(dict_var["pmapStore"], "r") as h5fout:
                for ii, indvox in enumerate(batch_vox):
                    if ii % 100 == 0:  # Check the progress every 10 steps
                        ctime = time.time() - startTime
                        logtxt = (
                            f"Voxel {ii} in {len(batch_vox)} : {int(ctime//60)} min and {int(ctime%60)} sec\n"
                        )
                        with open(logFile, "a") as log:
                            log.write(logtxt)
                    vox_pmap = h5fout["tract_voxel"][
                        f"{indvox[0]}_{indvox[1]}_{indvox[2]}_vox"
                    ][:]
                    selected_p = vox_pmap[mask]
                    sum_p = selected_p.sum()  # For the normalizing step
                    if sum_p:
                        shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = (
                            np.dot(shared_2D_bold_np, selected_p) / sum_p
                        )
                    else:
                        shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = np.zeros(
                            nbTR, dtype="f"
                        )
        elif dict_var["prior_type"] == "nii":
            for ii, indvox in enumerate(batch_vox):
                if ii % 100 == 0:  # Check the progress every 100 steps
                    ctime = time.time() - startTime
                    logtxt = f"Voxel {ii} in {len(batch_vox)} : {int(ctime//60)} min and {int(ctime%60)} sec\n"
                    with open(logFile, "a") as log:
                        log.write(logtxt)
                # Load the probability map of the current voxel, combine it with the functional signal,
                # and add it to the results
                try:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii.gz"
                    vox_pmap_img = nib.load(os.path.join(dict_var["pmapStore"], mapf))
                except FileNotFoundError:
                    mapf = f"probaMaps_{indvox[0]}_{indvox[1]}_{indvox[2]}_vox.nii"
                    vox_pmap_img = nib.load(os.path.join(dict_var["pmapStore"], mapf))
                vox_pmap = vox_pmap_img.get_fdata(dtype="float32")
                selected_p = vox_pmap[mask]
                sum_p = selected_p.sum()  # For the normalizing step
                if sum_p:
                    shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = (
                        np.dot(shared_2D_bold_np, selected_p) / sum_p
                    )
                else:  # Maybe I could just skip this instead of inputing zeros?
                    shared4Dout_np[indvox[0], indvox[1], indvox[2], :] = np.zeros(
                        nbTR, dtype="f"
                    )


# %%
def run_functionnectome(settingFilePath, from_GUI=False):
    """
    Main functionction. Run the computation and call the other functions.

    Parameters
    ----------
    settingFilePath : string
        Path to the settings file (.fcntm)
    """
    print("Process starting...")
    st = time.time()

    print("Loading settings")
    # Read the setting file given to set the input variables
    settingsVar = readSettings(settingFilePath)

    results_dir_root = settingsVar['results_dir_root']
    anatype = settingsVar['anatype']
    nb_of_batchs = settingsVar['nb_of_batchs']
    prior_type = settingsVar['prior_type']
    priorsH5 = settingsVar['priorsH5']
    subIDpos = settingsVar['subIDpos']
    maskOutput = settingsVar['maskOutput']
    subNb = settingsVar['subNb']
    bold_paths = settingsVar['bold_paths']
    if anatype == "voxel":
        mask_nb = settingsVar['mask_nb']
        masks_vox = settingsVar['masks_vox']

    optSett = settingsVar['optSett']
    if optSett:
        opt_h5_loc = settingsVar['opt_h5_loc']
        opt_template_path = settingsVar['opt_template_path']
        opt_pmap_vox_loc = settingsVar['opt_pmap_vox_loc']
        opt_pmap_region_loc = settingsVar['opt_pmap_region_loc']
        opt_regions_loc = settingsVar['opt_regions_loc']

    # %% Checking for the existence of priors and asking what to do if none found
    if not optSett:  # If non default priors are used, override this
        pkgPath = os.path.dirname(__file__)
        jsonPath = os.path.join(pkgPath, "priors_paths.json")
        if os.path.exists(jsonPath):
            newJson = False
            with open(jsonPath, "r") as jsonP:
                priors_paths = json.load(jsonP)
            priors_paths = updateOldJson(jsonPath, priors_paths)
        else:  # Create a new dict to store the filepaths, filled below
            newJson = True
            priors_paths = {
                "template": "",
                "regions": "",
                "region_pmap": "",
                "voxel_pmap": "",
            }

        if prior_type == "h5":
            if priorsH5 not in priors_paths.keys() or not os.path.exists(priors_paths[priorsH5]):
                # Missing h5 priors, so downloading required
                newJson = True
                h5P = ""
                if from_GUI:
                    ask_h5 = Ask_hdf5_path(priorsH5, priors_paths)
                    ask_h5.mainloop()
                    h5P = ask_h5.prior_path_h5
                    priors_paths = ask_h5.dictH5
                else:
                    askDL = input(
                        "No HDF5 priors file was found. To download it, type 'D' and Enter. "
                        "If you already downloaded it before, type 'S' and Enter, "
                        "then provide the path to the file.\n"
                        "D : Download / S : Select\n"
                    )
                    if askDL.upper().strip() == "D":
                        prior_dirpath_h5 = input(
                            "Type (or paste) the path to the folder "
                            "where you want to save the priors:\n"
                        )
                        prior_dirpath_h5 = prior_dirpath_h5.strip()
                        prior_dirpath_h5 = prior_dirpath_h5.replace(
                            "\\ ", " "
                        )  # If pasted from a file explorer
                        if prior_dirpath_h5 and os.path.exists(prior_dirpath_h5):
                            askDLall = input(
                                "Do you wish to download the whole suit of priors "
                                "(for later use) or only the current priors? To download "
                                "all the priors, type 'A' and Enter. "
                                "To download only the current priors, type 'S' and Enter.\n"
                                "A : All / O : Only current\n"
                            )
                            if askDLall.upper().strip() in ("O", "A"):
                                h5P = Download_H5(prior_dirpath_h5, priorsH5)
                                if askDLall.upper().strip() == "A" and os.path.exists(h5P):
                                    priors_paths = DL_missingH5(priors_paths, prior_dirpath_h5, h5P)
                            else:
                                print("Wrong entry (neither A nor O). Canceled...")
                    elif askDL.upper().strip() == "S":
                        h5P = input(
                            "Type (or paste) the path to the HDF5 priors file:\n"
                        )
                        h5P = h5P.strip()
                        h5P = h5P.replace("\\ ", " ")  # If pasted from a file explorer
                    else:
                        print("Wrong entry (neither D nor S). Canceled...")

                if os.path.exists(h5P):
                    priors_paths[priorsH5] = h5P
                    print(f"Selected file for HDF5 priors : {h5P}")
                else:
                    raise Exception(
                        "No correct path to the priors file was provided (or the dowloading failed). "
                        "Stopping the program."
                    )

        if prior_type == "nii":
            if not (
                priors_paths["template"] and os.path.exists(priors_paths["template"])
            ):
                # Missing template in priors
                newJson = True
                templP = ""
                if from_GUI:
                    root = tk.Tk()
                    root.withdraw()
                    canc = messagebox.askokcancel(
                        "No template found", "No brain template found. Select the file?"
                    )
                    templP = ""
                    if canc:
                        home = os.path.expanduser("~")
                        templP = filedialog.askopenfilename(
                            parent=root,
                            initialdir=home,
                            title="Select the brain template",
                            filetypes=[("NIfTI", ".nii .gz")],
                        )
                    root.destroy()
                else:
                    templP = input(
                        "No brain template found.\nType (or paste) the path to the file:\n"
                    )
                    templP = templP.strip()
                    templP = templP.replace(
                        "\\ ", " "
                    )  # If pasted from a file explorer
                if os.path.exists(templP):
                    priors_paths["template"] = templP
                    print(f"Selected file for brain template : {templP}")
                else:
                    raise Exception(
                        "Using NifTI priors, but no brain template was provided. Stopping the program."
                    )

            if anatype == "region":
                if not (
                    priors_paths["regions"] and os.path.exists(priors_paths["regions"])
                ):
                    # missing folder containing the masks of each brain region
                    newJson = True
                    regP = ""
                    if from_GUI:
                        root = tk.Tk()
                        root.withdraw()
                        canc = messagebox.askokcancel(
                            "Brain regions masks not found",
                            "The folder containing the brain regions was not found.\n"
                            "Select the folder?\n"
                            "/!\\ Don't forget to open i.e. double-click on) "
                            "the last selected folder",
                        )
                        regP = ""
                        if canc:
                            home = os.path.expanduser("~")
                            regP = filedialog.askdirectory(
                                parent=root,
                                initialdir=home,
                                title="Select the regions folder",
                            )
                        root.destroy()
                    else:
                        regP = input(
                            "The folder containing the brain regions was not found."
                            "\nType (or paste) the path to the folder:\n"
                        )
                        regP = regP.strip()
                        regP = regP.replace(
                            "\\ ", " "
                        )  # If pasted from a file explorer

                    if os.path.exists(regP):
                        priors_paths["regions"] = regP
                        print(f"Selected folder for region masks : {regP}")
                    else:
                        raise Exception(
                            "Using NifTI priors, but no brain regions were provided. Stopping the program."
                        )

                if not (
                    priors_paths["region_pmap"]
                    and os.path.exists(priors_paths["region_pmap"])
                ):
                    # missing folder containing the proba maps of each brain region
                    newJson = True
                    pmapsP = ""
                    if from_GUI:
                        root = tk.Tk()
                        root.withdraw()
                        canc = messagebox.askokcancel(
                            "Brain regions probability maps not found",
                            "The folder containing the brain regions "
                            "probability maps was not found.\n"
                            "Select the folder?\n"
                            "/!\\ Don't forget to open (i.e. double-click on) "
                            "the last selected folder",
                        )
                        pmapsP = ""
                        if canc:
                            home = os.path.expanduser("~")
                            pmapsP = filedialog.askdirectory(
                                parent=root,
                                initialdir=home,
                                title="Select the regions probability maps folder",
                            )
                        root.destroy()
                    else:
                        pmapsP = input(
                            "The folder containing the brain regions probability maps "
                            "was not found.\nType (or paste) the path to the folder:\n"
                        )
                        pmapsP = pmapsP.strip()
                        pmapsP = pmapsP.replace(
                            "\\ ", " "
                        )  # If pasted from a file explorer

                    if os.path.exists(pmapsP):
                        priors_paths["region_pmap"] = pmapsP
                        print(f"Selected folder for region proba maps : {regP}")
                    else:
                        raise Exception(
                            "Using NifTI priors, but no brain regions "
                            "probability maps were provided. Stopping the program."
                        )

            if anatype == "voxel":
                if not (
                    priors_paths["voxel_pmap"]
                    and os.path.exists(priors_paths["voxel_pmap"])
                ):
                    # missing folder containing the proba maps of each brain voxels
                    newJson = True
                    pmapsP = ""
                    if from_GUI:
                        root = tk.Tk()
                        root.withdraw()
                        canc = messagebox.askokcancel(
                            "Brain voxels probability maps not found",
                            "The folder containing the brain voxels "
                            "probability maps was not found. Select the folder?",
                        )
                        pmapsP = ""
                        if canc:
                            home = os.path.expanduser("~")
                            pmapsP = filedialog.askdirectory(
                                parent=root,
                                initialdir=home,
                                title="Select the voxels probability maps folder",
                            )
                        root.destroy()
                    else:
                        pmapsP = input(
                            "The folder containing the brain voxels probability maps "
                            "was not found.\nType (or paste) the path to the folder:\n"
                        )
                        pmapsP = pmapsP.strip()
                        pmapsP = pmapsP.replace(
                            "\\ ", " "
                        )  # If pasted from a file explorer

                    if os.path.exists(pmapsP):
                        priors_paths["voxel_pmap"] = pmapsP
                    else:
                        raise Exception(
                            "Using NifTI priors with voxelwise analysis, "
                            " but no folder containing the probability maps was provided"
                        )
        if newJson:
            with open(jsonPath, "w") as jsonP:
                json.dump(priors_paths, jsonP)

    # %% Association of the priors path with their variables
    if prior_type == "nii":
        if optSett:
            if not (
                opt_template_path
                and (opt_pmap_vox_loc or (opt_pmap_region_loc and opt_regions_loc))
            ):
                raise Exception("Using optional settings, but some are not filled.")
            template_path = opt_template_path
            if anatype == "region":
                pmap_loc = opt_pmap_region_loc
                regions_loc = opt_regions_loc
            elif anatype == "voxel":
                pmap_loc = opt_pmap_vox_loc
            else:
                raise Exception('Bad type of analysis (not "voxel" nor "region")')
        else:
            template_path = priors_paths["template"]
            if anatype == "region":
                pmap_loc = priors_paths["region_pmap"]
                regions_loc = priors_paths["regions"]
            elif anatype == "voxel":
                pmap_loc = priors_paths["voxel_pmap"]
            else:
                raise Exception('Bad type of analysis (not "voxel" nor "region")')
    elif prior_type == "h5":
        if optSett:
            if opt_h5_loc:
                h5_loc = opt_h5_loc
            else:
                raise Exception("Using custum H5 priors, but no path given.")
        else:
            h5_loc = priors_paths[priorsH5]
        pmap_loc = h5_loc
        regions_loc = h5_loc
    else:
        raise Exception('Bad type of priors (not "nii" nor "h5")')

    Path(results_dir_root).mkdir(
        parents=True, exist_ok=True
    )  # Create results_dir_root if needed
    if (
        not os.path.dirname(settingFilePath) == results_dir_root
    ):  # If the setting file is imported from somewhere else
        n = 1
        fpath = os.path.join(results_dir_root, "settings.fcntm")  # Default name
        while os.path.exists(fpath):
            fpath = os.path.join(results_dir_root, f"settings{n}.fcntm")
            n += 1
        shutil.copyfile(
            settingFilePath, fpath
        )  # Save the settings into the result directory

    # Get the basic info about the input (shape, file header, list of regions, ...)
    if prior_type == "nii":
        # MNI template
        template_img = nib.load(template_path)
        template_vol = template_img.get_fdata().astype(bool)  # binarized
        if anatype == "region":
            listRegionsFiles = glob.glob(os.path.join(regions_loc, "*.nii*"))
            listRegions = [
                os.path.basename(reg).rsplit(".", 2)[0] for reg in listRegionsFiles
            ]
            listRegions.sort()
            regions_batchs = np.array_split(listRegions, nb_of_batchs)
            base_region_img = nib.load(listRegionsFiles[0])
            affine3D = base_region_img.affine
            header3D = base_region_img.header

        elif anatype == "voxel":
            listVoxFiles = glob.glob(os.path.join(pmap_loc, "*.nii*"))
            base_vox_img = nib.load(listVoxFiles[0])
            affine3D = base_vox_img.affine
            header3D = base_vox_img.header
            listVoxFiles = None
    elif prior_type == "h5":
        with h5py.File(pmap_loc, "r") as h5fout:
            template_vol = h5fout["template"][:]
            template_vol = template_vol.astype(bool)

            if anatype == "region":
                # Check if there are regionwise prirs in the H5
                if "tract_region" not in h5fout or \
                        "mask_region" not in h5fout or \
                        len(h5fout["tract_region"]) == 1:
                    raise KeyError("Sorry, it seems the current priors do not "
                                   "have regionwise maps available. If you really need them, "
                                   "do not hesitate and concact me by email or via Github.")
                hdr = h5fout["tract_region"].attrs["header"]
                hdr3D = eval(hdr)
                header3D = nib.Nifti1Header()
                for key in header3D.keys():
                    header3D[key] = hdr3D[key]
                affine3D = header3D.get_sform()
                listRegions = list(h5fout["tract_region"].keys())
                listRegions.sort()  # Should already be sorted anyways
                regions_batchs = np.array_split(listRegions, nb_of_batchs)
            elif anatype == "voxel":
                hdr = h5fout["tract_voxel"].attrs["header"]
                hdr3D = eval(hdr)
                header3D = nib.Nifti1Header()
                for key in header3D.keys():
                    header3D[key] = hdr3D[key]
                affine3D = header3D.get_sform()

    # Retrieves all subjects' unique IDs from their path
    IDs = getUniqueIDs(bold_paths, subIDpos)

    # %% Start the loop over all the input files (i.e. "subjects")
    for isub, subID, boldf in zip(range(len(bold_paths)), IDs, bold_paths):
        print(f"Processing subject {subID} in {anatype}wise analysis")
        results_dir = os.path.join(results_dir_root, anatype + "wise_analysis", subID)
        finalOutPath = os.path.join(results_dir, "functionnectome.nii.gz")
        if os.path.exists(finalOutPath):
            print(f"Output file already exists for subject {subID}. Skiping...")
            continue
        Path(results_dir).mkdir(
            parents=True, exist_ok=True
        )  # Create results_dir if needed

        # If there are some old and older log files (there shouldn't though...)
        old_logs = glob.glob(os.path.join(results_dir, "log_*.txt"))
        very_old_logs = glob.glob(os.path.join(results_dir, "old_log_*.txt"))
        for veryoldlog in very_old_logs:  # Delete the older logs
            os.remove(veryoldlog)
        for oldlog in old_logs:  # And save (for now) the old (but more recent) logs
            logname = "old_" + os.path.basename(oldlog)
            os.rename(oldlog, os.path.join(results_dir, logname))

        # %% Run the regionwise analysis
        if anatype == "region":
            # Launching parallel processing for the sum all the regions' probability maps
            # for later normalization
            sumpath = os.path.join(results_dir_root, "sum_probaMaps_regions.nii.gz")
            if not os.path.exists(sumpath):
                with multiprocessing.Pool(
                    processes=nb_of_batchs,
                    initializer=init_worker_sumPmaps,
                    initargs=(template_vol.shape, pmap_loc, prior_type, results_dir),
                ) as pool:
                    print("Launching parallel computation: Sum of probability maps")
                    out_batch_sum = pool.map_async(Sum_regionwise_pmaps, regions_batchs)
                    # Diplay the progress
                    percent = None  # to keep track of the previous
                    while not out_batch_sum.ready():
                        percent = LogDiplayPercent(results_dir, percent)
                        time.sleep(1)
                    sys.stdout.write(
                        "\rProgress of the current process: 100%  Completed           \n"
                    )
                    sys.stdout.flush()
                    out_batch_sum = out_batch_sum.get()
                    if any(
                        [a is None for a in out_batch_sum]
                    ):  # There was a problem in a process
                        raise Exception(
                            "One of the process did not yield a good result. Check the logs."
                        )
                    logfiles = glob.glob(os.path.join(results_dir, "log_*.txt"))
                    for logf in logfiles:
                        os.remove(logf)

                print("Multiprocessing done. Sum & save now.")
                sum_pmap_all = np.sum(out_batch_sum, 0)
                out_batch_sum = None  # Release the RAM
                sum_pmap_img = nib.Nifti1Image(sum_pmap_all, affine3D, header3D)
                nib.save(sum_pmap_img, sumpath)
            else:
                alreadyThereMsg = (
                    "Sum of probability maps already computed previously. Reloading it.\n"
                    "WARNING: If you changed the regions for the analysis, "
                    'stop the process, delete the "sum_probaMaps_regions.nii.gz" file '
                    "and relaunch the analysis."
                )
                print(alreadyThereMsg)
                sum_pmap_img = nib.load(sumpath)
                sum_pmap_all = sum_pmap_img.get_fdata(dtype="float32")

            # Loading the 4D BOLD file
            print("Loading 4D data (as float32).")
            bold_img = nib.load(boldf)
            bold_header = bold_img.header
            bold_affine = bold_img.affine
            # Checking if the data has proper dimension and orientation
            flipLR = False
            if not (bold_affine == affine3D).all():
                if (bold_affine[0] == -affine3D[0]).all():
                    warnMsgOrient = (
                        "The orientation of the input 4D volume seems to be in RAS orientation, i.e."
                        " left/right flipped compared to the orientation of the anatomical priors "
                        "(which should be the MNI_152 space with LAS orientation).\nThe 4D volume "
                        "will be flipped to LAS during the processing but the output will be flipped back "
                        "to the original 4D orientation (RAS)."
                    )
                    warnings.warn(warnMsgOrient)
                    flipLR = True
                    print("Orientation of the input 4D volume:")
                    print(bold_affine.astype(int))
                    print("\nOrientation of the priors:")
                    print(affine3D.astype(int))
                else:
                    print(
                        "The orientation of the input 4D volume in not the same as "
                        "orientation of the anatomical priors (which should be the "
                        "MNI_152 2mm space):\n"
                        "Anatomical white matter prior's orientation:"
                    )
                    print(affine3D.astype(int))
                    print("\nInput 4D volume's orientation:")
                    print(bold_affine.astype(int))
                    raise ValueError("Wrong data orientation, or not in MNI152 space.")
            bold_vol = bold_img.get_fdata(dtype="float32")
            if bold_img.ndim == 3:
                warnings.warn(
                    'Input volume is 3D, presumably a statistical map.\nThe Functionnectome is meant to be used with '
                    '4D volumes, before GLM or other signal analysis. Using 3D maps as input may result in unreliable '
                    'values in the projection.'
                )
                bold_header = None
                bold_vol = np.expand_dims(bold_vol, -1)
            bold_shape = bold_vol.shape
            if flipLR:
                bold_vol = np.flip(bold_vol, 0)

            # Computing the DataFrame containing the median timeseries of all the regions
            print("Computation of BOLD median for each region")
            regions_BOLD_median = pd.DataFrame(columns=listRegions)
            if prior_type == "nii":
                for reg in listRegions:
                    try:
                        region_img = nib.load(
                            os.path.join(regions_loc, f"{reg}.nii.gz")
                        )
                    except FileNotFoundError:
                        region_img = nib.load(os.path.join(regions_loc, f"{reg}.nii"))
                    try:
                        region_vol = region_img.get_fdata(
                            dtype=region_img.get_data_dtype()
                        )
                    except ValueError:
                        region_vol = region_img.get_fdata().astype(
                            region_img.get_data_dtype()
                        )
                    region_vol *= template_vol  # masking
                    if region_vol.sum():  # if region not empty
                        region_BOLD = bold_vol[np.where(region_vol)]
                        regions_BOLD_median[reg] = np.median(region_BOLD, 0)
            elif prior_type == "h5":
                with h5py.File(pmap_loc, "r") as h5fout:
                    grp_reg = h5fout["mask_region"]
                    for reg in listRegions:
                        region_vol = grp_reg[reg][:]
                        region_vol *= template_vol  # masking
                        if region_vol.sum():  # if region not empty
                            region_BOLD = bold_vol[np.where(region_vol)]
                            regions_BOLD_median[reg] = np.median(region_BOLD, 0)

            # Release the RAM
            bold_vol = bold_img = None

            # Launching parallel processing for the functionnectome computation proper

            # Create a shared RawArray containing the data from the BOLD regionwise DataFrame
            boldDFshape = regions_BOLD_median.shape
            bold_DF_shared = multiprocessing.RawArray("f", int(np.prod(boldDFshape)))
            # Manipulate the RawArray as a numpy array
            bold_DF_shared_np = np.frombuffer(bold_DF_shared, "f").reshape(boldDFshape)
            np.copyto(
                bold_DF_shared_np, regions_BOLD_median.values
            )  # Filling the RawArray
            regions_BOLD_median = None

            # Create a shared RawArray that will contain the results
            # Puting the time dim first (for contiguous data array,
            # necessary to avoid copy with shared memory access)
            bold_reshape = (bold_shape[-1], *bold_shape[:-1])
            fun_4D_shared = multiprocessing.RawArray("f", int(np.prod(bold_reshape)))

            with multiprocessing.Pool(
                processes=nb_of_batchs,
                initializer=init_worker_regionwise,
                initargs=(
                    fun_4D_shared,
                    bold_DF_shared,
                    bold_reshape,
                    boldDFshape,
                    listRegions,
                    nb_of_batchs,
                    pmap_loc,
                    prior_type,
                    results_dir,
                ),
            ) as pool:
                poolCheck = pool.map_async(
                    Regionwise_functionnectome, range(nb_of_batchs)
                )
                # Diplay the progress
                percent = None  # to keep track of the preivous
                while not poolCheck.ready():
                    percent = LogDiplayPercent(results_dir, percent)
                    time.sleep(1)
                poolCheck.get()
                sys.stdout.write(
                    "\rProgress of the current process: 100%  Completed           \n"
                )
                sys.stdout.flush()
                logfiles = glob.glob(os.path.join(results_dir, "log_*.txt"))
                for logf in logfiles:
                    os.remove(logf)
            print(
                "Multiprocessing done. Application of the proportionality and saving results."
            )
            sum_pmap4D_all = np.frombuffer(fun_4D_shared, "f").reshape(bold_reshape)
            # Applying proportionality
            sum_pmap4D_all = np.divide(
                sum_pmap4D_all,
                np.expand_dims(sum_pmap_all, 0),
                out=sum_pmap4D_all,
                where=np.expand_dims(sum_pmap_all, 0) != 0,
            )
            sum_pmap4D_all = np.moveaxis(sum_pmap4D_all, 0, -1)
            # Masking the output with the template
            if maskOutput:
                sum_pmap4D_all *= np.expand_dims(template_vol, -1)
            # Saving the results
            if flipLR:
                sum_pmap4D_all = np.flip(sum_pmap4D_all, 0)
            sum_pmap4D_img = nib.Nifti1Image(sum_pmap4D_all, bold_affine, bold_header)
            nib.save(sum_pmap4D_img, finalOutPath)
            time.sleep(5)  # Waiting a bit, just in case...
            sum_pmap4D_img = sum_pmap4D_all = None
        # %% Run the voxelwise analysis
        elif anatype == "voxel":
            # Loading subject's mask, restriction to voxels inside the template,
            # and getting the list of voxels' indexes
            if len(masks_vox) > 1:
                mask_path = masks_vox[isub]
            elif len(masks_vox) == 1:
                mask_path = masks_vox[0]
            else:
                mask_path = ""

            if mask_path:
                voxel_mask_img = nib.load(mask_path)
                mask_affine = voxel_mask_img.affine
                voxel_mask = voxel_mask_img.get_fdata().astype(bool)
                if not (mask_affine == affine3D).all():  # Check the orientation of the mask
                    if (mask_affine[0] == -affine3D[0]).all():
                        voxel_mask = np.flip(voxel_mask, 0)
                    else:
                        raise ValueError("Wrong mask orientation, or not in MNI152 space (2x2x2 mm3).")
                voxel_mask *= template_vol
            else:
                print("No mask given => using all voxels... Not a super good idea.")
                voxel_mask = template_vol
            ind_mask1 = np.nonzero(voxel_mask)
            ind_mask2 = np.transpose(ind_mask1).astype("i")
            voxel_mask_img = None

            # Division of arg_voxels in batches used by the grid
            split_ind = np.array_split(ind_mask2, nb_of_batchs)

            # Summing all the proba maps, for later "normalization"
            if mask_nb == 1 and subNb > 1:
                precomp = True  # Stores wether there is a precomputing of the sum of probamaps or not
                firstSubRes = os.path.join(
                    results_dir_root, anatype + "wise_analysis", IDs[0]
                )
                sumpath = os.path.join(firstSubRes, "sum_probaMaps_voxel.nii.gz")
            else:
                precomp = False

            if precomp and not os.path.exists(sumpath):
                print("Launching parallel computation: Sum of Probability maps")
                with multiprocessing.Pool(
                    processes=nb_of_batchs,
                    initializer=init_worker_sumPmaps,
                    initargs=(template_vol.shape, pmap_loc, prior_type, results_dir),
                ) as pool:
                    out_batch_sum = pool.map_async(Sum_voxelwise_pmaps, split_ind)
                    # Diplay the progress
                    percent = None  # to keep track of the preivous
                    while not out_batch_sum.ready():
                        percent = LogDiplayPercent(results_dir, percent)
                        time.sleep(1)
                    sys.stdout.write(
                        "\rProgress of the current process: 100%  Completed           \n"
                    )
                    sys.stdout.flush()
                    out_batch_sum = out_batch_sum.get()
                    logfiles = glob.glob(os.path.join(results_dir, "log_*.txt"))
                    for logf in logfiles:
                        os.remove(logf)
                sum_pmap_all = np.sum(out_batch_sum, 0)
                out_batch_sum = None  # Release the RAM
                sum_pmap_img = nib.Nifti1Image(sum_pmap_all, affine3D, header3D)
                nib.save(sum_pmap_img, sumpath)
            elif precomp and os.path.exists(sumpath):
                alreadyThereMsg = (
                    "Sum of probability maps already computed previously. Reloading it.\n"
                    "WARNING: If you changed the mask for the analysis, "
                    'stop the process, delete the "sum_probaMaps_voxel.nii.gz" file '
                    "and relaunch the analysis."
                )
                # print(alreadyThereMsg)
                warnings.warn(alreadyThereMsg)
                sum_pmap_img = nib.load(sumpath)
                try:
                    sum_pmap_all = sum_pmap_img.get_fdata(
                        dtype=sum_pmap_img.get_data_dtype()
                    )
                except ValueError:
                    sum_pmap_all = sum_pmap_img.get_fdata().astype(
                        sum_pmap_img.get_data_dtype()
                    )

            # Loading the 4D BOLD file
            print("Loading 4D data (as float32).")
            bold_img = nib.load(boldf)
            bold_header = bold_img.header
            bold_affine = bold_img.affine
            # Checking if the data has proper dimension and orientation
            flipLR = False
            if not (bold_affine == affine3D).all():
                if (bold_affine[0] == -affine3D[0]).all():
                    warnMsgOrient = (
                        "The orientation of the input 4D volume seems to be left/right flipped "
                        "compared to the orientation of the anatomical priors (which should be "
                        "the MNI_152 orientation).\nThe 4D volume will be flipped during the processing "
                        "but the output will be flipped back to the original 4D orientation."
                    )
                    warnings.warn(warnMsgOrient)
                    flipLR = True
                    print("Orientation of the input 4D volume:")
                    print(bold_affine.astype(int))
                    print("orientation of the priors:")
                    print(affine3D.astype(int))
                else:
                    print(
                        "The orientation of the input 4D volume in not "
                        "the same as orientation of the anatomical "
                        "priors (which should be the MNI_152 orientation):\n"
                        "Anatomical white matter prior's orientation:"
                    )
                    print(affine3D.astype(int))
                    print("Input 4D volume's orientation:")
                    print(bold_affine.astype(int))
                    raise ValueError("Wrong data orientation, or not in MNI152 space.")
            bold_vol = bold_img.get_fdata(dtype="float32")
            if bold_img.ndim == 3:
                # raise ValueError(
                #     "The input NIfTI volume is 3D. The Functionnectome only accepts 4D volumes."
                # )
                warnings.warn(
                    'Input volume is 3D, presumably a statistical map.\nThe Functionnectome is meant to be used with '
                    '4D volumes, before GLM or other signal analysis. Using 3D maps as input may result in unreliable '
                    'values in the projection.'
                )
                bold_header = None
                bold_vol = np.expand_dims(bold_vol, -1)
            if flipLR:
                bold_vol = np.flip(bold_vol, 0)

            # Creating shared memory variables accessed by the parrallel processes
            bold_shape = bold_vol.shape
            # Select the voxels from the mask => makes a 2D array (flattened sptial x temporal)
            bold_vol_2D = bold_vol[ind_mask1]
            # Release the RAM
            bold_vol = bold_img = ind_mask1 = None
            bold_2D_shared = multiprocessing.RawArray(
                "f", int(np.prod(bold_vol_2D.shape))
            )
            # Manipulate the RawArray as a numpy array
            bold_2D_shared_np = np.frombuffer(bold_2D_shared, "f").reshape(
                bold_vol_2D.T.shape
            )
            np.copyto(bold_2D_shared_np, bold_vol_2D.T)  # Filling the RawArray
            bold_vol_2D = None

            # Create a shared RawArray containing the index of each used voxel
            # ind_mask_shared = multiprocessing.RawArray('i', int(np.prod(ind_mask2.shape)))
            # ind_mask_shared_np = np.frombuffer(ind_mask_shared, 'i').reshape(ind_mask2.shape)
            # np.copyto(ind_mask_shared_np, ind_mask2)
            # ind_mask2 = None

            # Create a shared RawArray containing the index of each used voxel
            ind_template = np.argwhere(template_vol).astype("uint16")
            ind_template_shared = multiprocessing.RawArray(
                "i", int(np.prod(ind_template.shape))
            )
            ind_template_shared_np = np.frombuffer(ind_template_shared, "i").reshape(
                ind_template.shape
            )
            np.copyto(ind_template_shared_np, ind_template)
            ind_template = None

            # Create a shared RawArray that will contain the results
            # bold_reshape = (bold_shape[-1], )+(bold_shape[:-1])
            bold_reshape = bold_shape
            fun_4D_shared = multiprocessing.RawArray("f", int(np.prod(bold_reshape)))

            with multiprocessing.Pool(
                processes=nb_of_batchs,
                initializer=init_worker_voxelwise2,
                initargs=(
                    fun_4D_shared,
                    bold_reshape,
                    nb_of_batchs,
                    bold_2D_shared,
                    prior_type,
                    ind_template_shared,
                    voxel_mask,
                    pmap_loc,
                    results_dir,
                    precomp,
                ),
            ) as pool:
                poolCheck = pool.map_async(
                    Voxelwise_functionnectome2, range(nb_of_batchs)
                )
                percent = None  # to keep track of the preivous
                # Diplay the progress
                while not poolCheck.ready():
                    percent = LogDiplayPercent(results_dir, percent)
                    time.sleep(1)
                poolCheck.get()
                sys.stdout.write(
                    "\rProgress of the current process: 100%  Completed           \n"
                )
                sys.stdout.flush()
                logfiles = glob.glob(os.path.join(results_dir, "log_*.txt"))
                for logf in logfiles:
                    os.remove(logf)
            bold_2D_shared = bold_2D_shared_np = None

            sum_pmap4D_all = np.frombuffer(fun_4D_shared, "f").reshape(bold_reshape)
            if precomp:
                print(
                    "Multiprocessing done. Application of the proportionality and saving results."
                )
                # Applying proportionality
                # sum_pmap4D_all = np.divide(sum_pmap4D_all,
                #                            np.expand_dims(sum_pmap_all, 0),
                #                            out=sum_pmap4D_all,
                #                            where=np.expand_dims(sum_pmap_all, 0) != 0)
                # sum_pmap4D_all = np.moveaxis(sum_pmap4D_all, 0, -1)

                sum_pmap_all = np.expand_dims(sum_pmap_all, -1)
                sum_pmap4D_all = np.divide(
                    sum_pmap4D_all,
                    sum_pmap_all,
                    out=sum_pmap4D_all,
                    where=sum_pmap_all != 0,
                )
            else:
                print("Multiprocessing done. Saving results.")
            # Masking out the stray voxels out of the brain
            if maskOutput:
                sum_pmap4D_all *= np.expand_dims(template_vol, -1)
            # Saving the results
            if flipLR:
                sum_pmap4D_all = np.flip(sum_pmap4D_all, 0)
            sum_pmap4D_img = nib.Nifti1Image(sum_pmap4D_all, bold_affine, bold_header)
            nib.save(sum_pmap4D_img, finalOutPath)
            time.sleep(5)
            sum_pmap4D_img = sum_pmap4D_all = None

    print(f"Total run time : {time.time()-st}")


# %% Run the code if the script is called directly, the path to the setting file must be given as argument
def main():
    if sys.version_info[0] < 3:
        raise Exception(
            "Must be using Python 3. And probably Python 3.6 (or superior)."
        )
    if sys.version_info[1] < 6:
        warnings.warn("Python version < 3.6 |nIt might not work. Consider updating.")
    usageMsg = "Usage: Functionnectome <settings_file.fcntm>"
    if not len(sys.argv) == 2:
        print("Wrong number of input arguments.")
        print(usageMsg)
    else:
        settingFilePath = sys.argv[1]
        if not settingFilePath[-6:] == ".fcntm":
            if settingFilePath in ["-h", "--help"]:
                print(usageMsg)
            else:
                print('Wrong settings file (extension is not ".fcntm").')
                print(usageMsg)
        else:
            run_functionnectome(settingFilePath)


if __name__ == "__main__":
    main()
