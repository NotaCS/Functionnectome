#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:56:19 2020

@author: nozais

Si les fichiers 4D sont tous dans le même dossier, avec donc des noms différents du type "mySubXXX.nii.gz",
les fichiers de sortie seront placés dans des dossiers nommés "*/mySubXXX_functionnectome/"

Bold paths and mask paths are expected to follow the same type of naming convention.
Indeed, both lists containing them are sorted, and the paths are paired using their index
"""


import tkinter as tk
from tkinter import ttk
import os
import sys
import json
import warnings
import multiprocessing as mp
from tkinter import filedialog
from tkinter import messagebox
from pathlib import Path
import pkg_resources
import darkdetect


import Functionnectome.functionnectome as fun
from Functionnectome.functionnectome import PRIORS_H5  # , PRIORS_URL, PRIORS_ZIP
version = pkg_resources.require("Functionnectome")[0].version


class Functionnectome_GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # self.geometry('600x320')
        if darkdetect.isDark():
            self.bg_color = 'slategrey'
        else:
            self.bg_color = 'lightgrey'
        self.configure(background=self.bg_color)
        if version:
            self.title(f"Functionnectome processing (v{version})")
        else:
            self.title("Functionnectome processing")
        self.home = os.path.expanduser("~")
        self.cwd = os.getcwd()  # change to the last chosen folder along the way
        self.bold_paths = []  # List of the paths to the BOLD files
        self.run = False  # To run or not to run (the Functionnectome)
        # Message diplayed in a label about the number of BOLD files selected:
        self.numFiles = tk.StringVar()
        self.numFiles.set("0 BOLD files selected")
        self.nbFiles = tk.IntVar()  # The number of BOLD files selected
        self.nbFiles.set(0)
        self.nbFiles.trace("w", self.updateLbl)
        # List of the paths to the mask(s) used in voxelwise analysis:
        self.mask_paths = []
        # Buffer for mask_paths when the choice window is still open:
        self.mask_paths_tmp = []
        self.nbMasks = tk.IntVar()  # Number of masks selected
        self.nbMasks.set(0)
        # Part of the first BOLD path selected as subject ID:
        self.exSubName = tk.StringVar()
        self.exSubName.set("Ex: ")
        # Decomposition of the first BOLD path into its subcomponents (devided by "/" or "\"):
        self.exPath = ()
        self.subInPath = tk.StringVar()  # Location of the subject's ID in the path
        self.subInPath.set("0")
        # Output directory where the results will be written:
        self.outDir = tk.StringVar()
        # Optional white matter mask to select the output funtome voxels
        self.wmMask = tk.StringVar()
        self.ana_type = tk.StringVar()  # Type of analysis (voxelwise or regionwise)
        self.ana_type.trace("w", self.activation_maskBtn)
        self.nb_parallel_proc = tk.StringVar()
        self.nb_parallel_proc.set("1")
        self.priors = tk.StringVar()  # (h5 or nii)
        self.priors.set("h5")  # Used to be a choice, now enforce h5 only in the GUI
        self.priorsFileList = PRIORS_H5
        self.priorsChoice = tk.StringVar()
        self.priorsChoice.set(self.priorsFileList[0])
        self.customPriors = tk.StringVar()
        self.prior_path_h5 = ''  # Path to the h5 file (used when DL priors)
        self.priors_paths = {  # Initialize the dict (with the nii priors empty info)
            "template": "",
            "regions": "",
            "region_pmap": "",
            "voxel_pmap": "",
        }
        ipad = 3
        epad = 10

        # BOLD choice frame, first part : Select BOLD files from one folder
        self.fBOLD = tk.Frame(self, bd=1, relief="sunken", bg=self.bg_color)
        self.lbl1 = tk.Label(
            self.fBOLD,
            text="Select BOLD files from one folder",
            font="Helvetica 12 bold",
            bg=self.bg_color
        )
        self.buttonBold1 = tk.Button(
            self.fBOLD, text="Choose files", command=self.get_bold1, highlightbackground=self.bg_color,
        )
        self.numFiles_lbl1 = tk.Label(self.fBOLD, textvariable=self.numFiles, bg=self.bg_color)

        self.fBOLD.grid(
            column=0,
            row=0,
            padx=epad,
            pady=epad,
            ipadx=ipad,
            ipady=ipad,
            columnspan=2,
            rowspan=1,
            sticky="news",
        )
        self.lbl1.grid(column=0, row=0, columnspan=2, padx=5, pady=5, sticky="W")
        self.buttonBold1.grid(column=0, row=1, padx=5)
        self.numFiles_lbl1.grid(column=1, row=1)

        self.lblor = tk.Label(
            self.fBOLD,
            text="----------    or    ----------",
            font="Helvetica 12 italic",
            bg=self.bg_color,
        )
        self.lblor.grid(column=0, row=2, columnspan=2, sticky="news", pady=10)

        # BOLD frame, second part : Select BOLD files from several folders
        # self.f2 = tk.Frame(self, bd=1, relief='sunken')
        self.lbl2a = tk.Label(
            self.fBOLD,
            text="Select BOLD files by pasting paths",
            font="Helvetica 12 bold",
            bg=self.bg_color,
        )
        self.buttonBold2 = tk.Button(
            self.fBOLD,
            text="Choose files",
            command=lambda: self.get_files_paste("bold_paths", self.nbFiles),
            highlightbackground=self.bg_color,
        )
        self.numFiles_lbl2 = tk.Label(self.fBOLD, textvariable=self.numFiles, bg=self.bg_color)
        self.lbl2b = tk.Label(
            self.fBOLD, text="Position of the subject ID in the path:", bg=self.bg_color
        )
        self.posSubName = tk.Spinbox(
            self.fBOLD,
            from_=0,
            to=0,
            width=5,
            textvariable=self.subInPath,
            command=self.update_exSub,
            background=self.bg_color
        )
        self.exSub = tk.Label(self.fBOLD, textvariable=self.exSubName, bg=self.bg_color)

        # self.f2.grid(column=0, row=1, padx=epad, pady=epad, ipadx=ipad, ipady=ipad, columnspan=2, sticky='news')
        self.lbl2a.grid(column=0, row=3, columnspan=2, padx=5, pady=5, sticky="W")
        self.buttonBold2.grid(column=0, row=4, padx=5, pady=5)
        self.numFiles_lbl2.grid(column=1, row=4)
        self.lbl2b.grid(column=0, row=5, columnspan=2, padx=5, sticky="W")
        self.posSubName.grid(column=0, row=6)
        self.exSub.grid(column=1, row=6, sticky="W")

        # View the selected files
        self.butonViewFiles = tk.Button(
            self, text="View selected files", command=self.view_files, highlightbackground=self.bg_color,
        )
        self.butonViewFiles.grid(
            column=0, row=1, columnspan=2, padx=epad, pady=epad, sticky="ew"
        )

        # Output frame : Choose output folder
        self.fOut = tk.Frame(self, bd=1, relief="sunken", bg=self.bg_color)
        self.lbl3a = tk.Label(self.fOut, text="Output folder", font="Helvetica 12 bold", bg=self.bg_color)
        self.outDirIn = tk.Entry(self.fOut, bd=1, textvariable=self.outDir, width=30)
        self.buttonOutDir = tk.Button(self.fOut, text="...", command=self.get_outDir, highlightbackground=self.bg_color)
        self.lbl3b = tk.Label(
            self.fOut, text="White matter output mask (optional):", font="Helvetica 12 bold", bg=self.bg_color
        )
        self.wm_maskIn = tk.Entry(self.fOut, bd=1, textvariable=self.wmMask, width=30)
        self.buttonWMmask = tk.Button(self.fOut, text="...", command=self.get_wmMask, highlightbackground=self.bg_color)

        self.fOut.grid(
            column=2,
            row=1,
            padx=epad,
            pady=epad,
            ipadx=ipad,
            ipady=ipad,
            columnspan=2,
            sticky="news",
        )
        self.lbl3a.grid(column=0, row=0, columnspan=1, sticky="W", padx=5, pady=5)
        self.outDirIn.grid(column=0, row=1, columnspan=2, sticky="W", padx=5)
        self.buttonOutDir.grid(column=2, row=1, sticky="W")
        self.lbl3b.grid(column=0, row=2, columnspan=1, sticky="W", padx=5, pady=5)
        self.wm_maskIn.grid(column=0, row=3, columnspan=2, sticky="W", padx=5)
        self.buttonWMmask.grid(column=2, row=3, sticky="W")

        # Analysis frame : Choose analysis options
        self.fAna = tk.Frame(self, bd=1, relief="sunken", bg=self.bg_color)
        self.lbl4a = tk.Label(
            self.fAna,
            text="Choose the type of analysis to run:",
            font="Helvetica 12 bold",
            bg=self.bg_color,
        )
        self.ana_region = tk.Radiobutton(
            self.fAna,
            text="Region-wise functionnectome",
            variable=self.ana_type,
            value="region",
            bg=self.bg_color
        )
        self.ana_voxel = tk.Radiobutton(
            self.fAna,
            text="Voxel-wise functionnectome",
            variable=self.ana_type,
            value="voxel",
            bg=self.bg_color,
        )
        self.buttonMask = tk.Button(
            self.fAna, text="Select grey matter mask(s)", command=self.get_masks, highlightbackground=self.bg_color
        )
        self.lbl4b = tk.Label(
            self.fAna, text="Number of parallel processes:", font="Helvetica 12 bold", bg=self.bg_color
        )
        self.parallel_proc = tk.Spinbox(
            self.fAna,
            from_=1,
            to=os.cpu_count(),
            width=5,
            textvariable=self.nb_parallel_proc,
            bg=self.bg_color
        )

        self.lbl4c = tk.Label(
            self.fAna, text="Choice of priors: ", font="Helvetica 12 bold", bg=self.bg_color
        )
        self.buttonPriors = tk.Button(
            self.fAna, text="Select priors", command=self.selectPriors,  highlightbackground=self.bg_color
        )
        self.lblpriors1 = tk.Label(
            self.fAna, text="Selected priors: ",
            font="Helvetica 12 italic", bg=self.bg_color
        )
        self.lblpriors2 = tk.Label(
            self.fAna, textvariable=self.priorsChoice,
            font="Helvetica 12 italic", bg=self.bg_color
        )
        self.ana_type.set("voxel")

        self.fAna.grid(
            column=2,
            row=0,
            padx=epad,
            pady=epad,
            ipadx=ipad,
            ipady=ipad,
            columnspan=2,
            rowspan=1,
            sticky="news",
        )
        self.lbl4a.grid(column=0, row=0, columnspan=2, sticky="W", padx=5, pady=5)
        self.ana_region.grid(column=0, row=2, columnspan=2, sticky="W")
        self.ana_voxel.grid(column=0, row=3, columnspan=2, sticky="W")
        self.buttonMask.grid(column=0, row=4, columnspan=2, padx=5, sticky="EW")
        self.lbl4b.grid(column=0, row=5, columnspan=2, padx=5, pady=10, sticky="W")
        self.parallel_proc.grid(column=2, row=5, sticky="W")
        self.lbl4c.grid(column=0, row=6, columnspan=1, sticky="W", padx=5, pady=5)
        self.buttonPriors.grid(column=1, row=6, columnspan=1, sticky="EW", padx=5, pady=5)
        self.lblpriors1.grid(column=0, row=7, columnspan=1, sticky="W", padx=5, pady=5)
        self.lblpriors2.grid(column=1, row=7, columnspan=2, sticky="W", padx=5, pady=5)

        # Bottom buttons
        self.saveBtn = tk.Button(self, text="Save", command=self.choseFileAndSave, highlightbackground=self.bg_color)
        self.loadBtn = tk.Button(self, text="Load", command=self.loadSettings, highlightbackground=self.bg_color)
        self.lauchBtn = tk.Button(self, text="Launch", command=self.launchAna, highlightbackground=self.bg_color)
        self.quitBtn = tk.Button(self, text="Exit", command=self.destroy, highlightbackground=self.bg_color)

        self.saveBtn.grid(column=0, row=3, padx=epad, pady=epad)
        self.loadBtn.grid(column=1, row=3, padx=epad, pady=epad)
        self.lauchBtn.grid(column=2, row=3, padx=epad, pady=epad)
        self.quitBtn.grid(column=3, row=3, padx=epad, pady=epad)

    # %% Open the window for masks selection
    def get_masks(self):
        def cancel_btn():
            self.nbMasks.set(len(self.mask_paths))
            top_mask.destroy()
            top_mask.update()

        def choose_masks():
            masks_path_tmp = filedialog.askopenfilenames(
                parent=top_mask,
                initialdir=self.cwd,
                title="Choose the masks files",
                filetypes=[("Nifti files", ".nii .gz")],
            )
            if masks_path_tmp:
                self.mask_paths_tmp = list(masks_path_tmp)
                self.mask_paths_tmp.sort()
                self.nbMasks.set(len(masks_path_tmp))
                self.cwd = os.path.dirname(masks_path_tmp[0])

        def choose_mask():
            mask_file_tmp = filedialog.askopenfilename(
                parent=top_mask,
                initialdir=self.cwd,
                title="Choose the mask file",
                filetypes=[("Nifti files", ".nii .gz")],
            )
            if mask_file_tmp:
                self.cwd = os.path.dirname(mask_file_tmp)
                self.mask_paths_tmp = [mask_file_tmp]
                self.nbMasks.set(1)

        def view_masks():
            if self.mask_paths_tmp:
                ckeckWinMask = tk.Toplevel()
                ckeckWinMask.title("Selected files")
                preMsg = [
                    f"File {ii+1} : " + txt
                    for ii, txt in enumerate(self.mask_paths_tmp)
                ]
                filesInMsg = tk.StringVar()
                msg = tk.Message(ckeckWinMask, textvariable=filesInMsg, width=700)
                msg.pack()
                filesInMsg.set("\n".join(preMsg))
            else:
                messagebox.showinfo(
                    "Selected masks", "No file selected yet.", parent=top_mask
                )

        def ok_maskBtn():
            if (
                nbmask.get() == "common"
                and self.nbMasks.get() == 1
                or nbmask.get() == "subjectwise"
                and self.nbMasks.get() == self.nbFiles.get()
            ):

                self.mask_paths = self.mask_paths_tmp
                top_mask.destroy()
                top_mask.update()
            elif nbmask.get() == "common" and self.nbMasks.get() > 1:
                messagebox.showwarning(
                    "Bad number of masks",
                    (
                        'There are multiple masks selected (check the "View files" button), '
                        'but the "Same mask for all" option is selected.'
                    ),
                    parent=top_mask,
                )
            elif (
                nbmask.get() == "subjectwise"
                and self.nbMasks.get() != self.nbFiles.get()
            ):
                messagebox.showwarning(
                    "Bad number of masks",
                    (
                        f"The number of selected masks ({self.nbMasks.get()}) "
                        f"doesn't match the number of selected BOLD files ({self.nbFiles.get()})"
                    ),
                    parent=top_mask,
                )
            elif self.nbMasks.get() == 0:
                continue_answ = messagebox.askokcancel(
                    title="No mask selected",
                    message="No mask selected. Continue?",
                    parent=top_mask,
                )
                if continue_answ:
                    self.mask_paths = self.mask_paths_tmp
                    top_mask.destroy()
                    top_mask.update()
                else:
                    return

        def activation_btn(*arg):
            if nbmask.get() == "subjectwise":
                btn_paste.config(state="normal")
                btn_select1.config(state="normal")
                btn_select2.config(state="disabled")
            else:
                btn_paste.config(state="disabled")
                btn_select1.config(state="disabled")
                btn_select2.config(state="normal")

        top_mask = tk.Toplevel(self)
        top_mask.configure(background=self.bg_color)
        top_mask.title("Masking options")
        top_mask.grab_set()
        self.mask_paths_tmp = self.mask_paths
        nbmask = tk.StringVar()
        nbmask.trace("w", activation_btn)

        nb_mask1 = tk.Radiobutton(
            top_mask, text="One mask per subject", variable=nbmask, value="subjectwise", bg=self.bg_color
        )
        nb_mask2 = tk.Radiobutton(
            top_mask, text="Same mask for all", variable=nbmask, value="common", bg=self.bg_color
        )
        btn_paste = tk.Button(
            top_mask,
            text="Paste paths",
            command=lambda: self.get_files_paste("mask_paths_tmp", self.nbMasks),
            highlightbackground=self.bg_color
        )
        btn_select1 = tk.Button(top_mask, text="Select files", command=choose_masks, highlightbackground=self.bg_color)
        btn_select2 = tk.Button(top_mask, text="Select the file", command=choose_mask,
                                highlightbackground=self.bg_color)
        btn_cancel = tk.Button(top_mask, text="Cancel", command=cancel_btn, highlightbackground=self.bg_color)
        btn_view = tk.Button(top_mask, text="View selected masks",
                             command=view_masks, highlightbackground=self.bg_color)
        btn_ok = tk.Button(top_mask, text="OK", command=ok_maskBtn, highlightbackground=self.bg_color)

        nb_mask1.grid(column=0, row=0, rowspan=2, columnspan=2, padx=5, sticky="w")
        nb_mask2.grid(column=0, row=2, columnspan=2, padx=5, sticky="w")
        btn_paste.grid(column=2, row=0, pady=(10, 0), sticky="ew")
        btn_select1.grid(column=2, row=1, pady=(0, 10), sticky="ew")
        btn_select2.grid(column=2, row=2, pady=10, sticky="ew")
        btn_view.grid(column=0, row=3, columnspan=3, pady=5, sticky="we")
        btn_cancel.grid(column=0, row=4, pady=10, sticky="w")
        btn_ok.grid(column=2, row=4, pady=10, sticky="ew")

        if len(self.mask_paths_tmp) == 1:
            nbmask.set("common")
        else:
            nbmask.set("subjectwise")

    # %%  Open the window for priors selection
    def selectPriors(self):
        pkgPath = os.path.dirname(__file__)
        jsonPath = os.path.join(pkgPath, "priors_paths.json")
        if os.path.exists(jsonPath):
            with open(jsonPath, "r") as jsonP:
                self.priors_paths = json.load(jsonP)
            self.priors_paths = fun.updateOldJson(jsonPath, self.priors_paths)
        missingH5 = fun.find_missingH5(self.priors_paths)
        DLedH5 = fun.find_dlwedH5(self.priors_paths)

        def selectPriorsF(priors):
            i = self.priorsFileList.index(priors)
            ppath = filedialog.askopenfilename(
                parent=top_priors,
                initialdir=self.cwd,
                title=f"Choose priors file for {priors}",
                filetypes=[("HDF5 file", ".h5")],
            )
            if ppath:
                self.cwd = os.path.dirname(ppath)
                self.priors_paths[priors] = ppath
                with open(jsonPath, "w") as jsonP:
                    json.dump(self.priors_paths, jsonP)
                self.dlBool[i].set(False)
                self.dlChecks[i].config(state="disabled")
                self.selectWidgts[i].config(state="disabled")

        def selectCustomF():
            customPath = filedialog.askopenfilename(
                parent=top_priors,
                initialdir=self.cwd,
                title="Choose custom priors file",
                filetypes=[("HDF5 file", ".h5")],
            )
            if customPath:
                self.cwd = os.path.dirname(customPath)
                self.customPriors.set(customPath)

        def showPath(p):
            ppath = self.priors_paths[p]
            messagebox.showinfo("Priors local file path", ppath)

        def showCustom():
            messagebox.showinfo("Custom priors file path", self.customPriors.get())

        def manualDL():
            checkBools = [b.get() for b in self.dlBool]
            if not any(checkBools):
                return
            prior_dirpath_h5 = filedialog.askdirectory(
                initialdir=self.cwd, parent=top_priors, title="Choose where to save the priors"
            )
            if prior_dirpath_h5:
                self.cwd = prior_dirpath_h5
                for i, priors in enumerate(self.priorsFileList):
                    if checkBools[i]:
                        print(f'Downloading: {priors}')
                        pool = mp.Pool(processes=1)
                        res = pool.apply_async(fun.Download_H5, (prior_dirpath_h5, priors))
                        DLwindow = tk.Toplevel(self)
                        DLwindow.title("Downloading")
                        DLwindow.grab_set()
                        pb = ttk.Progressbar(
                            DLwindow,
                            orient='horizontal',
                            mode='indeterminate',
                            length=280
                        )
                        infolabel = ttk.Label(DLwindow,
                                              text='Downloading in progress... Check the terminal for more details.',
                                              )
                        infolabel.grid(column=0, row=0)
                        pb.grid(column=0, row=1)
                        pb.start()
                        DLwindow.update()
                        DLwindow.after(1000, self.check_if_running, res, pool, DLwindow)  # Also fill self.prior_path_h5
                        self.wait_window(DLwindow)
                        if os.path.exists(self.prior_path_h5):
                            self.priors_paths[priors] = self.prior_path_h5
                            with open(jsonPath, "w") as jsonP:
                                json.dump(self.priors_paths, jsonP)
                            self.dlBool[i].set(False)
                            self.dlChecks[i].config(state="disabled")
                            self.selectWidgts[i].config(state="disabled")

        def ok_priorsBtn():
            top_priors.destroy()
            top_priors.update()

        top_priors = tk.Toplevel(self)
        top_priors.configure(background=self.bg_color)
        top_priors.title("Priors selection and download")
        top_priors.grab_set()

        self.lblTitle = tk.Label(
            top_priors, text="Selected the priors for the analysis:",
            font="Helvetica 12 bold", bg=self.bg_color
        )
        self.lblDL = tk.Label(
            top_priors, text="To download",
            font="Helvetica 12 bold", bg=self.bg_color
        )
        self.radioListPriors = []
        self.dlChecks = []
        self.dlBool = []
        self.selectWidgts = []
        self.fun4button = []
        for i, priors in enumerate(self.priorsFileList):
            self.radioListPriors.append(
                tk.Radiobutton(
                    top_priors,
                    text=priors,
                    variable=self.priorsChoice,
                    value=priors,
                    bg=self.bg_color
                )
            )

            self.dlBool.append(tk.BooleanVar())
            self.dlChecks.append(
                tk.Checkbutton(top_priors, variable=self.dlBool[i], bg=self.bg_color)
            )
            if priors in missingH5:
                self.fun4button.append(lambda priors=priors: selectPriorsF(priors))
                self.selectWidgts.append(
                    tk.Button(top_priors, text="Select", command=self.fun4button[i],
                              highlightbackground=self.bg_color)
                )
            elif priors in DLedH5:
                if self.ana_type.get() == 'region':  # Too slow for voxelwise test
                    if not fun.testPriorsH5(self.priors_paths[priors], 'region'):
                        self.radioListPriors[-1].config(state="disabled")
                self.fun4button.append(lambda priors=priors: showPath(priors))
                self.selectWidgts.append(
                    tk.Button(top_priors, text="View local path", command=self.fun4button[i],
                              highlightbackground=self.bg_color)
                )
                self.dlChecks[i].config(state="disabled")
        btn_DL = tk.Button(top_priors, text="Manual download", command=manualDL, highlightbackground=self.bg_color)
        if len(missingH5) == 0:
            btn_DL.config(state="disabled")

        customRadio = tk.Radiobutton(
            top_priors,
            text="Custom priors",
            variable=self.priorsChoice,
            value="Custom priors",
            bg=self.bg_color
        )
        customSelectBtn = tk.Button(
            top_priors, text="Select", command=selectCustomF,
            highlightbackground=self.bg_color
        )
        customShowBtn = tk.Button(
            top_priors, text="View selected file",
            command=showCustom,
            highlightbackground=self.bg_color
        )

        btn_ok = tk.Button(top_priors, text="OK", command=ok_priorsBtn, highlightbackground=self.bg_color)

        self.lblTitle.grid(column=0, row=0, padx=5, pady=5, sticky="W")
        self.lblDL.grid(column=1, row=0, padx=5, pady=5, sticky="W")
        for i, priors in enumerate(self.priorsFileList):
            self.radioListPriors[i].grid(column=0, row=i+1, padx=5, pady=5, sticky="W")
            self.dlChecks[i].grid(column=1, row=i+1, sticky="EW")
            self.selectWidgts[i].grid(column=2, row=i+1, padx=5, pady=5, sticky="W")

        lastRow = len(self.radioListPriors) + 3
        btn_DL.grid(column=1, row=lastRow - 2, sticky="w")
        customRadio.grid(column=0, row=lastRow - 1, sticky="ew")
        customSelectBtn.grid(column=1, row=lastRow - 1, sticky="ew")
        customShowBtn.grid(column=2, row=lastRow - 1, sticky="ew")
        btn_ok.grid(column=0, row=lastRow, sticky="ew")

    # %%
    def launchAna(self):  # Create output folder, close the GUI and run the analysis
        settingsTxt = self.saveSettings()
        if settingsTxt == 0:
            return
        Path(self.outDir.get()).mkdir(parents=True, exist_ok=True)
        self.fpath = os.path.join(self.outDir.get(), "settings.fcntm")
        n = 1
        while os.path.exists(
            self.fpath
        ):  # Add a number in the filename if there is already one existing
            self.fpath = os.path.join(self.outDir.get(), f"settings{n}.fcntm")
            n += 1
        with open(self.fpath, "w+") as f:
            f.write(settingsTxt)
        self.run = True
        self.destroy()
        self.update()

    def get_bold1(self):
        """
        Select the files from one directory using a dialog window
        """
        self.exSubName.set("Ex: ")
        self.exPath = ()
        self.subInPath.set("0")
        bold_paths_tmp = filedialog.askopenfilenames(
            parent=self.fBOLD,
            initialdir=self.cwd,
            title="Choose the BOLD files",
            filetypes=[("Nifti files", ".nii .gz")],
        )
        if bold_paths_tmp:  # to manage empty answers and "Cancel" use
            self.bold_paths = list(bold_paths_tmp)
            self.bold_paths.sort()
            self.cwd = os.path.dirname(bold_paths_tmp[0])
        self.nbFiles.set(len(self.bold_paths))

    def update_exSub(self):
        if self.exPath:
            self.exSubName.set("Ex: " + self.exPath[int(self.posSubName.get())])
        else:
            self.exSubName.set("Ex: ")

    def view_files(self):
        if self.bold_paths:
            ckeckWin = tk.Toplevel()
            ckeckWin.title("Selected files")
            preMsg = [f"File {ii+1} : " + txt for ii, txt in enumerate(self.bold_paths)]
            filesInMsg = tk.StringVar()
            msg = tk.Message(ckeckWin, textvariable=filesInMsg, width=700)
            msg.pack()
            filesInMsg.set("\n".join(preMsg))
        else:
            messagebox.showinfo("Selected files", "No file selected yet.")

    def check_if_running(self, rez, mppool, window):
        """Check every second if the multiprocessing 'rez' process in 'mppool' is finished."""
        if not rez.ready():
            window.update()
            window.after(1000, self.check_if_running, rez, mppool, window)
        else:
            mppool.close()
            mppool.join()
            self.prior_path_h5 = rez.get()
            window.destroy()

    def get_outDir(self):
        odir = self.outDir.get()
        if not odir:
            odir = self.cwd
        outDir_tmp = filedialog.askdirectory(
            initialdir=odir,
            parent=self.fOut,
            title="Choose or create the output folder",
        )
        if outDir_tmp:  # to manage empty answers and "Cancel" use
            self.outDir.set(outDir_tmp)
            self.cwd = os.path.dirname(outDir_tmp)
            # if not os.path.isdir(outDir_tmp):
            #     os.mkdir(outDir_tmp)

    def get_wmMask(self):
        wmf_tmp = filedialog.askopenfilename(
            parent=self.fOut,
            initialdir=self.cwd,
            title="Choose the white matter mask file",
            filetypes=[("Nifti files", ".nii .gz")],
        )
        if wmf_tmp:
            self.cwd = os.path.dirname(wmf_tmp)
            self.wmMask.set(wmf_tmp)

    def updateLbl(self, *args):
        self.numFiles.set(f"{self.nbFiles.get()} BOLD files selected")
        if self.nbFiles.get() == 0:
            self.exPath = ()
            self.exSubName.set("Ex: ")
            self.subInPath.set("0")

    def activation_maskBtn(self, *args):
        if self.ana_type.get() == "voxel":
            self.buttonMask.config(state="normal")
        else:
            self.buttonMask.config(state="disabled")

    # %%
    def get_files_paste(self, paths_varname, nbpaths):
        """
        Display a textbox so that the user car directly paste-in the files paths
        """
        pasteWin = tk.Toplevel()
        pasteWin.configure(background=self.bg_color)
        pasteWin.title("Select the files")
        pasteWin.grab_set()

        paths = getattr(self, paths_varname)

        lblgetfile = tk.Label(pasteWin, text="Paste the paths to all the files here :", bg=self.bg_color)
        txtPaths = tk.Text(pasteWin)
        # if the files are tagged as "bad", display them in red
        txtPaths.tag_configure("bad", background="red", foreground="white")
        scrY = tk.Scrollbar(pasteWin, orient=tk.VERTICAL, command=txtPaths.yview)
        txtPaths.config(yscrollcommand=scrY.set)

        lblgetfile.grid(column=0, row=0, columnspan=3)
        txtPaths.grid(column=0, row=1, columnspan=3)
        scrY.grid(column=3, row=1, sticky="NS")

        for ii, fpath in enumerate(paths):
            txtPaths.insert(f"{ii}.0", fpath + "\n")

        def cancel_btn():
            pasteWin.destroy()
            pasteWin.update()

        def clear_btn():
            txtPaths.delete("1.0", tk.END)

        def ok_btn():
            prePath1 = txtPaths.get("1.0", tk.END).split("\n")
            # prePath1 = list(map(lambda s: "".join(s.split()), prePath1))  # remove all whitespaces (WHY??? Windows?)
            prePath1 = [
                p.strip() for p in prePath1
            ]  # Remove leading and trailing whitespaces⎄
            prePath2 = list(filter(None, prePath1))  # remove empty lines
            prePath2.sort()

            if not prePath2:  # if there is nothing in the textbox
                pasteWin.destroy()
                pasteWin.update()
                setattr(self, paths_varname, [])
                nbpaths.set(0)
                return

            check_goodNii = [
                (p.endswith((".nii", ".nii.gz")) and os.path.exists(p))
                for p in prePath2
            ]

            if not all(check_goodNii):  # if at leat one extension is bad
                messagebox.showwarning(
                    "Bad files",
                    (
                        "Some paths entered do not exist or do not have the proper "
                        "extension (.nii or .nii.gz)'."
                    ),
                    parent=pasteWin,
                )
                txtPaths.delete("1.0", tk.END)
                for ii, (fpath, cpath) in enumerate(zip(prePath2, check_goodNii)):
                    if cpath:
                        txtPaths.insert(f"{ii}.0", fpath + "\n")
                    else:
                        txtPaths.insert(f"{ii}.0", fpath + "\n", "bad")
                return

            # Decompose the paths into its componenents (separated by the OS separator "/" or "\")
            # to find where they start to be different
            decomposedPaths = [
                tuple(filter(None, os.path.normpath(pp2).split(os.path.sep)))
                for pp2 in prePath2
            ]

            continue_answ = True
            lenPaths = [len(p) for p in decomposedPaths]
            if paths_varname == "bold_paths" and not lenPaths[1:] == lenPaths[:-1]:
                # If the paths have the same number of directories
                msg = (
                    "The paths entered do not all have the same length "
                    "(i.e. each path do not have the same number of sub-directories).\n"
                    "This might hinder (or even break) the naming of the output folders and files."
                )
                continue_answ = messagebox.askokcancel(
                    title="Uneven path length", message=msg, parent=pasteWin
                )

            if (
                continue_answ
            ):  # if everything is good, the window closes and the data is retreived
                setattr(self, paths_varname, prePath2)
                nbpaths.set(len(prePath2))
                pasteWin.destroy()
                pasteWin.update()
                if paths_varname == "bold_paths":
                    # Try to guess what the subject ID is based on the BOLD paths entered
                    nameGuessed = 0
                    lenPath = min([len(dp) for dp in decomposedPaths]) - 1
                    for ii in range(
                        lenPath
                    ):  # Scan through and compare the paths components
                        subPaths = [p[ii] for p in decomposedPaths]
                        if (
                            subPaths[1:] == subPaths[:-1]
                        ):  # Test if all values are equal
                            nameGuessed += 1
                        else:  # When they are different, stops : it's probably the ID of the subject
                            break
                    self.exPath = decomposedPaths[0]
                    self.exSubName.set("Ex: " + self.exPath[nameGuessed])
                    self.posSubName.config(to=lenPath)
                    self.subInPath.set(str(nameGuessed))

        btnCancel = tk.Button(pasteWin, text="Cancel", command=cancel_btn, highlightbackground=self.bg_color)
        btnCancel.grid(column=0, row=2)

        btnClear = tk.Button(pasteWin, text="Clear", command=clear_btn, highlightbackground=self.bg_color)
        btnClear.grid(column=1, row=2)

        btnOK = tk.Button(pasteWin, text="OK", command=ok_btn, highlightbackground=self.bg_color)
        btnOK.grid(column=2, row=2)

    # %%
    def saveSettings(self):
        if not self.outDir.get():
            messagebox.showwarning(
                title="No output folder",
                message=(
                    "No output folder selected for the analysis.\n"
                    "Complete the field and retry."
                ),
                parent=self,
            )
            return 0
        if not self.bold_paths:
            messagebox.showwarning(
                title="No BOLD file selected",
                message="No BOLD file selected.\nSelect at least one file and retry.",
                parent=self,
            )
            return 0
        if self.ana_type.get() == "voxel" and not self.mask_paths:
            msg = (
                "No mask has been selected for the voxelwise analysis.\n"
                + "Default behavior is to process all the brain's voxels.\n"
                + "That's a whole lot of voxels... Are you sure you want to continue?"
            )
            continue_answ = messagebox.askokcancel(
                title="No mask selected", message=msg, parent=self
            )
            if not continue_answ:
                return 0
        if (
            self.ana_type.get() == "voxel"
            and self.nbMasks.get() > 1
            and self.nbMasks.get() != self.nbFiles.get()
        ):
            messagebox.showwarning(
                "Bad number of masks",
                (
                    "There are multiple masks selected, "
                    f"but, the number of selected masks ({self.nbMasks.get()}) "
                    f"doesn't match the number of selected BOLD files ({self.nbFiles.get()})."
                ),
                parent=self,
            )
            return 0

        posID = self.subInPath.get() if self.exPath else "-1"
        firstSep = 1 if (self.bold_paths[0][0] == os.path.sep and int(posID) > 0) else 0
        listID = [
            os.path.normpath(boldpath).split(os.path.sep)[int(posID) + firstSep]
            for boldpath in self.bold_paths
        ]
        if len(set(listID)) < len(listID):  # If the subjects' ID are not all unique
            confirm_save = messagebox.askokcancel(
                title="Non-unique subject ID",
                message=(
                    "There are multiple identical subject IDs, "
                    "defined through the files' paths: If you have multiple runs per subjects, "
                    "ignore this message. Otherwise, the position "
                    "of the subject ID in the path is probably wrong."
                ),
                parent=self,
            )
            if not confirm_save:
                return 0

        outputMask = str(self.wmMask.get())
        if not outputMask:
            outputMask = '1'  # always mask the output with the template

        if self.priorsChoice.get() == "Custom priors":
            if os.path.isfile(self.customPriors.get()):
                settingsTxt = fun.makeSettingsTxt(
                    self.outDir.get(), self.ana_type.get(), self.nb_parallel_proc.get(), self.priors.get(),
                    self.priorsChoice.get(), posID, outputMask, self.nbFiles.get(), self.bold_paths,
                    self.nbMasks.get(), self.mask_paths,
                    optSett=True, opt_h5_loc=self.customPriors.get()
                )
            else:
                messagebox.showwarning(
                    "No correct custom priors",
                    (
                        "The priors chosen for the analysis are custom priors. However, no correct "
                        f"file was detected following the given file path ({self.customPriors.get()})."
                    ),
                    parent=self,
                )
                return 0
        else:
            settingsTxt = fun.makeSettingsTxt(
                self.outDir.get(), self.ana_type.get(), self.nb_parallel_proc.get(), self.priors.get(),
                self.priorsChoice.get(), posID, outputMask, self.nbFiles.get(), self.bold_paths,
                self.nbMasks.get(), self.mask_paths)
        return settingsTxt

    def choseFileAndSave(self):
        settingsTxt = self.saveSettings()
        if settingsTxt == 0:
            return
        f = filedialog.asksaveasfile(
            parent=self.fBOLD,
            initialdir=self.cwd,
            mode="w",
            defaultextension=".fcntm",
            title="Save settings (.fcntm)",
            filetypes=[("Setting files", ".fcntm"), ("All files", ".*")],
        )
        if f is not None:  # asksaveasfile return `None` if dialog closed with "cancel".
            f.write(settingsTxt)
            f.close()

    def loadSettings(self):
        settingFilePath = filedialog.askopenfilename(
            parent=self,
            initialdir=self.cwd,
            title="Choose the settings file to load",
            filetypes=[("Setting files", ".fcntm"), ("All files", ".*")],
        )
        if not settingFilePath:
            return
        self.cwd = os.path.dirname(settingFilePath)
        settingsDict = fun.readSettings(settingFilePath, forGUI=True)
        if isinstance(settingsDict, Exception):
            messagebox.showwarning('Error', settingsDict.args[0])
            return
        self.outDir.set(settingsDict['results_dir_root'])
        wm_maskF = settingsDict['maskOutput']
        if wm_maskF in ['0', '1']:
            self.wmMask.set('')
        else:
            self.wmMask.set(wm_maskF)
        self.ana_type.set(settingsDict['anatype'])
        self.nb_parallel_proc.set(settingsDict['nb_of_batchs'])
        self.nbFiles.set(settingsDict['subNb'])
        self.priors.set(settingsDict['prior_type'])
        self.priorsChoice.set(settingsDict['priorsH5'])
        self.bold_paths = settingsDict['bold_paths']
        if settingsDict['subIDpos'] == -1:
            self.subInPath.set("0")
            self.exSubName.set("Ex: ")
            self.exPath = ()
        else:
            self.subInPath.set(settingsDict['subIDpos'])
            self.exPath = tuple(
                filter(
                    None, os.path.normpath(self.bold_paths[0]).split(os.path.sep)
                )
            )
            self.posSubName.config(to=len(self.exPath) - 1)
            self.exSubName.set("Ex: " + self.exPath[int(self.subInPath.get())])
        if settingsDict['mask_nb']:
            self.nbMasks.set(settingsDict['mask_nb'])
            self.mask_paths = settingsDict['masks_vox']
        else:
            self.nbMasks.set(0)
            self.mask_paths = []


# %%
def run_gui():
    if sys.version_info[0] < 3:
        raise Exception(
            "Must be using Python 3. And probably Python 3.6 (or superior)."
        )
    if sys.version_info[1] < 6:
        warnings.warn("Python version < 3.6 |nIt might not work. Consider updating.")
    functionnectome_GUI = Functionnectome_GUI()
    functionnectome_GUI.mainloop()
    if functionnectome_GUI.run:
        fun.run_functionnectome(functionnectome_GUI.fpath, from_GUI=True)


if __name__ == "__main__":
    run_gui()
