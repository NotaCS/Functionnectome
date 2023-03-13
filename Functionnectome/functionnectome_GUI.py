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

try:
    import Functionnectome.functionnectome as fun
    from Functionnectome.functionnectome import PRIORS_H5  # , PRIORS_URL, PRIORS_ZIP
    version = pkg_resources.require("Functionnectome")[0].version
except ModuleNotFoundError:
    print(
        "The Functionnectome module was not found (probably not installed via pip)."
        " Importing functions from the folder where the current script was saved..."
    )
    import functionnectome as fun
    from functionnectome import PRIORS_H5  # , PRIORS_URL, PRIORS_ZIP
    version = ''


class Functionnectome_GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # self.geometry('600x320')
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
        self.priors = tk.StringVar()  # h5 or nii
        self.priors.trace("w", self.activation_h5)
        self.priorsChoice = tk.StringVar()
        self.prior_path_h5 = ''  # Path to the h5 file (used when DL priors)
        self.DLall = tk.BooleanVar()  # To check if all priors should be manually DLed
        self.priors_paths = {  # Initialize the dict (with the nii priors empty info)
            "template": "",
            "regions": "",
            "region_pmap": "",
            "voxel_pmap": "",
        }
        ipad = 3
        epad = 10

        # BOLD choice frame, first part : Select BOLD files from one folder
        self.fBOLD = tk.Frame(self, bd=1, relief="sunken")
        self.lbl1 = tk.Label(
            self.fBOLD,
            text="Select BOLD files from one folder",
            font="Helvetica 12 bold",
        )
        self.buttonBold1 = tk.Button(
            self.fBOLD, text="Choose files", command=self.get_bold1
        )
        self.numFiles_lbl1 = tk.Label(self.fBOLD, textvariable=self.numFiles)

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
        )
        self.lblor.grid(column=0, row=2, columnspan=2, sticky="news", pady=10)

        # BOLD frame, second part : Select BOLD files from several folders
        # self.f2 = tk.Frame(self, bd=1, relief='sunken')
        self.lbl2a = tk.Label(
            self.fBOLD,
            text="Select BOLD files by pasting paths",
            font="Helvetica 12 bold",
        )
        self.buttonBold2 = tk.Button(
            self.fBOLD,
            text="Choose files",
            command=lambda: self.get_files_paste("bold_paths", self.nbFiles),
        )
        self.numFiles_lbl2 = tk.Label(self.fBOLD, textvariable=self.numFiles)
        self.lbl2b = tk.Label(
            self.fBOLD, text="Position of the subject ID in the path:"
        )
        self.posSubName = tk.Spinbox(
            self.fBOLD,
            from_=0,
            to=0,
            width=5,
            textvariable=self.subInPath,
            command=self.update_exSub,
        )
        self.exSub = tk.Label(self.fBOLD, textvariable=self.exSubName)

        # self.f2.grid(column=0, row=1, padx=epad, pady=epad, ipadx=ipad, ipady=ipad, columnspan=2, sticky='news')
        self.lbl2a.grid(column=0, row=3, columnspan=2, padx=5, pady=5, sticky="W")
        self.buttonBold2.grid(column=0, row=4, padx=5, pady=5)
        self.numFiles_lbl2.grid(column=1, row=4)
        self.lbl2b.grid(column=0, row=5, columnspan=2, padx=5, sticky="W")
        self.posSubName.grid(column=0, row=6)
        self.exSub.grid(column=1, row=6, sticky="W")

        # View the selected files
        self.butonViewFiles = tk.Button(
            self, text="View selected files", command=self.view_files
        )
        self.butonViewFiles.grid(
            column=0, row=1, columnspan=2, padx=epad, pady=epad, sticky="ew"
        )

        # Output frame : Choose output folder
        self.fOut = tk.Frame(self, bd=1, relief="sunken")
        self.lbl3a = tk.Label(self.fOut, text="Output folder", font="Helvetica 12 bold")
        self.outDirIn = tk.Entry(self.fOut, bd=1, textvariable=self.outDir, width=30)
        self.buttonOutDir = tk.Button(self.fOut, text="...", command=self.get_outDir)
        self.lbl3b = tk.Label(
            self.fOut, text="White matter output mask (optional):", font="Helvetica 12 bold"
        )
        self.wm_maskIn = tk.Entry(self.fOut, bd=1, textvariable=self.wmMask, width=30)
        self.buttonWMmask = tk.Button(self.fOut, text="...", command=self.get_wmMask)

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
        self.fAna = tk.Frame(self, bd=1, relief="sunken")
        self.lbl4a = tk.Label(
            self.fAna,
            text="Choose the type of analysis to run:",
            font="Helvetica 12 bold",
        )
        self.ana_region = tk.Radiobutton(
            self.fAna,
            text="Region-wise functionnectome",
            variable=self.ana_type,
            value="region",
        )
        self.ana_voxel = tk.Radiobutton(
            self.fAna,
            text="Voxel-wise functionnectome",
            variable=self.ana_type,
            value="voxel",
        )
        self.buttonMask = tk.Button(
            self.fAna, text="Select grey matter mask(s)", command=self.get_masks
        )
        self.lbl4b = tk.Label(
            self.fAna, text="Number of parallel processes:", font="Helvetica 12 bold"
        )
        self.parallel_proc = tk.Spinbox(
            self.fAna,
            from_=1,
            to=os.cpu_count(),
            width=5,
            textvariable=self.nb_parallel_proc,
        )
        self.lbl4c = tk.Label(
            self.fAna, text="Priors are stored as:", font="Helvetica 12 bold"
        )
        self.lbl4d = tk.Label(
            self.fAna, text="Choice of priors:", font="Helvetica 12 bold"
        )
        self.lbl4e = tk.Label(
            self.fAna, text="DL all:", font="Helvetica 12 bold"
        )
        self.priorH5 = tk.Radiobutton(
            self.fAna, text="One HDF5 file", variable=self.priors, value="h5"
        )
        self.priorNii = tk.Radiobutton(
            self.fAna, text="Multiple NIfTI files", variable=self.priors, value="nii"
        )
        self.priorsFileList = list(PRIORS_H5)
        self.h5files = ttk.Combobox(self.fAna,
                                    values=self.priorsFileList,
                                    textvariable=self.priorsChoice,
                                    state="readonly")
        self.ana_type.set("voxel")
        self.h5files.current(self.priorsFileList.index('V1.D.WB - Whole brain, Deterministic (legacy)'))
        self.buttonDL = tk.Button(
            self.fAna, text="Manual download", command=self.manualDLpriors
        )
        self.chkDLall = tk.Checkbutton(self.fAna, variable=self.DLall)
        self.priors.set("h5")

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
        self.ana_region.grid(column=0, row=2, sticky="W")
        self.ana_voxel.grid(column=0, row=3, sticky="W")
        self.buttonMask.grid(column=0, row=4, padx=5, sticky="EW")
        self.lbl4b.grid(column=0, row=5, padx=5, pady=10, sticky="W")
        self.parallel_proc.grid(column=1, row=5, sticky="W")
        self.lbl4c.grid(column=0, row=6, columnspan=2, sticky="W", padx=5, pady=5)
        self.lbl4d.grid(column=1, row=6, columnspan=2, sticky="W", padx=5, pady=5)
        self.priorH5.grid(column=0, row=7, sticky="W")
        self.priorNii.grid(column=0, row=8, sticky="W")
        self.h5files.grid(column=1, row=7, columnspan=3, sticky="WE")
        self.buttonDL.grid(column=1, row=8)
        self.chkDLall.grid(column=3, row=8)
        self.lbl4e.grid(column=2, row=8)

        # Bottom buttons
        self.saveBtn = tk.Button(self, text="Save", command=self.choseFileAndSave)
        self.loadBtn = tk.Button(self, text="Load", command=self.loadSettings)
        self.lauchBtn = tk.Button(self, text="Launch", command=self.launchAna)
        self.quitBtn = tk.Button(self, text="Exit", command=self.destroy)

        self.saveBtn.grid(column=0, row=3, padx=epad, pady=epad)
        self.loadBtn.grid(column=1, row=3, padx=epad, pady=epad)
        self.lauchBtn.grid(column=2, row=3, padx=epad, pady=epad)
        self.quitBtn.grid(column=3, row=3, padx=epad, pady=epad)

    # %% Onpen the window for masks selection
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
        top_mask.title("Masking options")
        top_mask.grab_set()
        self.mask_paths_tmp = self.mask_paths
        nbmask = tk.StringVar()
        nbmask.trace("w", activation_btn)

        nb_mask1 = tk.Radiobutton(
            top_mask, text="One mask per subject", variable=nbmask, value="subjectwise"
        )
        nb_mask2 = tk.Radiobutton(
            top_mask, text="Same mask for all", variable=nbmask, value="common"
        )
        btn_paste = tk.Button(
            top_mask,
            text="Paste paths",
            command=lambda: self.get_files_paste("mask_paths_tmp", self.nbMasks),
        )
        btn_select1 = tk.Button(top_mask, text="Select files", command=choose_masks)
        btn_select2 = tk.Button(top_mask, text="Select the file", command=choose_mask)
        btn_cancel = tk.Button(top_mask, text="Cancel", command=cancel_btn)
        btn_view = tk.Button(top_mask, text="View selected masks", command=view_masks)
        btn_ok = tk.Button(top_mask, text="OK", command=ok_maskBtn)

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

    def activation_h5(self, *args):
        if self.priors.get() == "h5":
            self.h5files.state(statespec=["!disabled"])
            self.buttonDL.config(state="normal")
        elif self.priors.get() == "nii":
            self.h5files.state(statespec=["disabled"])
            self.buttonDL.config(state="disabled")

    def check_if_running(self, rez, mppool, window):
        """Check every second if the multiprocessing 'rez' process in 'mppool' is finished."""
        if not rez.ready():
            window.update()
            window.after(1000, self.check_if_running, rez, mppool, window)
        else:
            mppool.close()
            mppool.join()
            if self.DLall.get():  # meaning the fun.DL_missingH5 function is running
                self.priors_paths = rez.get()
            else:  # meaning the fun.Download_H5 function is running
                self.prior_path_h5 = rez.get()
            window.destroy()

    def manualDLpriors(self):
        '''
        Download the selected priors (if they are not already downloaded)
        '''
        # First check if there is a json file with the priors path
        currentPriors = self.priorsChoice.get()
        pkgPath = os.path.dirname(__file__)
        jsonPath = os.path.join(pkgPath, "priors_paths.json")
        if os.path.exists(jsonPath):
            with open(jsonPath, "r") as jsonP:
                self.priors_paths = json.load(jsonP)
            self.priors_paths = fun.updateOldJson(jsonPath, self.priors_paths)
        missingH5 = fun.find_missingH5(self.priors_paths)
        if self.DLall.get() and not missingH5:
            messagebox.showinfo("Already there", "All the priors have already been previously downloaded.")
        elif (
                not self.DLall.get()
                and currentPriors in self.priors_paths.keys()
                and os.path.exists(self.priors_paths[currentPriors])
        ):
            messagebox.showinfo("Already there", "The selected priors have already been previously downloaded.")
        else:
            priorFiles = [f for f in self.priors_paths.values() if os.path.exists(f)]
            if len(priorFiles):
                self.home = os.path.dirname(priorFiles[-1])
            prior_dirpath_h5 = filedialog.askdirectory(
                initialdir=self.home, parent=self, title="Choose where to save the priors"
            )
            if prior_dirpath_h5:
                pool = mp.Pool(processes=1)
                if self.DLall.get():
                    res = pool.apply_async(fun.DL_missingH5, (self.priors_paths, prior_dirpath_h5))
                else:
                    res = pool.apply_async(fun.Download_H5, (prior_dirpath_h5, currentPriors))
                DLwindow = tk.Toplevel(self)
                DLwindow.title("Downloading")
                DLwindow.grab_set()
                pb = ttk.Progressbar(
                    DLwindow,
                    orient='horizontal',
                    mode='indeterminate',
                    length=280
                )
                infolabel = ttk.Label(DLwindow, text='Downloading in progress... Check the terminal for more details.')
                infolabel.grid(column=0, row=0)
                pb.grid(column=0, row=1)
                pb.start()
                DLwindow.update()
                DLwindow.after(1000, self.check_if_running, res, pool, DLwindow)
                self.wait_window(DLwindow)
                stillMissing = fun.find_missingH5(self.priors_paths)
                if os.path.exists(self.prior_path_h5) and not self.DLall.get():
                    self.priors_paths[currentPriors] = self.prior_path_h5
                    with open(jsonPath, "w") as jsonP:
                        json.dump(self.priors_paths, jsonP)
                elif self.DLall.get() and not stillMissing:
                    with open(jsonPath, "w") as jsonP:
                        json.dump(self.priors_paths, jsonP)
                else:
                    messagebox.showwarning(
                        title="Priors missing",
                        message="The downloaded files cannot be found.\nCheck the folder you chose and retry.",
                        parent=self,
                    )

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
        pasteWin.title("Select the files")
        pasteWin.grab_set()

        paths = getattr(self, paths_varname)

        lblgetfile = tk.Label(pasteWin, text="Paste the paths to all the files here :")
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

        btnCancel = tk.Button(pasteWin, text="Cancel", command=cancel_btn)
        btnCancel.grid(column=0, row=2)

        btnClear = tk.Button(pasteWin, text="Clear", command=clear_btn)
        btnClear.grid(column=1, row=2)

        btnOK = tk.Button(pasteWin, text="OK", command=ok_btn)
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
        settingsTxt = (
            "Output folder:\n"
            "\t" + self.outDir.get() + "\n"
            "Analysis ('voxel' or 'region'):\n"
            "\t" + self.ana_type.get() + "\n"
            "Number of parallel processes:\n"
            "\t" + self.nb_parallel_proc.get() + "\n"
            "Priors stored as ('h5' or 'nii'):\n"
            "\t" + self.priors.get() + "\n"
            "HDF5 priors:\n"
            "\t" + self.priorsChoice.get() + "\n"
            "Position of the subjects ID in their path:\n"
            "\t" + posID + "\n"
            "Mask the output:\n"
            "\t" + outputMask + "\n"
            "Number of subjects:\n"
            "\t" + str(self.nbFiles.get()) + "\n"
            "Number of masks:\n"
            "\t" + str(self.nbMasks.get()) + "\n"
            "Subject's BOLD paths:\n"
            + "".join([subpath + "\n" for subpath in self.bold_paths])
            + "\n"
            "Masks for voxelwise analysis:\n"
            + "".join([subpath + "\n" for subpath in self.mask_paths])
        )
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
