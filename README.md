# Functionnectome
The `Functionnectome` is a python package which apply the eponym method and combines the functional signal from distant voxels of the classical
fMRI 4D volume using their probabilistic structural relationship given by anatomical priors of the involved brain circuits.

To launch the `Functionnectome`, if it was properly installed with the `pip` command (see **Installation** below), open a termial and run the command `FunctionnectomeGUI` to use the GUI, or `Functionnectome <setting_file.fcntm>` if you already have a settings file ready.

# Documentation
The complete manual is available of the project's GitHub page (https://github.com/NotaCS/Functionnectome) under the name The_Functionnectome_Userguide.pdf 

# System Requirements
## Hardware
The `Functionnectome` can run on a standard computer depending on the type of analysis. A region-wise analysis should run quickly without trouble, but a voxel-wise analysis would require at least 8 (or better, 16) CPUs to run in a sensible amount of time (see the FAQ in the documentation for more details).

## Operating system
The `Functionnectome` is supported for MacOS and Linux. It might work on Windows but has yet to be tested.
The package has been tested on Linux (Centos7) and MacOS. Test on Windows will come later.

## Python Dependencies
The package mostly requires default Python libraries. It also depends on classic scientific libraries:
```
Numpy
Pandas
Nibabel
H5py
```

# Installation
Using command lines:
`pip install git+https://github.com/NotaCS/Functionnectome.git`
or
`python -m pip install git+https://github.com/NotaCS/Functionnectome.git`
It is also possible to manually download the scripts from the GitHub and run them.
More details are available in the manual.

# Run time
The run time may vary a lot depending on many different factors (from hardware to input data).
As a rule of thumb, the region-wise analysis can run on a dataset in under 10 min on a typical desktop computer. Conversely the voxel-wise analysis is not meant for such hardware (it would take more than a day for one dataset), and usually requires a computational grid (or a computer with many cores). More details are available in the manunal's FAQ.

# Using the package
A quick guide and a comprehensive guide with step by step are documented in the manual.

# Using the scripts directly
The main program, **functionnectome.py**, runs the computation. It requires one argument: the path to a text file (with the ".fcntm" extension) containing the properly formated input parameters.
The *.fcntm* file can be easily created using the GUI accompanying the package: **functionnectome_GUI.py**
