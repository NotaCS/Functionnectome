# Functionnectome
The `Functionnectome` is a python package which apply the eponym method and combines the functional signal from distant voxels of the classical
fMRI 4D volume using their probabilistic structural relationship given by anatomical priors of the involved brain circuits.

To launch the `Functionnectome`, if it was properly installed with the `pip` command (see **Installation** below), open a terminal and run the 
command `FunctionnectomeGUI` to use the GUI, or `Functionnectome <setting_file.fcntm>` if you already have a settings file ready.

# WhiteRest
WhiteRest is a tool to explore the potential impact of white matter lesions on resting-state networks. It is installed along the Functionnectome 
and is based on the WhiteRest atlas. More information is available in the WhiteRest manual.

# Documentation
The complete manual is available of the project's GitHub page (https://github.com/NotaCS/Functionnectome) under the name The_Functionnectome_Userguide.pdf 

# System Requirements
## Hardware
The `Functionnectome` can run on a standard computer depending on the type of analysis. A voxel-wise analysis would require at least 4 
(or better, 12 or 16) CPUs to shorten the run-time. A region-wise analysis should run on almost any computer.

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

- `pip install git+https://github.com/NotaCS/Functionnectome.git`

or

- `python -m pip install git+https://github.com/NotaCS/Functionnectome.git`

It is also possible to manually download the scripts from the GitHub and run them.
More details are available in the manual.

# Update
Using command lines:

- `pip install --upgrade git+https://github.com/NotaCS/Functionnectome.git`

or

- `python -m pip install --upgrade git+https://github.com/NotaCS/Functionnectome.git`

# Run time
The run time may vary a lot depending on many different factors (from hardware to input data).
As a rule of thumb, the voxel-wise analysis can run on a dataset in under 10 to 20 min on a typical desktop computer.
The region-wise analysis used to be faster but I'm not sure it is still the case (haven't tested in a while).

# Using the package
After installing the package, enter the `FunctionnectomeGUI` command in the terminal to start using the interface.

A quick guide and a comprehensive guide with step by step are documented in the manual.

# Using the scripts directly
The main program, **functionnectome.py**, runs the computation. It requires one argument: the path to a text file 
(with the ".fcntm" extension) containing the properly formated input parameters.
The *.fcntm* file can be easily created using the GUI accompanying the package: **functionnectome_GUI.py**
