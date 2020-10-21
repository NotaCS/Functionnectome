# Functionnectome

This package contains all the necessary tools to run the Functionnectome method.
The Functionnectome is a method that projects BOLD signal from fMRI dataset onto the brain white matter supporting the related brain function.

The main program, **Functionnectome.py**, runs the computation. It requires one argument, the path to a text file (with the ".fcntm" extension) containing the properly formated input parameters.
The *.fcntm* file can be easily created using the GUI accompanying the package: **functionnectome_GUI.py**

Additionally, a *makeHDF5prior.py* script is also provided. It can be used to generate an HDF5 file with your custom priors.
(I still have to code a program to ease the creation of such custom priors, though...)

For more details, refer to the documentation file.
