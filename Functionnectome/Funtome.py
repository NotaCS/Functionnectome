#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanches the Functionnectome purely from command line, without using a .fcntm settings file.

/!\\ If you have spaces in your file paths, put them between quote marks

For example:
    -od /myHome/my data/results
will not work because of the space between 'my' and 'data'.
Instead, do:
    -od '/myHome/my data/results'

"""

import argparse
import sys
import os
from Functionnectome.functionnectome import Functionnectome  # Yes, I know...
from Functionnectome.quickDisco import checkH5


class MyParser(argparse.ArgumentParser):  # Used to diplay the help if no argment is given
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def _build_arg_parser(h5Labels):

    p = MyParser(formatter_class=argparse.RawTextHelpFormatter,
                 description=__doc__)
    # Required argument
    p.add_argument("-od", "--results_dir_root",
                   help="Path to the output folder.",
                   required=True)
    p.add_argument("-at", "--anatype",
                   help='Type of analysis to run, voxelwise or regionwise ("voxel" or "region").',
                   required=True, choices=['voxel', 'region'], default='voxel')
    p.add_argument("-pp", "--nb_of_batchs",
                   help="Number of processes (batchs) to launch in parallel.",
                   required=True, type=int, default=1)
    p.add_argument("-pt", "--prior_type",
                   help='Type of storage for the priors, Nifty or HDF5 ("nii" or "h5").',
                   required=True, choices=['h5', 'nii'], default='h5')
    p.add_argument("-h5", "--priorsH5",
                   help=('Label identifying the HDF5 priors to be used, or path to a custom \n'
                         'HDF5 priors file. (available labels: \n' + ' \n'.join(h5Labels.keys()) + ').'),
                   metavar="PRIORS_LABEL/PATH")
    p.add_argument("-id", "--subIDpos",
                   help='Position, in the bold file path, of the identifier for each input volume.',
                   required=True, type=int)
    p.add_argument("-mo", "--maskOutput",
                   help=("'0' or '1' (should be a bool in fact), whether to remove signal out of"
                         "the brain template on the output functionnectome volume. Default is 1\n"
                         "and should probably always be."
                         "\nOR\n"
                         "Can also be the path to a white matter mask defining the voxels where\n"
                         "the Functionnectome will be computed. This can significantly shorten\n"
                         "the computation time."),
                   required=True)
    p.add_argument("-in", "--subNb",
                   help='Number of input functional files.',
                   required=True, type=int)
    p.add_argument("-mn", "--mask_nb",
                   help=('Number of grey matter mask given. Can be 1 (same mask for all),\n'
                         'or the same as bold_paths (one mask per input volume), or 0 in case\n'
                         'of regionwise analysis. The default is 0.'),
                   required=True, type=int, default=0)
    p.add_argument("-ib", "--bold_paths",
                   help="List of all the input functional files (str).",
                   required=True, nargs='+')
    p.add_argument("-im", "--masks_vox",
                   help="List of the grey matter masks file paths (str).",
                   required=True, nargs='+')
    return p


def main():
    priorsOK, txtH5, h5Labels, priors_paths = checkH5()
    parser = _build_arg_parser(h5Labels)
    args = vars(parser.parse_args())
    if args['nb_of_batchs'] < 1:
        raise ValueError('The number of processes should be equal or more than 1')
    if args['prior_type'] == 'h5':
        if args['priorsH5'] is None:
            raise parser.error(
                "The --priorsH5 (or -h5) argument is missing. It is required when the priors type is h5.")
        if args['priorsH5'] in list(h5Labels.keys()):
            if priorsOK:
                args['priorsH5'] = h5Labels[args['priorsH5']]  # replacing the short label by the long one
            else:
                raise ValueError(txtH5)
        else:
            if os.path.splitext(args['priorsH5'])[-1] == ".h5" and os.path.exists(args['priorsH5']):
                args['optSett'] = True
                args['opt_h5_loc'] = args['priorsH5']
                args['priorsH5'] = list(h5Labels.keys())[0]  # Placeholder
            else:
                raise parser.error('No correct priors label or path were given')
    Functionnectome(**args)


# %%
if __name__ == "__main__":
    main()
