"""
Interface for the U.S. Treasury Yield Curve Rates data.
Source: https://www.quandl.com/data/USTREASURY/YIELD-Treasury-Yield-Curve-Rates

DESCRIPTION
These rates are commonly referred to as "Constant Maturity Treasury" rates, or
CMTs. Yields are interpolated by the Treasury from the daily yield curve. This
curve, which relates the yield on a security to its time to maturity is based
on the closing market bid yields on actively traded Treasury securities in the
over-the-counter market. These market yields are calculated from composites of
quotations obtained by the Federal Reserve Bank of New York. The yield values
are read from the yield curve at fixed maturities, currently 1, 3 and 6 months
and 1, 2, 3, 5, 7, 10, 20, and 30 years. This method provides a yield for a 10
year maturity, for example, even if no outstanding security has exactly 10
years remaining to maturity.

Available time series that correspond to fixed times to maturity:
'1 MO', '3 MO', '6 MO',
'1 YR', '2 YR', '3YR', '5 YR', '7 YR' ,'10 YR', '20 YR', '30 YR'.
"""
import os
import sys

import numpy as np
import pandas as pd

import Quandl

def load_data(start=0., stop=100., t_step=1,
              ins=['6 MO', '1 YR', '3 YR', '5 YR'], outs=['10 YR'], verbose=1):
    """Load the U.S. Treasury Yield Curve Rates data.

    Arguments:
    ----------
        t_step : uint (default: 1)
            Take data points t_step days from each other.
        start : float in [0., 100.)
        stop : float in (0., 100.]
        ins : list of str (default: ['3 MO', '6 MO', '1 YR', '3 YR'])
            Names of the fields (times to maturity) to use as inputs.
        outs : list of str (default: ['10 YR'])
            Names of the fields (times to maturity) to use as outputs.
        verbose : uint (default: 1)
    """
    if 'DATA_PATH' not in os.environ:
        raise Exception("Cannot find DATA_PATH variable in the environment. "
                        "DATA_PATH should be the folder that contains "
                        "`quandl/` directory where the data is cached. "
                        "Please export DATA_PATH before using `load_data`.")

    # Create a directory for Quandl data if it does not exist
    quandl_data_path = os.path.join(os.environ['DATA_PATH'], 'quandl')
    if not os.path.isdir(quandl_data_path):
        os.makedirs(quandl_data_path)

    if verbose:
        sys.stdout.write('Loading data...')
        sys.stdout.flush()

    # Load the data
    dataset_path = os.path.join(quandl_data_path, 'USTYCR.pkl')
    if not os.path.exists(dataset_path):
        USTYCR = Quandl.get("USTREASURY/YIELD")
        USTYCR.to_pickle(dataset_path)
    else:
        USTYCR = pd.read_pickle(dataset_path)

    X = USTYCR[ins].as_matrix()
    Y = USTYCR[outs].as_matrix()

    start = int((start/100.) * len(X))
    stop = int((stop/100.) * len(X))

    # Select the data points
    X = X[start:stop:t_step,:]
    Y = Y[start:stop:t_step,:]

    if verbose:
        sys.stdout.write('Done.\n')
        print('# of loaded points: %d' % len(X))

    return X, Y
