"""
Interface for the self-driving car sensors dataset.

Description of the data:
"times"                 : seconds from start
"fiber_accel"           : m/s**2
"fiber_compass"         : this is a concatenation of x_y_z below
"fiber_compass_x"       : magnetic north (we are not sure actually ;))
"fiber_compass_y"       : orthogonal to magnetic north in car plane
"fiber_compass_z"       : orthogonal to x and y
"fiber_gyro"            : deg/s - roll, pitch, yaw in car-centric frame
"gps_1_pos"             : ECEF coordinates
"gps_1_vel"             : ECEF velocity
"gps_2_pos"             : ECEF coordinates (another GPS sensor)
"gps_2_vel"             : ECEF velocity (another GPS sensor)
"imu_compass"           : same as compass above
"imu_gyro"              : deg/s - roll, pitch, yaw in car-centric frame
"speed"                 : m/s
"speed_abs"             : m/s
"steering_angle"        : deg/m
"velodyne_gps"          : ECEF
"velodyne_imu"          : deg/s
"left_lanes"            : 4 + 4 coefficients of the cubic polynomials (x and y)
"right_lanes"           : 4 + 4 coefficients of the cubic polynomials (x and y)
"radar_leads"           : x-y (front-left) coordinates of a detected vehicle

About the self-driving car:
http://www.bloomberg.com/features/2015-george-hotz-self-driving-car/
"""
import os
import sys
import h5py

import numpy as np
import pandas as pd


def load_data(start=0., stop=100., t_step=1, set_name='full',
              ins=['gps_1_vel', 'fiber_gyro'], outs=['speed'],
              gps_to_enu=False,
              pts_per_lane=7,
              verbose=1):
    """Load the car sensors data.

    Arguments:
    ----------
        t_step : uint
            Take data points t_step apart from each other in time.
        start : float in [0., 100.)
        stop : float in (0., 100.]
        ins : list of str
            Names of the fields to use as inputs.
        outs : list of str
            Names of the fields to use as outputs.
        verbose : uint (default: 1)
    """
    if 'DATA_PATH' not in os.environ:
        raise Exception("Cannot find DATA_PATH variable in the environment. "
                        "DATA_PATH should be the folder that contains "
                        "`self-driving/` directory with car sensors data. "
                        "Please export DATA_PATH before loading the data.")

    datadir = os.path.join(os.environ['DATA_PATH'], 'self-driving')
    dataset_name = 'car_sensors{}.h5'.format(
        '_{}'.format(set_name) if set_name != 'full' else '')
    dataset_path = os.path.join(datadir, dataset_name)
    if not os.path.exists(dataset_path):
        raise Exception("Cannot find data: %s" % dataset_path)

    if verbose:
        print('Loading data from %s...' % os.path.basename(dataset_path))
        sys.stdout.flush()

    f = h5py.File(dataset_path, 'r')

    # Compute the data slice
    N = len(f['times'][:])
    start_t = int((start/100.) * N)
    stop_t = int((stop/100.) * N)
    idx = slice(start_t, stop_t, t_step)

    # Read & preprocess inputs
    input_nnz, input_vars = None, []
    for name in ins:
        X = f[name][idx]
        if gps_to_enu and name.startswith('gps'):
            assert X.shape[1] == 3
            x0, y0, z0 = X[0, 0], X[0, 1], X[0, 2]
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            X = ecef2enu(x, y, z, x0, y0, z0)
        if name.endswith('lanes'):
            if verbose:
                print('...constructing %s' % name)
                sys.stdout.flush()
            X, nnz = construct_lanes(X, pts_per_lane)
            input_nnz = nnz if input_nnz is None \
                        else np.logical_and(input_nnz, nnz)
        if name == 'radar_leads':
            X[X[:, 0] < 0] = np.nan
            X_df = pd.DataFrame(X).fillna(method='ffill')
            X = X_df.values
            # X[X[:, 0] < 0., 0], X[X[:, 1] < 0., 1] = 200., 0.
        if len(X.shape) == 1:
            input_vars.append(X[:, None])
        else:
            assert len(X.shape) == 2
            input_vars.append(X)

    # Read & preprocess targets
    target_nnz, target_vars = None, []
    for name in outs:
        Y = f[name][idx]
        if gps_to_enu and name.startswith('gps'):
            assert Y.shape[1] == 3
            x0, y0, z0 = Y[0, :]
            x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]
            Y = ecef2enu(x, y, z, x0, y0, z0)
        if name.endswith('lanes'):
            if verbose:
                print('...constructing %s' % name)
                sys.stdout.flush()
            Y, nnz = construct_lanes(Y, pts_per_lane)
            target_nnz = nnz if target_nnz is None \
                         else np.logical_and(target_nnz, nnz)
        if name == 'radar_leads':
            Y[Y[:, 0] < 0] = np.nan
            Y_df = pd.DataFrame(Y).fillna(method='ffill')
            Y = Y_df.values
            # Y[Y[:, 0] < 0., 0], Y[Y[:, 1] < 0., 1] = 200., 0.
        if len(Y.shape) == 1:
            target_vars.append(Y[:, None])
        else:
            assert len(Y.shape) == 2
            target_vars.append(Y)

    f.close()

    X = np.nan_to_num(np.concatenate(input_vars, axis=1)) if input_vars \
        else None
    Y = np.nan_to_num(np.concatenate(target_vars, axis=1)) if target_vars \
        else None

    # Make sure inputs and targets have the same indexes
    nnz = None
    if (input_nnz is not None) and (target_nnz is not None):
        nnz = np.logical_and(input_nnz, target_nnz)
    elif input_nnz is not None:
        nnz = input_nnz
    elif target_nnz is not None:
        nnz = target_nnz

    if (X is not None) and (nnz is not None):
        X = X[nnz]
    if (Y is not None) and (nnz is not None):
        Y = Y[nnz]

    if verbose:
        print('Done.')
        print('# of loaded points: %d' % len(X))
        if X is not None:
            print('Inputs shape: %s' % X.shape[1:])
        if Y is not None:
            print('Targets shape: %s' % Y.shape[1:])

    return X, Y


def construct_lanes(data, pts_per_lane):
    poly_x, poly_y = data[:,:4], data[:,4:]
    lane_x = np.vstack(
        [np.polyval(poly_x[t], np.linspace(0, 50, pts_per_lane))
         for t in xrange(poly_x.shape[0])])
    lane_y = np.vstack(
        [np.polyval(poly_y[t], np.linspace(0, 50, pts_per_lane))
         for t in xrange(poly_y.shape[0])])
    lane = np.hstack([lane_x, lane_y])
    nnz = lane_x.sum(axis=1) > 0.
    return lane, nnz


def ecef2enu(x, y, z, x0, y0, z0):
    """Convert ECEF to local coordinates.
    """
    lat0, lon0, alt0 = ecef2geodetic(x0, y0, z0)

    u, v, w = x - x0, y - y0, z - z0
    t     =  np.cos(lon0) * u + np.sin(lon0) * v
    East  = -np.sin(lon0) * u + np.cos(lon0) * v
    North = -np.sin(lat0) * t + np.cos(lat0) * w
    Up    =  np.cos(lat0) * t + np.sin(lat0) * w

    return np.vstack([East, North, Up]).T


def ecef2geodetic(x, y, z):
    """http://www.astro.uni.torun.pl/~kb/Papers/geod/Geod-BG.htm

    This algorithm provides a converging solution to the latitude equation in
    terms of the parametric or reduced latitude form (v).
    This algorithm provides a uniform solution over all latitudes as it does
    not involve division by cos(phi) or sin(phi).
    """
    ell = {}
    ell['a'] = 6378137.
    ell['f'] = 1. / 298.2572235630
    ell['b'] = ell['a'] * (1 - ell['f'])

    ea = ell['a']
    eb = ell['b']
    rad = np.hypot(x, y)
    # Constant required for Latitude equation
    rho = np.arctan2(eb * z, ea * rad)
    #Constant required for latitude equation
    c = (ea**2 - eb**2) / np.hypot(ea * rad, eb * z)
    # Starter for the Newtons Iteration Method
    vnew = np.arctan2(ea * z, eb * rad)
    # Initializing the parametric latitude
    v = 0
    count = 0
    while (v != vnew).any() and count < 5:
        v = vnew.copy()
        # Newtons Method for computing iterations
        vnew = v - ((2 * np.sin(v - rho) - c * np.sin(2 * v)) /
                    (2 * (np.cos(v - rho) - c * np.cos(2 * v))))
        count += 1

    # Computing latitude from the root of the latitude equation
    lat = np.arctan2(ea * np.tan(vnew), eb)
    lon = np.arctan2(y, x)
    alt = ((rad - ea * np.cos(vnew)) * np.cos(lat) +
           (z - eb * np.sin(vnew)) * np.sin(lat))

    return lat, lon, alt
