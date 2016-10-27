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

def load_data(start=0., stop=100., t_step=1,
              ins=['gps_1_vel', 'fiber_gyro'], outs=['speed'],
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

    dataset_path = os.path.join(os.environ['DATA_PATH'],
                                'self-driving', 'car_sensors.h5')
    if not os.path.exists(dataset_path):
        raise Exception("Cannot find data: %s" % dataset_path)

    if verbose:
        sys.stdout.write('Loading data from %s...' %
                         os.path.basename(dataset_path))
        sys.stdout.flush()

    f = h5py.File(dataset_path, 'r')

    input_vars = []
    for name in ins:
        if len(f[name][:].shape) == 1:
            input_vars.append(f[name][:][:, None])
        else:
            assert len(f[name][:].shape) == 2
            input_vars.append(f[name][:])

    output_vars = []
    for name in outs:
        if len(f[name][:].shape) == 1:
            output_vars.append(f[name][:][:, None])
        else:
            assert len(f[name][:].shape) == 2
            output_vars.append(f[name][:])

    X = np.nan_to_num(np.concatenate(input_vars, axis=1))
    Y = np.nan_to_num(np.concatenate(output_vars, axis=1))

    start = int((start/100.) * len(X))
    stop = int((stop/100.) * len(X))

    # Select the data points
    X = X[start:stop:t_step,:]
    Y = Y[start:stop:t_step,:]

    f.close()

    if verbose:
        sys.stdout.write('Done.\n')
        print('# of loaded points: %d' % len(X))

    return X, Y

def ecef2enu(x, y, z, x0, y0, z0):
    """Convert ECEF to local coordinates
    """
    lat0, lon0, alt0 = ecef2geodetic(x0, y0, z0)

    u, v, w = x - x0, y - y0, z - z0
    t     =  np.cos(lon0) * u + np.sin(lon0) * v
    East  = -np.sin(lon0) * u + np.cos(lon0) * v
    Up    =  np.cos(lat0) * t + np.sin(lat0) * w
    North = -np.sin(lat0) * t + np.cos(lat0) * w

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
