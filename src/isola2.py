#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (C) 2021 Triantafyllis Nikolaos

#    This file is part of Gisola.

#    Gisola is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, 
#    or any later version.

#    Gisola is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with Gisola.  If not, see <https://www.gnu.org/licenses/>.

import math, os, os.path, operator
from scipy.fft import irfft 
from statistics import median
import numpy as np, multiprocessing
from collections import OrderedDict
import shutil, subprocess, logging, glob
from obspy.geodetics.base import gps2dist_azimuth # for fast run: sudo -H pip3 install geographiclib
from obspy.core.event.source import FocalMechanism,  MomentTensor, NodalPlanes, \
                                    NodalPlane, PrincipalAxes, Axis, Tensor
from obspy.core.event.base import WaveformStreamID, DataUsed, CreationInfo
from obspy import read, UTCDateTime
from obspy.core.event.event import Event
from obspy.core.event.origin import Origin, OriginQuality
from obspy.core.event.magnitude import Magnitude
from obspy.geodetics import kilometers2degrees
from obspy.core import AttribDict
#import geo.sphere
# local
import config, event, modules.kagan
from pathlib import Path
import numpy as np
from scipy.signal import butter, lfilter
from obspy import read
# Keep Numba's JIT cache on the Linux filesystem so warmup is not
# recompiled on every run (DrvFs /mnt/c breaks cache persistence).
os.environ.setdefault("NUMBA_CACHE_DIR",
                      os.path.join(os.path.expanduser("~"), ".numba_cache"))
from numba import njit, prange, cuda, float64
import concurrent.futures
from functools import partial
import time
from threadpoolctl import threadpool_info
import pprint
import numba
import sys
NUM_OF_TIME_SAMPLES = 1024

# ---- bootstrap / plotting support --------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')          # figures are saved to files, never shown
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

try:
    from pyrocko import moment_tensor as pmt
    from pyrocko.plot import hudson, beachball
    HAVE_PYROCKO = True
except ImportError:
    HAVE_PYROCKO = False

# filled by run_python_inversion(), consumed by getBootstrapping()
_BOOT_CTX = None
_SUMMARY  = None   # filled by getBootstrapping(), printed by gisolaBootstrap.py
# filled by getBootstrapping(), consumed by renderSite()
_BOOT_STATS = None
# ------------------------------------------------------------------------

def kaganCalc(x):
    return modules.kagan.get_kagan_angle(*x[0:3], *x[3:6])

def cart2earth(lat,lon,x,y):
    """
    BSD 3-Clause License
    Copyright (c) 2017, Juno Inc.
    All rights reserved.
    https://github.com/gojuno/geo-py/blob/master/geo/sphere.py
    Calculate lat and lon from origin lat lon and cart position
    """
    def destination(point, distance, bearing):
        '''
            Given a start point, initial bearing, and distance, this will
            calculate the destina?tion point and final bearing travelling
            along a (shortest distance) great circle arc.

            (see http://www.movable-type.co.uk/scripts/latlong.htm)
        '''
        EARTH_MEAN_RADIUS = 6371008.8

        lon1, lat1 = (math.radians(coord) for coord in point)
        radians_bearing = math.radians(bearing)

        delta = distance / EARTH_MEAN_RADIUS

        lat2 = math.asin(
            math.sin(lat1)*math.cos(delta) +
            math.cos(lat1)*math.sin(delta)*math.cos(radians_bearing)
        )
        numerator = math.sin(radians_bearing) * math.sin(delta) * math.cos(lat1)
        denominator = math.cos(delta) - math.sin(lat1) * math.sin(lat2)

        lon2 = lon1 + math.atan2(numerator, denominator)

        lon2_deg = (math.degrees(lon2) + 540) % 360 - 180
        lat2_deg = math.degrees(lat2)

        return (lon2_deg, lat2_deg)

    dist=math.sqrt(x**2+y**2)*1000
    try:
        if x==0 and y==0:
            azm=0
        elif y>=0 and x>=0:
            azm=math.atan(x/y)
        elif y<0 and x>=0:
            azm=180-math.atan(x/abs(y))
        elif y<=0 and x<=0:
            azm=180+math.atan(abs(x)/abs(y))
        elif y>0 and x<0:
            azm=270+math.atan(abs(y)/abs(x))
    except ZeroDivisionError:
        azm=90 if x>0 else -90

    lon2,lat2=destination((lon,lat), dist, azm)
    return round(lat2, 4), round(lon2, 4)


def calculateVariance(observed, synthetic, tl):
    """
    Calculates variance reduction of stations' streams
    """
    with np.errstate(divide='raise'):
        try:
            dt = round((tl/float(NUM_OF_TIME_SAMPLES)),4)
            obs = np.array(observed)
            syn = np.array(synthetic)

            ds = obs-syn
            dsn = (np.linalg.norm(ds)**2)*dt
            d = (np.linalg.norm(obs)**2)*dt

            return round(1-(dsn/float(d)),2)
        except FloatingPointError:
            return None

def createSources():
    """
    Creating the Source files (source.dat)
    """
    # for each accepted grid rule, perform the following actions
    for i,rule in enumerate(config.gridRules):

        # create respective source dir
        os.makedirs(os.path.join(config.sourcedir,'grid'+str(i)))

        # calculate source points
        # for x,y (y=x) distance
        x=[]
        for drange in rule['Distance']:
            x=np.append(x,np.unique(np.append(-np.arange(*drange), \
            np.arange(*drange))))

        # for z depth
        z=[]
        for zrange in rule['Depth']:    
            z=np.append(z,np.unique(np.append(-np.arange(*zrange), \
            np.arange(*zrange))))
       
        # use event's depth as offset
        z+=round(config.org.depth/1000,1) # km
        # keep only >= 1 km 
        z=z[z>=1.0]

        # all possible source points
        mesh=np.array(np.meshgrid(x,x,z)).T.reshape(-1, 3)

        # write one file per chunk
        for k in range(math.ceil(len(mesh)/config.cfg['Green']['MaxSources'])):

            # write chunck size in text buffer and then in file
            # with the step of chuck size (=MaxSources)
            text=""
            for j,elem in enumerate(mesh[k*config.cfg['Green']['MaxSources']: \
            (k+1)*config.cfg['Green']['MaxSources']]):
                text+='{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                k*config.cfg['Green']['MaxSources']+j+1, *elem)
            with open(os.path.join(config.sourcedir,'grid'+str(i),\
            'source'+str(k)+'.dat'), 'w') as _:
                _.write(text)

        config.logger.info(('Grid index: {}\nCalculated source points: {}\n' + \
        'Dispatched in {} source file(s)').format(i, \
        len(mesh), k+1))

def createCrustals():
    """
    Copying the Crustal Model files (crustal.dat)
    """
    # create respective source dir
    os.makedirs(config.crustaldir)

    for i,rule in enumerate(config.crustalRules):
        try:
            config.logger.info(('Crustal model Index: {}\nCopying ' + \
            'crustal file: {} to {} directory').format(i, \
            rule['Filepath'], config.crustaldir))
            shutil.copy(rule['Filepath'], os.path.join(config.crustaldir, \
            'crustal'+str(i)+'.dat'))
        except:
            config.logger.info('Crustal model file: {} not found'.format( \
            rule['Filepath']))
            config.logger.info('Continue with next crustal model, if any')
            continue

def createGrdat():
    """
    Creating the Greens Function configuration (grdat.hed)
    """
    maxdist=max([rule[1] for rule in config.distRules])
    xl=2000000 if maxdist<=100 else int(np.ceil(20*1000*maxdist))
    # max freq used for calculation
    maxfreq=2.5*max([rule[3] for rule in config.freqRules])

    # create respective source dir
    os.makedirs(config.grdatdir)

    for i,tl in enumerate(config.windowRules):
        nfreq=np.ceil(tl*maxfreq)
        text=('&input\nnfreq={:n}\ntl={}\naw=1.0\nxl={}\nikmax=100000\n' + \
        'uconv=0.1E-03\nfref=1.\n/end\n').format(nfreq,tl,xl)
 
        with open(os.path.join(config.grdatdir,\
        'grdat'+str(i)+'.hed'), 'w') as _:
            _.write(text)

        config.logger.info('Greens\' Functions configuration index: {}'.format(i))

def createStations():
    """
    Creating the necessary Greens Function configuration (station.dat)
    """
    text=''

    lstations=[]
    for i,sta in enumerate(list(set([tr.stats.station for tr in config.st]))):
        distance, azimuth, _ = gps2dist_azimuth(config.org.latitude, \
        config.org.longitude, config.inv.select(station=sta)[0][0].latitude, \
        config.inv.select(station=sta)[0][0].longitude)

        lstations.append([distance,distance*math.cos(math.radians(azimuth))/1000.0, \
        distance*math.sin(math.radians(azimuth))/1000.0,0,sta])

    # sort by distance and write the necessary info
    for i,sta in enumerate(sorted(lstations, key=operator.itemgetter(0))):
        text+='{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(i+1,*sta[1:])
    
    with open(os.path.join(config.workdir,'station.dat'), 'w') as _:
        _.write(text)

@config.time
def calculateGreens():
    """
    Calculating Greens Function (gr.hes)
    """
    def getIdx(text):
        return ''.join(list(filter(str.isdigit, text)))

    # create respective source dir
    os.makedirs(config.greendir)

    for crustal in sorted(os.listdir(config.crustaldir)):

        for grdat in sorted(os.listdir(config.grdatdir)):

            for grid in sorted(os.listdir(config.sourcedir)):
                for source in sorted(os.listdir(os.path.join(\
                config.sourcedir,grid))):
                    try:
                        grhes='gr.{}.{}.{}.{}.hes'.format(getIdx(crustal),
                                                      getIdx(grdat),
                                                      getIdx(grid),
                                                      getIdx(source))

                        command='{} {} {} {} {} {}\n'.format(\
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)),config.cfg['Green']['ExePath']),
                                        os.path.join('station.dat'),
                                        os.path.join('crustals', crustal),
                                        os.path.join('grdat', grdat),
                                        os.path.join('sources', grid, source),
                                        os.path.join('greens', grhes))

                        proc=subprocess.Popen(command, cwd=config.workdir, \
                        shell=True, universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        out,err=proc.communicate()
                        config.logger.info(err)
                        config.logger.info(out)
                    except:
                        config.logger.info('It could not create {}'.format(grhes))
                        continue
 
def createInpinv():
    """
    Creating Inversions configuration files (inpinv)
    """
    os.makedirs(config.inpinvdir)

    for i,srule in enumerate(config.shiftRules):
        text='{} {} {}'.format(*srule)
        with open(os.path.join(config.inpinvdir, \
        'inpinv'+str(i)+'.dat'), 'w') as _:
            _.write(text)

def createAllstat():
    """
    Creating Inversions configuration files (inpinv)
    """
    # create respective source dir
    os.makedirs(config.allstatdir)

    # read all stations from station.dat file
    with open(os.path.join(config.workdir,'station.dat'), 'r') as _:
        stations=_.readlines()

    # create allstat dir
    for i,rule in enumerate(config.freqRules):
        text=''
        for stat in stations:
            sta=stat.split()
            _st=config.st.select(station=sta[4])

            if _st:
                text+='{} {} {} {} {} {}\n'.format(sta[4],
                      1 if _st.count() else 0,
                      1 if _st.select(component='N') else 0,
                      1 if _st.select(component='E') else 0,
                      1 if _st.select(component='Z') else 0,
                      ' '.join(map(str, rule)))

            else:
                text+='{} {} {} {} {} {}\n'.format(sta[4], 0, \
                      0, 0, 0, ' '.join(map(str, rule)))


            with open(os.path.join(config.allstatdir, \
            'allstat'+str(i)+'.dat'), 'w') as _:
                _.write(text[:-1])

def createRaw():
    """
    Creates raw file of data; contains 4 columns: time, N, E, Z data
    if there's less than 3 components data, it fills with ajacent data
    (dummy data) in order ISOLA to work
    """
    # create respective source dir
    os.makedirs(config.rawdir)

    # read all stations from station.dat file
    with open(os.path.join(config.workdir,'station.dat'), 'r') as _:
        stations=_.readlines()

    # break sts in unique triplets
    sts=list(set([(tr.stats.station, tr.stats.location, \
    tr.stats.channel[:-1]) for tr in config.st])) 

    # create rawfiles for each accepted tl
    for i,tl in enumerate(config.windowRules):

        # one dir for each tl found
        os.makedirs(os.path.join(config.rawdir,str(i)))

        _st=config.st.copy()
        # downsampling at known frequency at NUM_OF_TIME_SAMPLES elements
        _st.resample(NUM_OF_TIME_SAMPLES/tl)
        # be sure than no point exceeds number
        for tr in _st:
            tr.data=tr.data[:NUM_OF_TIME_SAMPLES]

        for stat in stations:
            _st2=_st.select(station=stat.split()[4])

            text=""
            if _st2.count():
                for k,time in enumerate(_st2[0].times()):
                    text+='{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(time, 
                          _st2.select(component='N')[0].data[k] if _st2.select(component='N') else 0,
                          _st2.select(component='E')[0].data[k] if _st2.select(component='E') else 0,
                          _st2.select(component='Z')[0].data[k] if _st2.select(component='Z') else 0
                          )

            else: # dummy data
                for _ in range(NUM_OF_TIME_SAMPLES):
                    text+='{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(0, 0, 0, 0)

            with open(os.path.join(config.rawdir,str(i),stat.split()[4].upper()+'raw.dat'), 'w') as _:
                _.write(text)

@config.time
def calculateInversions():

    # create respective source dir
    os.makedirs(os.path.join(config.inversiondir))

    for allstat in sorted(os.listdir(config.allstatdir)):

        for inpinv in sorted(os.listdir(config.inpinvdir)):

            for grhes in sorted(os.listdir(config.greendir)):
                _, icrustal, igrdat, igrid, isource, _ = grhes.split('.')

                invdir='{}.{}.{}.{}.{}.{}'.format(allstat.split('.')[0][7:], 
                       inpinv.split('.')[0][6:], icrustal, igrdat, igrid, isource)

                os.makedirs(os.path.join(config.inversiondir,invdir))

                command='{} {} {} {} {} {} {} {} {} {} \n'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)),config.cfg['Inversion']['ExePath']),
                            os.path.join('..', '..','allstat', allstat),
                            os.path.join('..', '..','inpinv', inpinv),
                            os.path.join('..', '..','grdat', 'grdat'+igrdat+'.hed'),
                            os.path.join('..', '..','greens', grhes),
                            os.path.join('..', '..','crustals', 'crustal'+icrustal+'.dat'),
                            os.path.join('..', '..','sources', 'grid'+igrid, 'source'+isource+'.dat'),
                            os.path.join('..', '..','station.dat'),
                            os.path.join('..', '..','raw', igrdat),
                            str(config.get_keyinv(config.cfg, origin_data=config.org))
                            )

                proc=subprocess.Popen(command, cwd=os.path.join(config.inversiondir, invdir), shell=True, universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out,err=proc.communicate()
                print(command)
                config.logger.info(err)
                config.logger.info(out)


@njit(fastmath=True, cache=True)
def cart2trendplunge(v):
    x, y, z = v
    if z < 0:
        x, y, z = -x, -y, -z
    trend = np.degrees(np.arctan2(y, x)) % 360.0
    plunge = np.degrees(np.arcsin(z))
    return trend, plunge


@njit(fastmath=True, cache=True)
def angles(n, s):
    EPS = 1e-3

    n0, n1, n2 = n[0], n[1], n[2]
    s0, s1, s2 = s[0], s[1], s[2]

    if n2 > 0:
        n0, n1, n2 = -n0, -n1, -n2
        s0, s1, s2 = -s0, -s1, -s2

    dip = np.degrees(np.arccos(-n2))

    if abs(abs(n2) - 1.0) < EPS:
        strike = np.degrees(np.arctan2(s1, s0)) % 360.0
        return round(strike), round(dip), 0.0

    sdi = 1.0 / np.sqrt(1.0 - n2 * n2)

    strike = np.degrees(np.arctan2(-n0 * sdi, n1 * sdi)) % 360.0

    CF = np.cos(np.radians(strike))
    SF = np.sin(np.radians(strike))

    rake = np.degrees(
        np.arctan2((s0 * SF - s1 * CF) / (-n2),
                   s0 * CF + s1 * SF)
    )

    return strike, dip, rake


@njit(fastmath=True, cache=True)
def build_moment_tensor(A):
    M = np.empty((3, 3))

    A1, A2, A3, A4, A5, A6 = A

    M[0, 0] = -A4 + A6
    M[1, 1] = -A5 + A6
    M[2, 2] = A4 + A5 + A6

    M[0, 1] = A1
    M[0, 2] = A2
    M[1, 2] = -A3

    M[1, 0] = M[0, 1]
    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]

    return M


@njit(fastmath=True, cache=True)
def normalize_moment_tensor(M):
    M0 = np.sqrt(
        0.5 * (M[0,0]**2 + M[1,1]**2 + M[2,2]**2)
        + M[0,1]**2 + M[0,2]**2 + M[1,2]**2
    )

    if M0 < 1e-20:
        return M, M0, False

    factor = 1.0 / (M0 * np.sqrt(2.0))

    for i in range(3):
        for j in range(3):
            M[i, j] *= factor

    return M, M0, True


@njit(fastmath=True, cache=True)
def eigendecomp_sorted(M):
    vals, vecs = np.linalg.eigh(M)

    # sort descending
    for i in range(2):
        for j in range(i+1, 3):
            if vals[i] < vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
                for k in range(3):
                    vecs[k, i], vecs[k, j] = vecs[k, j], vecs[k, i]

    # right-handed system
    cross = np.array([
        vecs[1,0]*vecs[2,1] - vecs[2,0]*vecs[1,1],
        vecs[2,0]*vecs[0,1] - vecs[0,0]*vecs[2,1],
        vecs[0,0]*vecs[1,1] - vecs[1,0]*vecs[0,1]
    ])

    dot = cross[0]*vecs[0,2] + cross[1]*vecs[1,2] + cross[2]*vecs[2,2]

    if dot < 0:
        for i in range(3):
            vecs[i,2] *= -1

    return vals, vecs


@njit(fastmath=True, cache=True)
def decompose(vals):
    AMV = vals[0] + vals[1] + vals[2]

    EN1 = np.empty(3)
    EN1MAX = 0.0
    EN1MIN = 1e30

    for i in range(3):
        EN1[i] = vals[i] - AMV / 3.0
        absval = abs(EN1[i])
        if absval > EN1MAX:
            EN1MAX = absval
        if absval < EN1MIN:
            EN1MIN = absval

    ISO = abs(AMV) / max(abs(vals[0]), abs(vals[1]), abs(vals[2])) / 3.0 * 100.0
    if AMV < 0:
        ISO *= -1

    if EN1MAX < 1e-20:
        return ISO, 0.0, 0.0

    EPS = -EN1MIN / abs(EN1MAX)

    CLVD = 2.0 * abs(EPS) * (100.0 - abs(ISO))
    DC = 100.0 - abs(ISO) - CLVD

    return ISO, CLVD, DC


# -----------------------------
# Main function
# -----------------------------

@njit(parallel=True, fastmath=True, cache=True) 
def silsub(moment_tensors):

    n_sources = moment_tensors.shape[0]

    plane1_out = np.zeros((n_sources, 3))
    plane2_out = np.zeros((n_sources, 3))
    M0_out = np.zeros(n_sources)
    Mw_out = np.zeros(n_sources)
    DC_out = np.zeros(n_sources)
    CLVD_out = np.zeros(n_sources)
    ISO_out = np.zeros(n_sources)
    P_axes_out = np.zeros((n_sources, 2))
    T_axes_out = np.zeros((n_sources, 2))
    B_axes_out = np.zeros((n_sources, 2))
    MT_out = np.zeros((n_sources, 3, 3))

    for k in prange(n_sources):

        A = np.zeros(6)
        for i in range(moment_tensors.shape[1]):
            A[i] = moment_tensors[k, i]

        M = build_moment_tensor(A)
        MT_out[k, :, :] = M
        M, M0, ok = normalize_moment_tensor(M)
        if not ok:
            continue

        Mw = (2.0 / 3.0) * np.log10(M0) - 6.0333

        vals, vecs = eigendecomp_sorted(M)

        ISO, CLVD, DC = decompose(vals)

        T = vecs[:, 0]
        P = vecs[:, 2]

        n1 = (T + P) / np.sqrt(2.0)
        n2 = (T - P) / np.sqrt(2.0)
        s1 = (T - P) / np.sqrt(2.0)
        s2 = (T + P) / np.sqrt(2.0)

        plane1_out[k, :] = angles(n1, s1)
        plane2_out[k, :] = angles(n2, s2)

        M0_out[k] = M0
        Mw_out[k] = Mw
        DC_out[k] = DC
        CLVD_out[k] = CLVD
        ISO_out[k] = ISO

        P_axes_out[k, :] = cart2trendplunge(P)
        T_axes_out[k, :] = cart2trendplunge(T)

        B = np.array([
            P[1]*T[2] - P[2]*T[1],
            P[2]*T[0] - P[0]*T[2],
            P[0]*T[1] - P[1]*T[0]
        ])

        B_axes_out[k, :] = cart2trendplunge(B)

    return (
        plane1_out, plane2_out,
        M0_out, Mw_out,
        DC_out, CLVD_out, ISO_out,
        P_axes_out, T_axes_out, B_axes_out,MT_out
    )


def load_hes_optimized(file_path, nfreq, nr):

    # Initialize Byte info to read .hes files
    BYTES_PER_COMPLEX = 16
    RECORDS_PER_BLOCK = 3
    CHANNELS_PER_RECORD = 6
    MARKER_SIZE = 4

    # 1. Read the entire file as bytes
    with open(file_path, "rb") as f:
        raw_data = f.read()
    
    # 2. Convert to a 1D array of uint8 or similar to handle byte-offsets
    # Alternatively, if markers are consistent, we can calculate the total block size
    # Total size of one "set" of 18 channels:
    # (Marker + 6*Complex + Marker) * 3
    single_record_full_size = MARKER_SIZE + (CHANNELS_PER_RECORD * BYTES_PER_COMPLEX) + MARKER_SIZE
    total_set_size = single_record_full_size * RECORDS_PER_BLOCK
    
   
    # Calculate how many bytes to skip to get to the next record's data
    dt = np.dtype([
        ('m1', 'i4'), 
        ('data', 'c16', (6,)), 
        ('m2', 'i4')
    ])
    
    # View the whole file as a series of Fortran records
    # This happens at C-speed, no Python loops!
    clean_view = np.frombuffer(raw_data, dtype=dt)['data'] # Shape: (N_total_records, 6)
    
    # =====================================================
    # AUTO-DETECT NUMBER OF SOURCES
    # =====================================================

    components = 6

    expected_per_source = nfreq * nr * RECORDS_PER_BLOCK

    total_records = clean_view.shape[0]

    if total_records % expected_per_source != 0:
        raise ValueError(
            f"Invalid HES file structure:\n"
            f"{total_records=} not divisible by "
            f"{expected_per_source=}"
        )

    ns_actual = total_records // expected_per_source

    # 4. Reshape and Reorganize
    # Current shape of clean_view: (ns * nfreq * nr * 3, 6)
    # We want to group the '3' records into the 18 channels
    reorganized = clean_view.reshape(ns_actual, nfreq, nr, 3, 6)
    
    # Transpose to your desired shape: (nfreq, channels, nr, ns)
    # The '3' and '6' combine to form the 18 channels
    final_data = reorganized.transpose(1, 3, 4, 2, 0).reshape(nfreq, 18, nr, ns_actual)
    
    return final_data


def build_G_transposed(U_all, CC, b, a, Dt, keep_pairs, components, NFFT, n_workers=None):
    """
    Builds the Greens matrix directly in the (n_sources, n_total, n_comp)
    layout that the inversion kernel needs.

    Numerically IDENTICAL to the previous pipeline
        irfft -> *CC -> lfilter -> cumsum -> *Dt -> reshape/transpose -> mask
    because every operation acts independently per (channel, source) column,
    so processing in source-chunks and dropping unused channels/components
    BEFORE the FFT cannot change any kept value.

    Speed comes from:
      - slicing the moment-tensor components 6 -> `components` and removed
        channels BEFORE the FFT/filter (less data through every stage)
      - chunking over sources + ThreadPoolExecutor (irfft/lfilter/cumsum
        release the GIL, so chunks run on multiple cores)
      - writing straight into the final kernel layout (no giant transpose +
        ascontiguousarray copy, no reorder copy inside the kernel)
      - peak memory drops from ~3x the array size to ~1x + small chunks
    """
    nfreq, _, nr, ns_total = U_all.shape
    n_kept = len(keep_pairs)

    # view: (nfreq, 3 records(N/E/Z), components, nr, ns)
    # drops the unused MT components before any heavy work
    U5 = U_all.reshape(nfreq, 3, 6, nr, ns_total)[:, :, :components]

    G = np.empty((ns_total, n_kept * NFFT, components), dtype=np.float64)

    if n_workers is None:
        n_workers = max(1, min(8, (os.cpu_count() or 1)))
    n_chunks = max(1, min(ns_total, 4 * n_workers))
    edges = np.linspace(0, ns_total, n_chunks + 1).astype(np.int64)

    def process(s0, s1):
        # exact same operation sequence as before, on a source-slice
        ts = irfft(U5[..., s0:s1], n=NFFT, axis=0) * NFFT   # (NFFT,3,comp,nr,nc)
        ts *= CC[:, None, None, None, None]
        ts = lfilter(b, a, ts, axis=0)
        np.cumsum(ts, axis=0, out=ts)
        ts *= Dt
        for row, (sta, rec) in enumerate(keep_pairs):
            # (NFFT, comp, nc) -> (nc, NFFT, comp)
            G[s0:s1, row * NFFT:(row + 1) * NFFT, :] = \
                ts[:, rec, :, sta, :].transpose(2, 0, 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process, int(edges[i]), int(edges[i + 1]))
                for i in range(n_chunks) if edges[i] < edges[i + 1]]
        for f in futs:
            f.result()   # re-raise any worker exception

    return G


@njit(parallel=True, fastmath=True, cache=True)
def compute_inversion_numbaCpu(G, obs_data, Dt, shifts_array, num_stations):
    """
    G: (n_sources, n_total, n_comp), C-contiguous -- already in the
    cache-friendly layout (built by build_G_transposed), so the old
    reorder copy that allocated a second multi-GB array is gone.

    Outputs identical to the previous version. Changes:
      - removed the G_in -> G reorder loop (input already transposed)
      - the per-source solve loop is now prange (it was serial); each k
        is independent, so results are unchanged
      - the global best is taken with an argmax over best_vr_per_source
        after the loops (same maximum as the old running comparison)
    """
    MULTIP = 1e20

    n_sources, n_total, n_comp = G.shape
    n_stations = n_total // 1024

    scale = MULTIP * Dt

    # ------------------------------------------------------------
    # Precompute R = GtG
    # ------------------------------------------------------------
    R = np.zeros((n_sources, n_comp, n_comp))

    for k in prange(n_sources):
        Gk = G[k]
        for i in range(n_total):
            for a in range(n_comp):
                for b in range(n_comp):
                    R[k, a, b] += Gk[i, a] * Gk[i, b]

    R *= scale

    # ------------------------------------------------------------
    # Singular values and condition numbers per source
    # ------------------------------------------------------------
    svals = np.zeros((n_sources, n_comp))
    cond_nums = np.zeros(n_sources)
    VARDAT = 2.0e-12  # fixed prior variance (matches Fortran ISOLA15)
    for k in prange(n_sources):
        eigvals = np.linalg.eigvalsh(R[k])
        svals_k = np.sqrt(eigvals / VARDAT) / 1e10
        svals[k, :] = svals_k
        cond_nums[k] = svals_k.max() / svals_k.min()

    # ------------------------------------------------------------
    # Data energy
    # ------------------------------------------------------------
    rr = np.sum(obs_data**2) * Dt

    # ------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------
    g = np.empty((n_sources, n_comp))

    best_vr_per_source = np.full(n_sources, -1.0)
    best_corr_per_source = np.full(n_sources, -1.0)
    best_shift_per_source = np.zeros(n_sources, dtype=np.int64)
    best_mech_per_source = np.zeros((n_sources, n_comp))

    n_shifts = shifts_array.shape[0]
    vr_mat = np.full((n_sources, n_shifts), -1.0)   # vr for every (source, shift)

    # ============================================================
    # Loop shifts
    # ============================================================
    for si in range(n_shifts):
        s = shifts_array[si]

        # --------------------------------------------------------
        # Compute g = Gt d  (zeroing fused into the same loop)
        # --------------------------------------------------------
        for k in prange(n_sources):
            for n in range(n_comp):
                g[k, n] = 0.0

            Gk = G[k]
            for stat in range(n_stations):
                base = stat * 1024
                for t in range(1024):
                    gf_idx = t - s
                    if 0 <= gf_idx < 1024:
                        d = obs_data[base + t]
                        for n in range(n_comp):
                            g[k, n] += Gk[base + gf_idx, n] * d

        g *= scale

        # --------------------------------------------------------
        # Solve per source -- now parallel (each k independent)
        # --------------------------------------------------------
        for k in prange(n_sources):

            a_k = np.linalg.solve(R[k], g[k])

            sum1 = 0.0
            for n in range(n_comp):
                sum1 += a_k[n] * g[k, n]

            vr = ((sum1 / MULTIP) / rr) * 100.0
            vr_mat[k, si] = vr

            if vr > best_vr_per_source[k]:
                best_vr_per_source[k] = vr
                best_shift_per_source[k] = s
                best_corr_per_source[k] = np.sqrt(max(vr, 0.0) / 100.0)
                for n in range(n_comp):
                    best_mech_per_source[k, n] = a_k[n]

    # ------------------------------------------------------------
    # Global best (same value as the old running comparison)
    # ------------------------------------------------------------
    best_idx = 0
    bv = best_vr_per_source[0]
    for k in range(1, n_sources):
        if best_vr_per_source[k] > bv:
            bv = best_vr_per_source[k]
            best_idx = k

    return (
        best_shift_per_source,
        best_mech_per_source,
        best_idx,
        best_vr_per_source,
        best_corr_per_source,
        svals,
        cond_nums,
        vr_mat
    )



@njit(parallel=True, fastmath=True, cache=True)
def compute_bootstraping_numbaCpu(G, obs_data, W, Dt, shifts_array, Coord):
    """
    Bootstrap inversion with station weights W.
    G: (n_sources, n_total, n_comp) - same layout as the inversion kernel.
    Identical results to the GISOLA2.0_TimeTest2 version (which reordered
    G internally); reorder removed, per-source solve loop parallelized.
    """
    MULTIP = 1e20

    n_sources, n_total, n_comp = G.shape
    n_stations = n_total // 1024

    # ------------------------------------------------------------
    # Precompute weighted R[k] = Gk^T W Gk
    # ------------------------------------------------------------
    R = np.zeros((n_sources, n_comp, n_comp))

    for k in prange(n_sources):
        Gk = G[k]
        for i in range(n_total):
            w = W[i]
            if w == 0.0:
                continue
            for a in range(n_comp):
                for b in range(n_comp):
                    R[k, a, b] += Gk[i, a] * Gk[i, b] * w

    R *= Dt * MULTIP

    # ------------------------------------------------------------
    # Weighted data norm d^T W d
    # ------------------------------------------------------------
    rr = 0.0
    for i in range(n_total):
        rr += obs_data[i] * obs_data[i] * W[i]
    rr *= Dt

    # ------------------------------------------------------------
    # Per-source best storage
    # ------------------------------------------------------------
    best_vr_per_source = np.full(n_sources, -1.0)
    best_shift_per_source = np.zeros(n_sources, dtype=np.int64)
    best_mech_per_source = np.zeros((n_sources, n_comp))

    g = np.empty((n_sources, n_comp))

    # ============================================================
    # Loop over shifts
    # ============================================================
    for s in shifts_array:

        t_start = 0 if s < 0 else s
        t_end = 1024 if s >= 0 else 1024 + s

        # ----- weighted g[k] = G^T W d (zeroing fused) -----
        for k in prange(n_sources):
            for n in range(n_comp):
                g[k, n] = 0.0

            Gk = G[k]
            for stat in range(n_stations):
                base = stat * 1024
                for t in range(t_start, t_end):
                    obs_idx = base + t
                    gf_idx = base + (t - s)
                    w = W[obs_idx]
                    if w == 0.0:
                        continue
                    val = obs_data[obs_idx]
                    for n in range(n_comp):
                        g[k, n] += Gk[gf_idx, n] * val * w

        g *= Dt * MULTIP

        # ----- solve per source (parallel, each k independent) -----
        for k in prange(n_sources):
            a_k = np.linalg.solve(R[k], g[k])

            sum1 = 0.0
            for n in range(n_comp):
                sum1 += a_k[n] * g[k, n]

            vr = ((sum1 / MULTIP) / rr) * 100.0

            if vr > best_vr_per_source[k]:
                best_vr_per_source[k] = vr
                best_shift_per_source[k] = s
                for n in range(n_comp):
                    best_mech_per_source[k, n] = a_k[n]

    # ============================================================
    # Select top of sources (same rule as TimeTest2)
    # ============================================================
    n_keep = int(0.2 * n_sources)
    if n_keep < 1:
        n_keep = 1

    idx_sorted = np.argsort(best_vr_per_source)[::-1]
    idx_keep = idx_sorted[:n_keep]

    return (
        best_shift_per_source[idx_keep],
        idx_keep,
        best_mech_per_source[idx_keep],
        best_vr_per_source[idx_keep],
        Coord[idx_keep],
    )


def numba_warmup():


    rng = np.random.default_rng()

    start = time.time()

    # -----------------------------------------------------
    # Small synthetic problem (forces JIT compilation)
    # -----------------------------------------------------
    NSTATIONS = 2
    WIN = 1024

    G_small = rng.random((4, WIN * NSTATIONS, 5), dtype=np.float64)
    Dobs_small = rng.random(WIN * NSTATIONS, dtype=np.float64)

    Timeshifts_small = np.array([0], dtype=np.int64)

    # -----------------------------------------------------
    # 1. Warm up inversion kernel
    # -----------------------------------------------------
    shift, mech, best_idx, vr, corr, singulars, condition, _vrm = compute_inversion_numbaCpu(
        G_small,
        Dobs_small,
        0.1,
        Timeshifts_small,
        NSTATIONS
    )

    # -----------------------------------------------------
    # 2. Warm up focal mechanism conversion
    # -----------------------------------------------------
    _ = silsub(mech)

    # -----------------------------------------------------
    # 3. (Optional) Bootstrap warmup (kept disabled)
    # -----------------------------------------------------
    """
    W_small = rng.random((1, NSTATIONS), dtype=np.float64)
    Coords_small = rng.random((4, 3), dtype=np.float64)

    _ = compute_bootstraping_numbaCpu(
        G_small,
        Dobs_small,
        W_small[0, :],
        0.1,
        Timeshifts_small,
        Coords_small
    )
    """

    end = time.time()

    return end - start


def initialize_inversion(job, config):
    """
    Minimal initialization step:
    - reads station file
    - builds Dobs
    - computes masks
    - builds filter + time parameters
    """

    # =========================================================
    # 1. Extract paths from job
    # =========================================================
    

    allstatPath = Path(job["allstat"])

    # =========================================================
    # 2. Read station configuration (allstat)
    # =========================================================
    Num_Stations = []
    Num_Channels = []
    Removed_Channels = []

    allstat_lines = []
    with open(allstatPath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            allstat_lines.append(line)
            parts = line.split()
            flags = np.array(list(map(int, parts[2:5])))

            Removed_Channels.append(np.where(flags == 0)[0])
            Num_Stations.append(parts[0])
            Num_Channels.append(np.sum(flags))
            f1 = float(parts[5])
            f2 = float(parts[8])

    Num_Stations = np.array(Num_Stations)
    Num_Channels = np.array(Num_Channels)

    StationsUsed = Num_Stations[Num_Channels > 0]
    ChannelsUsed = Num_Channels[Num_Channels > 0]

    # =========================================================
    # 2. Read station configuration (allstat)
    # =========================================================
    grdatPath = Path(job["grdat"])
    with open(grdatPath, "r") as f:
        for line in f:
            if "nfreq" in line:
                nfreq = int(line.split("=")[1])
            elif "tl" in line:
                tl = float(line.split("=")[1])
            elif "aw" in line:
                aw = float(line.split("=")[1])

    # =========================================================
    # 3. Load waveform data (Dobs)
    # =========================================================
    # Read from allstat file F1 F4. Compute the normalized frequencies
    Dt = tl/1024 # Dt to use ro resampling
    fs = 1/Dt # Fs to use for filter design
    low = f1 / (0.5 * fs)
    high = f2 / (0.5 * fs)
    b, a = butter(4, [low, high], btype='band')
    D = config.st
    D.resample(1024 / tl)

    CHANNEL_ORDER = ["N", "E", "Z"]
    ordered_signals = []

    for sta in StationsUsed:
        sta_traces = D.select(station=sta)

        chan_map = {}
        for tr in sta_traces:
            velocityObs = lfilter(b, a, tr.data)
            dt = tr.stats.delta
            displacement = np.cumsum(velocityObs) * dt
            chan_map[tr.stats.channel[-1]] = displacement

        for ch in CHANNEL_ORDER:
            if ch in chan_map:
                ordered_signals.append(chan_map[ch][:1024])

    Dobs = np.concatenate(ordered_signals)

    # =========================================================
    # 4. Build removeIdx mask
    # =========================================================
    removeIdx = []

    for i, rmv in enumerate(Removed_Channels):
        if len(rmv) > 0:
            for shift in rmv:
                step = i * 3 * 1024
                removeIdx.append(np.arange(step + shift * 1024,
                                           step + (shift + 1) * 1024))

    if len(removeIdx) > 0:
        removeIdx = np.concatenate(removeIdx)
    else:
        removeIdx = np.array([], dtype=int)

    # =========================================================
    # 5. Read inpinv - Create Timeshifts
    # =========================================================
    inpinvPath = Path(job["inpinv"])

    with open(inpinvPath, "r") as f:
        tmin, tstep, tmax = map(int, f.readline().split())
   
    Timeshifts = np.arange(tmin,tmax + 1,tstep).astype(int)

    # =========================================================
    # 6. Attenuation correction
    # =========================================================
    NFFT = 1024

    LOCAL_AW = -np.pi * aw / tl

    CK = np.arange(NFFT) / NFFT
    CC = np.exp(-LOCAL_AW * tl * CK) / tl

    CCm = CC[:, None, None, None]



    # =========================================================
    # 8. Basic metadata
    # =========================================================
    nr = len(Num_Channels)

    # =========================================================
    # 9. Package everything into setup object
    # =========================================================
    setup = {

        "Dobs": Dobs,
        "removeIdx": removeIdx,
        "StationsUsed": StationsUsed,
        "ChannelsUsed": ChannelsUsed,
        "Timeshifts": Timeshifts,
        "CCm": CCm,
        "nr": nr,
        "NFFT": NFFT,
        "b": b,
        "a": a,
        "Dt": Dt,
        "Fs": fs,
        "nfreq": nfreq,
        "tstep": tstep,
        "allstat_lines": allstat_lines,

    }

    return setup

def run_python_inversion(setup, job):

    startInv = time.time()

    # =====================================================
    # 1. Extract setup variables
    # =====================================================
    Dobs = setup["Dobs"]
    removeIdx = setup["removeIdx"]
    Timeshifts = setup["Timeshifts"]
    CCm = setup["CCm"]
    nr = setup["nr"]
    NFFT = setup["NFFT"]
    Dt = setup["Dt"]
    b = setup["b"]
    a = setup["a"]
    nfreq = setup["nfreq"]

    # =====================================================
    # 2. Greens files
    # =====================================================
    greens_files = job["greens_files"]   # numerically sorted upstream

    keyinv = 2
    components = 5 if keyinv == 2 else 6

    # =====================================================
    # 3. Channel mask as (station, record) pairs
    #    (same order the old keep_mask produced)
    # =====================================================
    keep_mask = np.ones(nr * 3 * NFFT, dtype=np.bool_)
    keep_mask[removeIdx] = False
    ch_keep = keep_mask.reshape(nr * 3, NFFT)[:, 0]
    keep_pairs = [(int(idx // 3), int(idx % 3)) for idx in np.where(ch_keep)[0]]

    # =====================================================
    # 4. Parallel Greens loading
    # =====================================================
    def load_single_chunk(filepath):
        return load_hes_optimized(filepath, nfreq, nr)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(load_single_chunk, greens_files))

    # real per-chunk sizes -> correct best_idx -> chunk mapping
    chunk_sizes = [U.shape[3] for U in results]
    chunk_bounds = np.cumsum([0] + chunk_sizes)
    total_sources = int(chunk_bounds[-1])

    U_all = np.empty((nfreq, 18, nr, total_sources), dtype=np.complex128)
    cur = 0
    for U_full in results:
        n_c = U_full.shape[3]
        U_all[:, :, :, cur:cur + n_c] = U_full
        cur += n_c
    del results

    endLoad = time.time()

    # =====================================================
    # 5. Elementary seismograms (chunked + threaded,
    #    numerically identical to the old pipeline)
    # =====================================================
    startElemse = time.time()

    CC = CCm.ravel()
    G = build_G_transposed(U_all, CC, b, a, Dt, keep_pairs, components, NFFT)
    del U_all

    endElemse = time.time()

    # =====================================================
    # 6. Run inversion
    # =====================================================
    Dobs = np.ascontiguousarray(Dobs, dtype=np.float64)
    Timeshifts = np.ascontiguousarray(Timeshifts, dtype=np.int64)

    startKernel = time.time()
    shift, mech, best_idx, vr, corr, singulars, condition, vr_mat = \
        compute_inversion_numbaCpu(G, Dobs, Dt, Timeshifts, nr)
    kernelTime = time.time() - startKernel

    # =====================================================
    # 7. Decomposition
    # =====================================================
    angl = silsub(mech)

    # =====================================================
    # 8. Extract best solution
    # =====================================================
    BestVR = vr[best_idx]
    BestCorr = corr[best_idx]
    BestGridPoint = best_idx
    BestMT = mech[best_idx]
    Bestshift = shift[best_idx]

    # chunk index from the REAL chunk boundaries (chunks can have
    # different sizes; integer division by a fixed ns was fragile)
    BestChunk = int(np.searchsorted(chunk_bounds, best_idx, side='right') - 1)

    BestGF = G[best_idx]   # (n_total, components) - same values as the old
                           # Gall[:, 0:components, best_idx]

    # =====================================================
    # 9. Synthetic waveforms (unchanged logic)
    # =====================================================
    n_stations = BestGF.shape[0] // NFFT
    G_reshaped = BestGF.reshape(n_stations, NFFT, BestGF.shape[1])
    G_shifted_mat = np.zeros_like(G_reshaped)

    if Bestshift > 0:
        G_shifted_mat[:, Bestshift:, :] = \
            G_reshaped[:, :-Bestshift, :]
    elif Bestshift < 0:
        s_abs = abs(Bestshift)
        G_shifted_mat[:, :-s_abs, :] = \
            G_reshaped[:, s_abs:, :]
    else:
        G_shifted_mat = G_reshaped

    GB_shift = G_shifted_mat.reshape(BestGF.shape[0], BestGF.shape[1])
    DsynthB = GB_shift @ BestMT

    # =====================================================
    # 10. Return everything (same keys/values as before)
    # =====================================================
    result = {
        "VR": BestVR,
        "Corr": BestCorr,
        "BestShift": Bestshift,
        "BestChunk": BestChunk,
        "BestMT": BestMT,
        "BestGF": BestGF,
        "Dsynth": DsynthB,
        "BestPlane1": angl[0][best_idx],
        "BestPlane2": angl[1][best_idx],
        "BestMo": angl[2][best_idx],
        "BestMw": angl[3][best_idx],
        "BestDC": angl[4][best_idx],
        "BestCLVD": angl[5][best_idx],
        "BestISO": angl[6][best_idx],
        "BestP": angl[7][best_idx],
        "BestT": angl[8][best_idx],
        "BestB": angl[9][best_idx],
        "Tensor": angl[10][best_idx],
        "Singulars": singulars[best_idx],
        "Condition": condition[best_idx]
    }

    # ---- keep everything getBootstrapping() needs in memory ----
    global _BOOT_CTX
    _BOOT_CTX = {
        "G": G, "vr": vr, "shift": shift, "best_idx": best_idx,
        "Dobs": Dobs, "Dt": Dt, "setup": setup, "result": result,
        "components": components,
        "vr_mat": vr_mat, "corr": corr, "angl": angl,
        "times": {"load": endLoad - startInv,
                  "elemse": endElemse - startElemse,
                  "kernel": kernelTime,
                  "total": time.time() - startInv},
    }

    return result


def save_results(result, outdir=None):

    print("\n================ INVERSION RESULTS ================\n")

    print(f"VR: {result['VR']}")
    print(f"Corr: {result['Corr']}")
    print(f"Best shift: {result['BestShift']}")
    print(f"Best chunk: {result['BestChunk']}")

    print("\nMT parameters:")
    print(result["BestMT"])

    # Planes
    p1 = result["BestPlane1"]
    p2 = result["BestPlane2"]

    print(f"\nPlane 1: Strike = {p1[0]:.1f}, Dip = {p1[1]:.1f}, Rake = {p1[2]:.1f}")
    print(f"Plane 2: Strike = {p2[0]:.1f}, Dip = {p2[1]:.1f}, Rake = {p2[2]:.1f}")

    # Magnitudes / source parameters
    print(f"\nM0: {result['BestMo']}")
    print(f"Mw: {result['BestMw']}")
    print(f"DC%: {result['BestDC']}")
    print(f"CLVD%: {result['BestCLVD']}")
    print(f"ISO: {result['BestISO']}")

    # Axes
    print("\nP axes:", np.round(result["BestP"]))
    print("T axes:", np.round(result["BestT"]))
    print("B axes:", np.round(result["BestB"]))

    # Tensor diagnostics
    print("\nSingulars:", result["Singulars"])
    print("Condition number:", result["Condition"])

    # CMT location (populated by calculateInversions2 from source files)
    if "Xcmt" in result:
        print(f"\nCMT grid position:  X={result['Xcmt']:.2f} km"
              f"  Y={result['Ycmt']:.2f} km"
              f"  Depth={result['Zcmt']:.2f} km")
    if "CMT_lat" in result:
        print(f"CMT location:       Lat={result['CMT_lat']:.4f}"
              f"  Lon={result['CMT_lon']:.4f}")

    print("\n==================================================\n")

@config.time
def calculateInversions2():

    os.makedirs(config.inversiondir, exist_ok=True)

    def _chunk_no(fname):
        # gr.<crustal>.<grdat>.<grid>.<CHUNK>.hes -> sort by CHUNK numerically.
        # Plain sorted() was lexicographic (0,1,10,11,...,19,2,20,...), so the
        # source axis of Gall did not match the grid-point numbering and the
        # best solution was attributed to the wrong chunk / grid point.
        return int(os.path.basename(fname).split('.')[-2])

    greens_files = sorted(
        [os.path.join(config.greendir, f)
         for f in os.listdir(config.greendir) if f.endswith(".hes")],
        key=_chunk_no)
    greens = [os.path.basename(f) for f in greens_files]
    base = greens[0]
    _, icrustal, igrdat, igrid, _, _ = base.split('.')

    for allstat in sorted(os.listdir(config.allstatdir)):

        for inpinv in sorted(os.listdir(config.inpinvdir)):

            invdir='{}.{}.{}.{}.{}'.format(
                allstat.split('.')[0][7:],
                inpinv.split('.')[0][6:],
                icrustal,
                igrdat,
                igrid
            )

            outdir = os.path.join(config.inversiondir, invdir)

            os.makedirs(outdir, exist_ok=True)

            job = {
                "allstat": os.path.join(config.allstatdir,allstat),
                "inpinv": os.path.join(config.inpinvdir,inpinv),
                "greens_dir": config.greendir,
                "greens_files": greens_files,
                "greens": greens,
                "crustal": os.path.join(config.crustaldir,f"crustal{icrustal}.dat"),
                "station": os.path.join(config.workdir,"station.dat"),
                "grdat": os.path.join(config.grdatdir,f"grdat{igrdat}.hed"),
                "raw": os.path.join(config.rawdir,igrdat),
                "outdir": outdir
            }

            setup = initialize_inversion(job, config)

            result = run_python_inversion(setup, job)

            # --- Read source grid files once to get CMT coordinates ---
            _gfiles = sorted(
                glob.glob(os.path.join(config.workdir, 'sources', 'grid0', '*.dat')),
                key=lambda f: int(''.join(c for c in os.path.basename(f) if c.isdigit())))
            _grid = np.vstack([np.loadtxt(f, usecols=(0, 1, 2, 3), ndmin=2)
                               for f in _gfiles])
            _bidx = int(_BOOT_CTX["best_idx"])
            _crow = _grid[_grid[:, 0] == _bidx + 1][0]
            result["Xcmt"] = float(_crow[1])
            result["Ycmt"] = float(_crow[2])
            result["Zcmt"] = float(_crow[3])
            _cmt_lat, _cmt_lon = cart2earth(
                config.org.latitude, config.org.longitude,
                result["Xcmt"], result["Ycmt"])
            result["CMT_lat"] = _cmt_lat
            result["CMT_lon"] = _cmt_lon
            # Update _BOOT_CTX so bootstrap uses these values from memory
            _BOOT_CTX["result"] = result
            _BOOT_CTX["grid"] = _grid
            # ---------------------------------------------------------

            save_results(result, outdir)



def _hudson_cloud_plot(u_vals, v_vals, VR_plot, center_mt, label, outpath):
    """Hudson projection of the (subsampled, precomputed) bootstrap cloud."""
    norm = mcolors.Normalize(vmin=VR_plot.min(), vmax=VR_plot.max())
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(8, 8))
    hudson.draw_axes(ax)

    sort_idx = np.argsort(VR_plot)  # higher VR plotted on top
    ax.scatter(u_vals[sort_idx], v_vals[sort_idx],
               c=cmap(norm(VR_plot[sort_idx])), s=10, alpha=0.6,
               edgecolors='none')

    u_b, v_b = hudson.project(center_mt)
    beachball.plot_beachball_mpl(
        center_mt, ax, position=(u_b, v_b), size=15,
        color_t='red', color_p='white', edgecolor='black',
        linewidth=1.0, alpha=1.0, zorder=10)
    ax.text(u_b, v_b + 0.05, label, color='red', fontweight='bold',
            ha='center', va='bottom', fontsize=9)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Variance Reduction (%)', rotation=270, labelpad=15)

    ax.set_aspect('equal')
    ax.set_title("Hudson Plot: Bootstrap Cloud with " + label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


@config.time
def getBootstrapping(num_iterations=100, num_realizations=100):
    """
    Bootstrap uncertainty analysis, ported from GISOLA2.0_TimeTest2.py.
    Reuses the Greens matrix and inversion results kept in memory by
    run_python_inversion(); all figures are saved to <workdir>/output/plots.
    """
    if _BOOT_CTX is None:
        config.logger.info('No inversion in memory - run getInversions() first')
        return

    startBoot = time.time()

    G = _BOOT_CTX["G"]
    vr = _BOOT_CTX["vr"]
    shift = _BOOT_CTX["shift"]
    best_idx = int(_BOOT_CTX["best_idx"])
    Dobs = _BOOT_CTX["Dobs"]
    Dt = _BOOT_CTX["Dt"]
    setup = _BOOT_CTX["setup"]
    result = _BOOT_CTX["result"]
    invtimes = _BOOT_CTX["times"]

    ChannelsUsed = np.asarray(setup["ChannelsUsed"], dtype=np.int64)
    tstep = int(setup["tstep"])

    plotdir = os.path.join(config.workdir, 'output', 'plots')
    os.makedirs(plotdir, exist_ok=True)

    # =========================================================
    # 1. Station-weight realizations (Dirichlet), as in TimeTest2
    # =========================================================
    dirichlet_samples = np.random.dirichlet(alpha=[1] * len(ChannelsUsed),
                                            size=num_realizations)
    W = []
    for weights in dirichlet_samples:
        W_single = []
        for count, weight in zip(ChannelsUsed, weights):
            W_single.extend([weight / count] * (int(count) * 1024))
        W.append(W_single)
    W = np.array(W)
    config.logger.debug("Bootstrap: W matrix %s, time elapsed: %.2f s",
                        W.shape, time.time() - startBoot)

    # =========================================================
    # 2. Grid and CMT coordinates from the inversion context
    #    (source files were already read in calculateInversions2())
    # =========================================================
    grid = _BOOT_CTX["grid"][:len(vr)]
    Xcmt = float(result["Xcmt"])
    Ycmt = float(result["Ycmt"])
    Zcmt = float(result["Zcmt"])
    config.logger.info(
        'Bootstrap CMT (from inversion): X={} Y={} Z={}'.format(Xcmt, Ycmt, Zcmt))
    gid, gx, gy, gz = grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]

    ErrorTresh = round(0.8 * vr.max())
    m = vr >= ErrorTresh
    xminb, xmaxb = gx[m].min(), gx[m].max()
    yminb, ymaxb = gy[m].min(), gy[m].max()
    zminb, zmaxb = gz[m].min(), gz[m].max()
    tminb, tmaxb = int(shift[m].min()), int(shift[m].max())

    xmin, xmax = gx.min(), gx.max()
    ymin, ymax = gy.min(), gy.max()
    zmin, zmax = gz.min(), gz.max()

    config.logger.debug("Max VR: %s, thresh 80%%: %s", vr.max(), ErrorTresh)
    config.logger.debug("Grid boundaries: X [%s, %s] Y [%s, %s] Z [%s, %s]",
                        xmin, xmax, ymin, ymax, zmin, zmax)
    config.logger.debug("CMT coordinates: X: %s, Y: %s, Z: %s", Xcmt, Ycmt, Zcmt)
    config.logger.debug("Bootstrap box: X [%s, %s] Y [%s, %s] Z [%s, %s] T [%s, %s]",
                        xminb, xmaxb, yminb, ymaxb, zminb, zmaxb, tminb, tmaxb)

    inbox = ((gx >= xminb) & (gx <= xmaxb) &
             (gy >= yminb) & (gy <= ymaxb) &
             (gz >= zminb) & (gz <= zmaxb))
    idx = gid[inbox].astype(np.int64)
    Coords = np.ascontiguousarray(grid[inbox][:, 1:4])

    try:
        Grid_Percentage = (((xmaxb - xminb) * (ymaxb - yminb) * (zmaxb - zminb)) /
                           ((xmax - xmin) * (ymax - ymin) * (zmax - zmin)))
        config.logger.debug("Grid percentage: %s, points: %s / %s, true percentage: %s",
                            Grid_Percentage, len(idx), grid.shape[0],
                            len(idx) / grid.shape[0])
    except ZeroDivisionError:
        pass

    idx0 = idx - 1
    valid = (idx0 >= 0) & (idx0 < G.shape[0])
    valid_idx = idx0[valid]
    Coords = Coords[valid]

    Gboot = np.ascontiguousarray(G[valid_idx])
    TimeshiftsBoot = np.arange(tminb, tmaxb + 1, tstep).astype(np.int64)
    config.logger.debug("Bootstrap GF selection: %.2f s -> %s grid points, %s shifts",
                        time.time() - startBoot, Gboot.shape[0], len(TimeshiftsBoot))

    # =========================================================
    # 2b. Whole grid coloured by VR + bootstrap subset figure
    # =========================================================
    try:
        bsel = np.zeros(len(gx), dtype=bool)
        bsel[np.where(inbox)[0][valid]] = True

        order = np.argsort(vr)   # high-VR points drawn on top
        norm = mcolors.Normalize(vmin=vr.min(), vmax=vr.max())
        cmap = plt.get_cmap('viridis')

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        panels = [(gx, gy, 'X (km)', 'Y (km)', (Xcmt, Ycmt), False),
                  (gx, gz, 'X (km)', 'Depth Z (km)', (Xcmt, Zcmt), True),
                  (gy, gz, 'Y (km)', 'Depth Z (km)', (Ycmt, Zcmt), True)]
        sc = None
        for ax, (u, v, xl, yl, cmt, flip) in zip(axes, panels):
            sc = ax.scatter(u[order], v[order], c=vr[order], cmap=cmap,
                            norm=norm, s=18, alpha=0.9, edgecolors='none',
                            label='Grid points (VR colour)')
            ax.scatter(u[bsel], v[bsel], facecolors='none', edgecolors='red',
                       s=46, linewidths=0.9, label='Bootstrap subset')
            ax.scatter([cmt[0]], [cmt[1]], marker='*', s=280, color='lime',
                       edgecolors='black', linewidths=1.0, zorder=5,
                       label='Best solution (CMT)')
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            if flip:
                ax.invert_yaxis()
        axes[0].legend(loc='best', fontsize=9, framealpha=0.9)
        cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label('Variance Reduction (%)')
        fig.suptitle('Source grid coloured by VR, with bootstrap subset',
                     fontsize=15, fontweight='bold')
        fig.savefig(os.path.join(plotdir, 'grid_vr_bootstrap.png'), dpi=150)
        plt.close(fig)
        config.logger.debug("Grid/VR overview figure saved: %s",
                            os.path.join(plotdir, 'grid_vr_bootstrap.png'))
    except Exception:
        import traceback
        traceback.print_exc()
        config.logger.warning("Grid/VR figure failed (bootstrap continues)")

    # =========================================================
    # 3. Bootstrap iterations
    # =========================================================
    BootstrapResults = []
    BootstrapAngles = []
    startIters = time.time()
    for i in range(num_iterations):
        res = compute_bootstraping_numbaCpu(Gboot, Dobs, W[i, :], Dt,
                                            TimeshiftsBoot, Coords)
        BootstrapResults.append(res)
        BootstrapAngles.append(silsub(res[2]))
    itersTime = time.time() - startIters
    config.logger.debug("Bootstrap iterations done: %s x %s grid points = %.2f s (%s ms/iter)",
                        num_iterations, Gboot.shape[0], itersTime,
                        round(1000.0 * itersTime / num_iterations))

    # =========================================================
    # 5. Aggregate + filter (as in TimeTest2)
    # =========================================================
    all_mechanisms = np.vstack([it[2] for it in BootstrapResults])
    all_shifts = np.concatenate([r[0] for r in BootstrapResults])
    all_coords = np.vstack([r[4] for r in BootstrapResults])

    plane1_all = np.vstack([r[0] for r in BootstrapAngles])
    plane2_all = np.vstack([r[1] for r in BootstrapAngles])
    M0_all = np.hstack([r[2] for r in BootstrapAngles])
    Mw_all = np.hstack([r[3] for r in BootstrapAngles])
    DC_all = np.hstack([r[4] for r in BootstrapAngles])
    CLVD_all = np.hstack([r[5] for r in BootstrapAngles])
    ISO_all = np.hstack([r[6] for r in BootstrapAngles])
    VR_all = np.concatenate([r[3] for r in BootstrapResults])

    config.logger.debug("Bootstrap fit boundaries, Max: %s, Min: %s",
                        VR_all.max(), VR_all.min())
    thresplot = 0.65 * VR_all.max()

    mask = VR_all > thresplot
    indices = np.where(mask)[0]

    mt_array = np.array(all_mechanisms)[mask]
    VR_all = VR_all[mask]
    all_coords = all_coords[mask]
    all_shifts = all_shifts[mask]
    plane1_all = plane1_all[mask]
    plane2_all = plane2_all[mask]
    M0_all = M0_all[mask]
    Mw_all = Mw_all[mask]
    DC_all = DC_all[mask]
    CLVD_all = CLVD_all[mask]
    ISO_all = ISO_all[mask]

    SolutionParts = np.column_stack((DC_all, CLVD_all, ISO_all))
    SolutionPartsMean = np.mean(SolutionParts, axis=0)
    Magnitudes = np.column_stack((M0_all, Mw_all))
    MagnitudesMean = np.mean(Magnitudes, axis=0)

    Planes = np.vstack([plane1_all, plane2_all])
    strikes = Planes[:, 0]
    threshold = np.mean(strikes)
    plane1_all = Planes[strikes < threshold]
    plane2_all = Planes[strikes >= threshold]
    plane1_mean = np.mean(plane1_all, axis=0)
    plane2_mean = np.mean(plane2_all, axis=0)

    mt_mean = np.mean(mt_array, axis=0)
    mt_std = np.std(mt_array, axis=0)
    mt_median = np.median(mt_array, axis=0)

    Bestplane1 = result["BestPlane1"]
    Bestplane2 = result["BestPlane2"]
    BestParts = np.array([result["BestDC"], result["BestCLVD"], result["BestISO"]])
    BestMagnitudes = np.array([result["BestMo"], result["BestMw"]])
    CMT_BEST = [result["BestShift"], Xcmt, Ycmt, Zcmt]

    # =========================================================
    # 6. Figures (saved, not shown)
    # =========================================================
    startPlots = time.time()
    try:
        _make_bootstrap_figures(
            plotdir, Dobs, result, VR_all, indices, mt_mean, mt_median,
            plane1_all, plane2_all, plane1_mean, plane2_mean,
            Bestplane1, Bestplane2, SolutionParts, SolutionPartsMean,
            BestParts, Magnitudes, MagnitudesMean, BestMagnitudes,
            all_shifts, all_coords, CMT_BEST, BootstrapAngles,
            (tminb, tmaxb), (xminb, xmaxb), (yminb, ymaxb), (zminb, zmaxb))
    except Exception:
        import traceback
        traceback.print_exc()
        print("WARNING: figure generation failed (results above are unaffected)")
    plotsTime = time.time() - startPlots

    # ---- Store bootstrap stats for renderSite() to embed in final HTML ----
    try:
        boot_stats = {
            'n_solutions': int(len(VR_all)),
            'n_iterations': int(num_iterations),
            'vr_max': float(VR_all.max()),
            'p1_med': np.median(plane1_all, axis=0),
            'p1_std': np.std(plane1_all, axis=0),
            'p2_med': np.median(plane2_all, axis=0),
            'p2_std': np.std(plane2_all, axis=0),
        }
        for _key, _arr in (('x', all_coords[:, 0]), ('y', all_coords[:, 1]),
                           ('z', all_coords[:, 2]), ('time', all_shifts * Dt),
                           ('Mo', M0_all), ('Mw', Mw_all), ('DC', DC_all)):
            _arr = np.asarray(_arr, dtype=float)
            boot_stats[_key] = (float(np.median(_arr)), float(np.mean(_arr)),
                                float(np.std(_arr)),
                                float(np.percentile(_arr, 2.5)),
                                float(np.percentile(_arr, 97.5)))
        global _BOOT_STATS
        _BOOT_STATS = (plotdir, boot_stats, result)
        config.logger.info('Bootstrap stats stored; renderSite() will embed them in index.html')
    except Exception:
        import traceback
        traceback.print_exc()
        print("WARNING: bootstrap stats storage failed (analysis unaffected)")

    # =========================================================
    # 7. Store final summary for deferred printing in gisolaBootstrap.py
    # =========================================================
    endBoot = time.time()
    computeTime = (endBoot - startBoot) - plotsTime
    global _SUMMARY
    _SUMMARY = {
        'invtimes':        invtimes,
        'num_iterations':  num_iterations,
        'n_grid':          Gboot.shape[0],
        'itersTime':       itersTime,
        'computeTime':     computeTime,
        'result':          result,
        'best_idx':        best_idx,
        'plane1_mean':     plane1_mean,
        'plane2_mean':     plane2_mean,
        'MagnitudesMean':  MagnitudesMean,
        'SolutionPartsMean': SolutionPartsMean,
        'mt_mean':         mt_mean,
        'mt_std':          mt_std,
        'plotsTime':       plotsTime,
        'plotdir':         plotdir,
    }


def _make_bootstrap_figures(plotdir, Dobs, result, VR_all, indices,
                            mt_mean, mt_median,
                            plane1_all, plane2_all, plane1_mean, plane2_mean,
                            Bestplane1, Bestplane2,
                            SolutionParts, SolutionPartsMean, BestParts,
                            Magnitudes, MagnitudesMean, BestMagnitudes,
                            all_shifts, all_coords, CMT_BEST, BootstrapAngles,
                            tlim, xlim, ylim, zlim):
    """All TimeTest2 figures, saved as PNG into plotdir."""

    # ---- waveform fit ----
    fig = plt.figure(figsize=(12, 4))
    plt.plot(Dobs)
    plt.plot(result["Dsynth"])
    plt.legend(['Dobs', 'Dsynth'])
    plt.title('Waveform fit (best solution)')
    fig.savefig(os.path.join(plotdir, 'fit_waveforms.png'), dpi=150)
    plt.close(fig)

    # ---- histogram helper ----
    def _hist_row(data2d, means, bests, titles, suptitle, fname, xrange=None):
        n = len(titles)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        for i in range(n):
            kw = {'range': xrange} if xrange else {}
            axes[i].hist(data2d[:, i], bins=20, color='skyblue',
                         edgecolor='black', alpha=0.7, **kw)
            axes[i].axvline(means[i], color='red', linestyle='dashed',
                            linewidth=2, label='Mean')
            axes[i].axvline(bests[i], color='green', linestyle='dashed',
                            linewidth=2, label='Best')
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Value')
            axes[i].legend()
        axes[0].set_ylabel('Frequency')
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(plotdir, fname), dpi=150)
        plt.close(fig)

    _hist_row(plane1_all, plane1_mean, Bestplane1, ['Strike', 'Dip', 'Rake'],
              'Histograms for plane 1 (N = %d)' % plane1_all.shape[0],
              'hist_plane1.png')
    _hist_row(plane2_all, plane2_mean, Bestplane2, ['Strike', 'Dip', 'Rake'],
              'Histograms for plane 2 (N = %d)' % plane2_all.shape[0],
              'hist_plane2.png')
    _hist_row(SolutionParts, SolutionPartsMean, BestParts,
              ['DC%', 'CLVD%', 'ISO%'], 'Histograms for solution decomposition',
              'hist_decomposition.png', xrange=(0, 100))
    _hist_row(Magnitudes, MagnitudesMean, BestMagnitudes, ['Mo', 'Mw'],
              'Histograms for Magnitudes', 'hist_magnitudes.png')

    # ---- time / centroid KDE panel ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Time Shift (s)', 'X Coordinate', 'Y Coordinate', 'Depth (Z)']
    data_list = [all_shifts, all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]]
    colors_ = ['salmon', 'lightgreen', 'orange', 'plum']
    custom_limits = [tlim, xlim, ylim, zlim]
    for i in range(4):
        data = np.asarray(data_list[i], dtype=float)
        low_lim, high_lim = custom_limits[i]
        mu = np.mean(data)
        ci_low, ci_high = np.percentile(data, [2.5, 97.5])
        axes[i].hist(data, bins=20, range=(low_lim, high_lim), color='gray',
                     edgecolor='black', alpha=0.15, density=True)
        if len(np.unique(data)) > 1:
            kde = gaussian_kde(data)
            kde.set_bandwidth(bw_method=kde.factor * 3.0)
            xr = np.linspace(low_lim, high_lim, 500)
            axes[i].plot(xr, kde(xr), color=colors_[i], lw=3, label='Smoothed KDE')
            ci_x = np.linspace(ci_low, ci_high, 200)
            axes[i].fill_between(ci_x, kde(ci_x), color=colors_[i], alpha=0.3,
                                 label='95%% CI: [%.1f, %.1f]' % (ci_low, ci_high))
        axes[i].axvline(mu, color='red', linestyle='-', linewidth=1.5,
                        label='Mean: %.2f' % mu)
        axes[i].axvline(CMT_BEST[i], color='green', linestyle='--', linewidth=2,
                        label='Best')
        axes[i].set_xlim(low_lim, high_lim)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].legend(fontsize='8', loc='upper right')
    fig.suptitle('Bootstrap Time and Centroid Uncertainty Distributions',
                 fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Probability Density')
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, 'hist_centroid_time.png'), dpi=150)
    plt.close(fig)

    # ---- ISO histogram ----
    iso = SolutionParts[:, 2]
    iso_median = np.median(iso)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(iso, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(iso_median, color='red', linestyle='dashed', linewidth=2,
               label='Median = %.1f%%' % iso_median)
    ax.set_xlabel("ISO (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_title("Moment Tensor Decomposition: ISO Component", fontweight='bold')
    fig.savefig(os.path.join(plotdir, 'hist_iso.png'), dpi=150)
    plt.close(fig)

    if not HAVE_PYROCKO:
        print("pyrocko not installed - skipping Hudson/beachball figures")
        return

    # ---- moment tensor objects: SUBSAMPLED for plotting speed ----
    # All statistics above use the FULL ensemble; the figures only need a
    # representative cloud. A seeded random subset preserves the VR/MT
    # distribution (covers the moment-tensor space proportionally) while
    # cutting figure time by ~10x.
    MAX_HUDSON, MAX_FUZZY = 500, 300
    all_mats = np.concatenate([res[-1] for res in BootstrapAngles])[indices]
    n_mt = all_mats.shape[0]
    rng = np.random.default_rng(1)
    if n_mt > MAX_HUDSON:
        sel = np.sort(rng.choice(n_mt, MAX_HUDSON, replace=False))
    else:
        sel = np.arange(n_mt)
    mt_plot = [pmt.MomentTensor(m=all_mats[i]) for i in sel]
    VR_plot = VR_all[sel]
    config.logger.debug("Bootstrap figures: plotting %s of %s solutions", len(sel), n_mt)

    # Hudson projection computed ONCE, reused by all three Hudson figures
    uv = np.array([hudson.project(mt) for mt in mt_plot])
    u_vals, v_vals = uv[:, 0], uv[:, 1]

    best_mt = pmt.MomentTensor(m=result["Tensor"])
    mt_mean6 = np.append(mt_mean, 0) if len(mt_mean) == 5 else mt_mean
    mean_mt = pmt.MomentTensor(m=build_moment_tensor(np.asarray(mt_mean6, dtype=np.float64)))
    mt_median6 = np.append(mt_median, 0) if len(mt_median) == 5 else mt_median
    median_mt = pmt.MomentTensor(m=build_moment_tensor(np.asarray(mt_median6, dtype=np.float64)))

    _hudson_cloud_plot(u_vals, v_vals, VR_plot, best_mt, 'Best Fit',
                       os.path.join(plotdir, 'hudson_best.png'))
    _hudson_cloud_plot(u_vals, v_vals, VR_plot, mean_mt, 'Mean solution',
                       os.path.join(plotdir, 'hudson_mean.png'))
    _hudson_cloud_plot(u_vals, v_vals, VR_plot, median_mt, 'Median solution',
                       os.path.join(plotdir, 'hudson_median.png'))

    # ---- best beachball ----
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    beachball.plot_beachball_mpl(
        best_mt, ax, beachball_type='full', position=(0, 0), size=200.0,
        color_t='red', color_p='white', edgecolor='black', linewidth=1.0,
        alpha=1.0)
    ax.set_xlim([-100, 100]); ax.set_ylim([-100, 100])
    ax.set_aspect('equal'); ax.axis('off')
    plt.title("Best Moment Tensor\nMw %.1f" % best_mt.moment_magnitude(), pad=20)
    fig.savefig(os.path.join(plotdir, 'beachball_best.png'), dpi=150)
    plt.close(fig)

    # ---- full-MT figure (colors: deviatoric, planes: DC) ----
    try:
        event_id = os.path.basename(os.path.dirname(config.workdir))
    except Exception:
        event_id = ''
    mt_dev = best_mt.deviatoric()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    beachball.plot_beachball_mpl(
        mt_dev, ax, beachball_type='full', position=(0, 0), size=250.0,
        color_t='red', color_p='white', edgecolor='black', linewidth=0.5,
        alpha=1.0, zorder=1)
    beachball.plot_beachball_mpl(
        best_mt, ax, beachball_type='dc', position=(0, 0), size=250.0,
        color_t='none', color_p='none', edgecolor='black', linewidth=2.5,
        zorder=2)
    ax.set_xlim([-150, 150]); ax.set_ylim([-110, 130])
    ax.set_aspect('equal'); ax.axis('off')
    ax.text(0, 155, "Event: %s, Mw: %.1f\n" % (event_id, best_mt.moment_magnitude()),
            ha='center', fontsize=13, fontweight='bold')
    ax.text(0, 140, "Full MT Inversion", ha='center', fontsize=11,
            style='italic', color='#333333')
    txt = ("DC: %.1f%%\nCLVD: %.1f%%\nISO: %.1f%%\nVR: %.1f%%" %
           (result['BestDC'], result['BestCLVD'], abs(result['BestISO']),
            result['VR']))
    ax.text(120, 80, txt, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="black", linewidth=1.2, alpha=0.9))
    fig.savefig(os.path.join(plotdir, 'beachball_full_mt.png'), dpi=150)
    plt.close(fig)

    # ---- mean beachball ----
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    beachball.plot_beachball_mpl(
        mean_mt, ax, beachball_type='full', position=(0, 0), size=100.0,
        color_t='red', color_p='white', edgecolor='black', linewidth=1.0,
        alpha=1.0)
    ax.set_xlim([-100, 100]); ax.set_ylim([-100, 100])
    ax.set_aspect('equal'); ax.axis('off')
    plt.title("Mean Moment Tensor\nMw %.1f" % mean_mt.moment_magnitude(), pad=20)
    fig.savefig(os.path.join(plotdir, 'beachball_mean.png'), dpi=150)
    plt.close(fig)

    # ---- fuzzy (uncertainty) beachball ----
    if len(sel) > MAX_FUZZY:
        sel_f = np.sort(rng.choice(sel, MAX_FUZZY, replace=False))
    else:
        sel_f = sel
    normalized_mts = []
    for i in sel_f:
        m_dev = pmt.MomentTensor(m=all_mats[i]).deviatoric()
        normalized_mts.append(pmt.MomentTensor(m=m_dev.m() / m_dev.scalar_moment()))
    best_norm = pmt.MomentTensor(m=best_mt.m() / best_mt.scalar_moment()).deviatoric()

    fig = plt.figure(figsize=(7., 7.))
    ax = fig.add_subplot(1, 1, 1)
    plot_kwargs = {'beachball_type': 'full', 'size': 250, 'position': (0, 0),
                   'color_t': 'black', 'edgecolor': 'black'}
    beachball.plot_fuzzy_beachball_mpl_pixmap(normalized_mts, ax, best_norm,
                                              **plot_kwargs)
    ax.set_xlim(-150, 150); ax.set_ylim(-150., 150)
    ax.set_aspect('equal'); ax.set_axis_off()
    plt.title("Moment Tensor Uncertainty (Bootstrap Ensemble, N=%d)"
              % len(normalized_mts), fontweight='bold')
    ax.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Best Solution')],
              loc='upper right', bbox_to_anchor=(1.1, 0.95))
    fig.savefig(os.path.join(plotdir, 'beachball_fuzzy.png'), dpi=150)
    plt.close(fig)




# =====================================================================
# RESULTS EXPORT + WEBSITE (replaces gatherResults/allplots for the
# Python inversion; reuses the original plot.py / web.py unchanged)
# =====================================================================

def _shifted_obs(Dobs, s, n_stations):
    """d_shifted[base+u] = Dobs[base+u+s] for valid u (per station block),
    so that g = Gk.T @ d_shifted reproduces the kernel correlation."""
    ds = np.zeros_like(Dobs)
    for stat in range(n_stations):
        b = stat * 1024
        if s >= 0:
            ds[b:b + 1024 - s] = Dobs[b + s:b + 1024]
        else:
            ds[b - s:b + 1024] = Dobs[b:b + 1024 + s]
    return ds


def _mechs_for_pairs(G, Dobs, Dt, shifts_array, pairs):
    """Mechanism (a-coefficients) for arbitrary (source, shift-index) pairs.
    Identical math to the inversion kernel."""
    MULTIP = 1e20
    scale = MULTIP * Dt
    n_sources, n_total, n_comp = G.shape
    n_stations = n_total // 1024

    by_s = {}
    for idx, (k, si) in enumerate(pairs):
        by_s.setdefault(si, []).append((idx, k))

    mechs = np.zeros((len(pairs), n_comp))
    R_cache = {}
    for si, items in by_s.items():
        s = int(shifts_array[si])
        ds = _shifted_obs(Dobs, s, n_stations)
        for idx, k in items:
            Gk = G[k]
            g = (Gk.T @ ds) * scale
            Rk = R_cache.get(k)
            if Rk is None:
                Rk = (Gk.T @ Gk) * scale
                R_cache[k] = Rk
            mechs[idx] = np.linalg.solve(Rk, g)
    return mechs


def _read_grid():
    files = sorted(
        glob.glob(os.path.join(config.workdir, 'sources', 'grid0', '*.dat')),
        key=lambda f: int(''.join(c for c in os.path.basename(f) if c.isdigit())))
    return np.vstack([np.loadtxt(f, usecols=(0, 1, 2, 3), ndmin=2) for f in files])


@config.time
def exportResults():
    """
    Builds everything the original site pipeline consumed from the Fortran
    outputs - but from the in-memory Python inversion (_BOOT_CTX):
      output/solutions, output/correlations, per-station fil/syn files,
      dsretc.lst, and the QuakeML output/event.xml (+ event_sc.xml).
    After this, the ORIGINAL plot.py / web.py run unchanged.
    """
    if _BOOT_CTX is None:
        config.logger.info('No inversion in memory - run getInversions() first')
        return

    G = _BOOT_CTX["G"]
    vr = _BOOT_CTX["vr"]
    shift = _BOOT_CTX["shift"]
    corr = _BOOT_CTX["corr"]
    best_idx = int(_BOOT_CTX["best_idx"])
    Dobs = _BOOT_CTX["Dobs"]
    Dt = _BOOT_CTX["Dt"]
    setup = _BOOT_CTX["setup"]
    result = _BOOT_CTX["result"]
    vr_mat = _BOOT_CTX["vr_mat"]
    angl = _BOOT_CTX["angl"]

    Timeshifts = np.asarray(setup["Timeshifts"], dtype=np.int64)
    tl = Dt * 1024.0
    n_sources = G.shape[0]
    n_stations_data = G.shape[1] // 1024

    evt = config.evt
    workdir = config.workdir
    os.makedirs(config.outputdir, exist_ok=True)

    # pseudo "best inversion dir" so the original plot functions find files
    config.bestinvdir = 'python'
    config.inversiondir = os.path.join(workdir, 'inversions')
    config.besttl = tl
    config.revise = False
    pseudo = os.path.join(config.inversiondir, config.bestinvdir)
    os.makedirs(pseudo, exist_ok=True)

    grid = _read_grid()[:n_sources]
    gid, gx, gy, gz = grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]
    bx, by, bz = gx[best_idx], gy[best_idx], gz[best_idx]

    plane1_all, plane2_all = angl[0], angl[1]
    M0_all, DC_all = angl[2], angl[4]

    # ----------------------------------------------------------------
    # 1. solutions file (same csv-ish format the old gatherResults wrote)
    #    [id, x, y, z, shift_samples, corr, moment, dc, s1,d1,r1, s2,d2,r2]
    # ----------------------------------------------------------------
    solutions = []
    for k in range(n_sources):
        solutions.append([float(k + 1), float(gx[k]), float(gy[k]), float(gz[k]),
                          float(shift[k]), float(corr[k]), float(M0_all[k]),
                          float(DC_all[k]),
                          float(plane1_all[k][0]), float(plane1_all[k][1]), float(plane1_all[k][2]),
                          float(plane2_all[k][0]), float(plane2_all[k][1]), float(plane2_all[k][2])])
    with open(os.path.join(config.outputdir, 'solutions'), 'w') as f:
        f.writelines(str(x)[1:-1] + '\n' for x in sorted(solutions))
    config.solutions = solutions

    best = max(solutions, key=lambda x: x[5])

    # ----------------------------------------------------------------
    # 2. correlations file: every shift for all depths at best (x, y)
    #    [id, x,y,z, time_s, corr, s1,d1,r1, s2,d2,r2, dc, iso, vard, mom]
    # ----------------------------------------------------------------
    col = np.where((gx == bx) & (gy == by))[0]
    pairs = [(int(k), int(si)) for k in col for si in range(len(Timeshifts))]
    mechs = _mechs_for_pairs(G, Dobs, Dt, Timeshifts, pairs)
    pang = silsub(mechs)

    corr_mat = np.sqrt(np.clip(vr_mat, 0.0, None) / 100.0)

    correlations = []
    for idx, (k, si) in enumerate(pairs):
        correlations.append([
            float(k + 1), float(gx[k]), float(gy[k]), float(gz[k]),
            float(Timeshifts[si] * Dt), float(corr_mat[k, si]),
            float(pang[0][idx][0]), float(pang[0][idx][1]), float(pang[0][idx][2]),
            float(pang[1][idx][0]), float(pang[1][idx][1]), float(pang[1][idx][2]),
            float(pang[4][idx]), float(pang[6][idx]), 0.0, float(pang[2][idx])])
    with open(os.path.join(config.outputdir, 'correlations'), 'w') as f:
        f.writelines(str(x)[1:-1] + '\n' for x in sorted(correlations))
    config.correlations = correlations

    # ----------------------------------------------------------------
    # 3. quality metrics: stvar / fmvar (same definitions as gatherResults)
    # ----------------------------------------------------------------
    bestcorr = float(corr[best_idx])
    thres = 0.9 * bestcorr
    stvar = float(np.mean(corr_mat >= thres))

    # fmvar: use per-source best mechanisms already computed by the inversion
    # kernel (stored in angl/corr inside _BOOT_CTX).  This avoids re-running
    # _mechs_for_pairs on thousands of (source, shift) pairs, which requires
    # random access across the full ~4 GB G matrix and dominated export time.
    above_thres = np.where(corr >= thres)[0]
    if len(above_thres) > 4000:
        above_thres = np.random.default_rng(0).choice(
            len(above_thres), 4000, replace=False)
    bp1 = result["BestPlane1"]
    kag = [modules.kagan.get_kagan_angle(
               bp1[0], bp1[1], bp1[2],
               float(angl[0][i][0]), float(angl[0][i][1]), float(angl[0][i][2]))
           for i in above_thres]
    fmvar = float(np.mean(kag)) if kag else 0.0

    # ----------------------------------------------------------------
    # 4. per-station fil/syn files + dsretc.lst (for plot.misfit/text/emsc)
    # ----------------------------------------------------------------
    allstat_lines = setup["allstat_lines"]
    nr = setup["nr"]
    keep_mask = np.ones(nr * 3 * 1024, dtype=np.bool_)
    keep_mask[setup["removeIdx"]] = False
    ch_keep = keep_mask.reshape(nr * 3, 1024)[:, 0]
    kept_rows = np.where(ch_keep)[0]
    row_of = {(int(idx // 3), int(idx % 3)): j for j, idx in enumerate(kept_rows)}

    Dsyn = np.asarray(result["Dsynth"])
    tcol = np.arange(1024) * Dt

    for i, line in enumerate(allstat_lines):
        sta = line.split()[0]
        cols_obs, cols_syn = [tcol], [tcol]
        for rec in range(3):                      # N, E, Z
            j = row_of.get((i, rec))
            if j is None:
                cols_obs.append(np.zeros(1024))
                cols_syn.append(np.zeros(1024))
            else:
                cols_obs.append(Dobs[j * 1024:(j + 1) * 1024])
                cols_syn.append(Dsyn[j * 1024:(j + 1) * 1024])
        np.savetxt(os.path.join(pseudo, sta + 'fil.dat'), np.column_stack(cols_obs), fmt='%.6e')
        np.savetxt(os.path.join(pseudo, sta + 'syn.dat'), np.column_stack(cols_syn), fmt='%.6e')

    bp2 = result["BestPlane2"]
    with open(os.path.join(pseudo, 'dsretc.lst'), 'w') as f:
        f.write('Gisola2 Python inversion\n\n'
                '   NP1: strike {:.0f}  dip {:.0f}  rake {:.0f}\n'
                '   NP2: strike {:.0f}  dip {:.0f}  rake {:.0f}\n'
                '   P axis: azm {:.0f} plunge {:.0f}   T axis: azm {:.0f} plunge {:.0f}\n'
                .format(bp1[0], bp1[1], bp1[2], bp2[0], bp2[1], bp2[2],
                        result["BestP"][0], result["BestP"][1],
                        result["BestT"][0], result["BestT"][1]))

    # ----------------------------------------------------------------
    # 5. QuakeML event (mirrors the original gatherResults, values ours)
    # ----------------------------------------------------------------
    fm = FocalMechanism()
    mt = MomentTensor()
    org = Origin()
    mag = Magnitude()

    horg = event.getOrigin(config.cfg, evt, config.cfg['Watcher']['Historical'])
    org.time = horg.time + float(result["BestShift"]) * (tl / NUM_OF_TIME_SAMPLES)
    _lat, _lon = cart2earth(horg.latitude, horg.longitude, bx, by)
    org.latitude = _lat
    org.longitude = _lon
    org.depth = bz * 1000
    org.depth_type = 'from moment tensor inversion'
    org.time_fixed = False
    org.epicenter_fixed = False
    org.origin_type = 'centroid'
    org.creation_info = CreationInfo(agency_id=config.cfg['Citation']['Agency'],
                                     author=config.cfg['Citation']['Author'],
                                     version=config.cfg['Citation']['Version'],
                                     creation_time=UTCDateTime.now())
    mag.origin_id = org.resource_id
    mag.magnitude_type = 'Mw'
    mag.creation_info = org.creation_info
    mt.creation_info = org.creation_info
    mt.derived_origin_id = org.resource_id
    mt.moment_magnitude_id = mag.resource_id
    mt.category = 'regional'
    mt.inversion_type = 'zero trace'
    org.evaluation_mode = 'automatic'
    org.evaluation_status = 'preliminary'

    mag.mag = float(result["BestMw"])
    mt.scalar_moment = float(result["BestMo"])
    mt.iso = float(result["BestISO"]) / 100.0
    mt.double_couple = float(result["BestDC"]) / 100.0
    mt.clvd = float(result["BestCLVD"]) / 100.0
    fm.nodal_planes = NodalPlanes(
        nodal_plane_1=NodalPlane(strike=float(bp1[0]), dip=float(bp1[1]), rake=float(bp1[2])),
        nodal_plane_2=NodalPlane(strike=float(bp2[0]), dip=float(bp2[1]), rake=float(bp2[2])))
    fm.principal_axes = PrincipalAxes(
        p_axis=Axis(azimuth=float(result["BestP"][0]), plunge=float(result["BestP"][1])),
        t_axis=Axis(azimuth=float(result["BestT"][0]), plunge=float(result["BestT"][1])),
        n_axis=Axis(azimuth=float(result["BestB"][0]), plunge=float(result["BestB"][1])))
    mt.variance = float(result["VR"]) / 100.0
    mt.variance_reduction = float(result["VR"])

    # NED (x=N, y=E, z=D) -> RTP/USE (r=Up, t=South, p=East)
    M = np.asarray(result["Tensor"], dtype=float)
    mt.tensor = Tensor(m_rr=M[2, 2], m_tt=M[0, 0], m_pp=M[1, 1],
                       m_rt=M[0, 2], m_rp=-M[1, 2], m_tp=-M[0, 1])

    minsn = float(np.min(result["Singulars"]))
    maxsn = float(np.max(result["Singulars"]))
    conum = float(result["Condition"])

    text = allstat_lines
    mt.data_used = [DataUsed(wave_type='combined', station_count=len(text),
                             component_count=sum(int(c) for sta in text for c in sta.split()[2:5]),
                             shortest_period=1 / float(text[0].split()[-1]),
                             longest_period=1 / float(text[0].split()[-4]))]

    st = read(os.path.join(workdir, 'streams_corrected.mseed'), headonly=True)
    with open(os.path.join(workdir, 'locations'), 'r') as f:
        stationinfo = f.readlines()
    text2 = list(filter(lambda x: int(x.split()[1]) and (int(x.split()[2]) or
                 int(x.split()[3]) or int(x.split()[4])), text))
    text2 = [_[0] for _ in text2]   # NOTE: replicates the ORIGINAL gatherResults
    # behaviour exactly (first character), so site values match the old pipeline
    stationinfo = list(map(eval, stationinfo))
    stationinfo2 = list(filter(lambda x: x[0] not in text2, stationinfo))
    dist = [float(_[4][0]) / 1000 for _ in stationinfo2]
    azm = np.array([float(_[4][1]) for _ in stationinfo2])
    azm.sort()
    azm = azm - np.roll(azm, 1)
    mag.azimuthal_gap = max(np.where(azm <= 0, 360 + azm, azm))

    components = AttribDict()
    components.namespace = 'custom'
    components.value = AttribDict()
    i = 1
    for li, line in enumerate(text):
        w = [0, 0, 0]
        tr = st.select(station=line.split()[0])[0]
        stainfo = [_ for _ in stationinfo if _[0] == tr.stats.station][0]
        sta, sw, w[0], w[1], w[2], *freqs = line.split()
        tr.stats['frequencies'] = freqs
        tr.stats['distance'] = round(float(stainfo[4][0]) / 1000, 1)
        tr.stats['latitude'] = float(stainfo[2])
        tr.stats['longitude'] = float(stainfo[3])
        for j, orient in enumerate(['N', 'E', 'Z']):
            tr.stats['channel'] = tr.stats.channel[:2] + orient
            tr.stats['weight'] = int(int(sw) and int(w[j]))
            rj = row_of.get((li, j))
            if tr.stats['weight'] and rj is not None:
                tr.stats['variance'] = calculateVariance(
                    Dobs[rj * 1024:(rj + 1) * 1024], Dsyn[rj * 1024:(rj + 1) * 1024], tl)
                fm.waveform_id.append(WaveformStreamID(
                    network_code=tr.stats.network, station_code=tr.stats.station,
                    location_code=tr.stats.location, channel_code=tr.stats.channel))
            else:
                tr.stats['variance'] = 'None'
                tr.stats['weight'] = 0
            for element in ['network', 'station', 'location', 'channel', 'latitude',
                            'longitude', 'variance', 'weight', 'distance', 'frequencies']:
                if element == 'network':
                    components.value['component_' + str(i)] = AttribDict()
                    components.value['component_' + str(i)].namespace = 'custom'
                    components.value['component_' + str(i)].value = AttribDict()
                components.value['component_' + str(i)].value[element] = AttribDict()
                components.value['component_' + str(i)].value[element].namespace = 'custom'
                components.value['component_' + str(i)].value[element].type = 'attribute'
                components.value['component_' + str(i)].value[element].value = tr.stats[element]
            i += 1

    fm.extra = AttribDict()
    fm.extra.components = components

    mag.station_count = mt.data_used[0].station_count
    mag.evaluation_mode = org.evaluation_mode
    mag.evaluation_status = org.evaluation_status
    fm.azimuthal_gap = mag.azimuthal_gap
    fm.triggering_origin_id = horg.resource_id
    fm.evaluation_mode = org.evaluation_mode
    fm.evaluation_status = org.evaluation_status
    fm.creation_info = org.creation_info
    fm.moment_tensor = mt
    org.quality = OriginQuality(used_station_count=mt.data_used[0].station_count,
                                azimuthal_gap=mag.azimuthal_gap,
                                minimum_distance=kilometers2degrees(min(dist)),
                                maximum_distance=kilometers2degrees(max(dist)))

    if mt.variance >= 0.6 and mt.data_used[0].station_count > 4:
        quality = 'A'
    elif (mt.variance >= 0.4 and mt.variance < 0.6 and
          mt.data_used[0].station_count >= 4) or (mt.variance >= 0.7 and
          (mt.data_used[0].station_count == 2 or mt.data_used[0].station_count == 3)):
        quality = 'B'
    elif (mt.variance >= 0.15 and mt.variance < 0.4 and
          mt.data_used[0].station_count > 4) or (mt.variance >= 0.2 and
          mt.variance < 0.4 and mt.data_used[0].station_count == 4) or \
         (mt.variance >= 0.2 and mt.variance < 0.7 and
          mt.data_used[0].station_count == 3) or (mt.variance >= 0.3 and
          mt.variance < 0.7 and mt.data_used[0].station_count == 2):
        quality = 'C'
    else:
        quality = 'D'
    if mt.clvd <= 0.2:
        quality += '1'
    elif mt.clvd <= 0.5:
        quality += '2'
    elif mt.clvd <= 0.8:
        quality += '3'
    else:
        quality += '4'

    mt.extra = OrderedDict()
    mt.extra['correlation'] = {'namespace': 'custom', 'value': bestcorr}
    mt.extra['quality'] = {'namespace': 'custom', 'value': quality}
    mt.extra['min_singular'] = {'namespace': 'custom', 'value': minsn}
    mt.extra['max_singular'] = {'namespace': 'custom', 'value': maxsn}
    mt.extra['condition_number'] = {'namespace': 'custom', 'value': conum}
    mt.extra['stvar'] = {'namespace': 'custom', 'value': stvar}
    mt.extra['fmvar'] = {'namespace': 'custom', 'value': fmvar}

    evt.focal_mechanisms.append(fm)
    evt.origins.append(org)
    evt.magnitudes.append(mag)
    evt.preferred_focal_mechanism_id = fm.resource_id

    evt.write(os.path.join(config.outputdir, 'event.xml'), format="QUAKEML")
    try:
        evt.write(os.path.join(config.outputdir, 'event_sc.xml'), format="SC3ML")
    except Exception:
        config.logger.info('SC3ML export failed (QuakeML written)')

    config.logger.info('exportResults: event.xml, solutions, correlations written')


@config.time
def renderSite():
    """
    Stage 1: the original figures + event page, fed by exportResults().
    Reuses the unmodified plot.py / web.py.
    """
    import plot as plotmod
    import web as webmod
    from obspy.core.event import read_events as _read_events

    config.revise = False
    config.evt = _read_events(os.path.join(config.outputdir, 'event.xml'))[0]

    for fn in (plotmod.beachball, plotmod.atlas, plotmod.streams, plotmod.misfit,
               plotmod.top, plotmod.northeast, plotmod.time, plotmod.text, plotmod.emsc):
        try:
            fn()
        except Exception:
            import traceback
            traceback.print_exc()
            print('WARNING: plot {} failed (site continues)'.format(getattr(fn, '__name__', '?')))
        plt.close('all')

    webmod.renderEvent()
    try:
        webmod.renderHome()
    except Exception:
        import traceback
        traceback.print_exc()
        print('WARNING: renderHome failed (event page is written)')

    # Embed bootstrap section into index.html if bootstrapping was run
    if _BOOT_STATS is not None:
        try:
            _renderBootstrapPage(*_BOOT_STATS)
        except Exception:
            import traceback
            traceback.print_exc()
            print('WARNING: bootstrap section embedding failed (index.html still valid)')

    config.logger.info('renderSite: index.html written in ' + config.outputdir)


def _renderBootstrapPage(plotdir, st, result):
    """
    Embeds bootstrap sections directly into index.html (the single final page),
    in the SAME visual style as the event page (Montserrat, centred bold titles
    with a thin grey rule), placed right after the "Best Moment Tensor Solution"
    section and BEFORE the Gisola logo footer. Sections, in order:
       1. Moment Tensor Statistics  - fuzzy uncertainty beachball (left) +
          MT-parameter statistics (NP1/NP2, DC, Mw, Mo).
       2. Location & Time Statistics - x/y/z/centroid-time table + their plot.
       3. Uncertainty Visualization  - Hudson plot + Mo/Mw histograms.
    """
    import re
    idx_path = os.path.join(config.outputdir, 'index.html')
    if not os.path.isfile(idx_path):
        print('index.html not found - run renderSite() first; bootstrap page skipped')
        return
    html = open(idx_path, encoding='utf-8').read()
    rel = os.path.basename(plotdir)

    FONT = "'Montserrat',sans-serif"
    BLUE = '#3AAEE0'

    # ---- style helpers that mirror the event-page look ----
    def title(text):
        # thin grey rule + centred bold h2, exactly like the template panels
        return (
            '<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
            'border="0" style="font-family:{f}"><tbody><tr>'
            '<td style="padding:5px" align="left">'
            '<table height="0px" align="center" border="0" cellpadding="0" '
            'cellspacing="0" width="100%" style="border-collapse:collapse;'
            'border-top:1px solid #BBBBBB"><tbody><tr style="vertical-align:top">'
            '<td style="font-size:0px;line-height:0px">&#160;</td></tr></tbody>'
            '</table></td></tr></tbody></table>'
            '<h2 style="margin:0;color:#000;line-height:140%;text-align:center;'
            'word-wrap:break-word;font-weight:normal;font-family:{f};font-size:20px">'
            '<strong>{t}</strong></h2>').format(f=FONT, t=text)

    def img(f, css):
        if not os.path.isfile(os.path.join(plotdir, f)):
            return ''
        return ('<a href="{r}/{f}" target="_blank"><img src="{r}/{f}" alt="{f}" '
                'style="{c}"></a>').format(r=rel, f=f, c=css)

    def kv(label, value):
        return ('<p style="font-size:14px;line-height:140%;margin:5px 0;'
                'font-family:{f}"><strong>{l}:</strong> {v}</p>').format(
            f=FONT, l=label, v=value)

    # ---- section 1: Moment Tensor Statistics (beachball left, stats right) ----
    p1, p2 = st['p1_med'], st['p2_med']
    p1s, p2s = st['p1_std'], st['p2_std']
    dc_med, _dcmean, dc_sd, _dclo, _dchi = st['DC']
    mw_med, _mwmean, mw_sd, _mwlo, _mwhi = st['Mw']
    mo_med = st['Mo'][0]

    mt_stats = (
        kv('NP1 strike/dip/rake',
           '{:.0f} / {:.0f} / {:.0f} &nbsp;(&plusmn;{:.0f}/{:.0f}/{:.0f})'.format(
               p1[0], p1[1], p1[2], p1s[0], p1s[1], p1s[2]))
        + kv('NP2 strike/dip/rake',
             '{:.0f} / {:.0f} / {:.0f} &nbsp;(&plusmn;{:.0f}/{:.0f}/{:.0f})'.format(
                 p2[0], p2[1], p2[2], p2s[0], p2s[1], p2s[2]))
        + kv('DC (%)', '{:.1f} &nbsp;(&plusmn;{:.1f})'.format(dc_med, dc_sd))
        + kv('Mw', '{:.2f} &nbsp;(&plusmn;{:.2f})'.format(mw_med, mw_sd))
        + kv('Mo (Nm)', '{:.3e}'.format(mo_med))
        + kv('Best solutions',
             '{} (from {} resamplings)'.format(st['n_solutions'], st['n_iterations']))
        + kv('Best VR (%)', '{:.1f}'.format(st['vr_max'])))

    mt_panel = (
        '<div style="display:flex;flex-wrap:wrap;gap:20px;align-items:center;'
        'justify-content:center;max-width:900px;margin:6px auto">'
        '<div style="flex:1 1 300px;min-width:260px;text-align:center">'
        + img('beachball_fuzzy.png', 'width:100%;max-width:320px;height:auto')
        + '</div>'
        '<div style="flex:1 1 340px;min-width:280px;text-align:left">'
        + mt_stats + '</div></div>')

    # ---- section 2: Location & Time statistics table + plot ----
    TD = ('border:1px solid #b8c4cc;padding:6px 12px;font-size:13px;'
          'text-align:center;font-family:' + FONT)
    TH = ('background:' + BLUE + ';color:#fff;padding:7px 12px;font-size:13px;'
          'font-family:' + FONT)

    def row(label, key, fmt, sfmt):
        med, mean, sd, lo, hi = st[key]
        return ('<tr><td style="{td};text-align:left"><strong>{l}</strong></td>'
                '<td style="{td}">{m}</td><td style="{td}">{a} &plusmn; {s}</td>'
                '<td style="{td}">[{lo}, {hi}]</td></tr>').format(
            td=TD, l=label, m=fmt.format(med), a=fmt.format(mean),
            s=sfmt.format(sd), lo=fmt.format(lo), hi=fmt.format(hi))

    loc_rows = (row('X (km)', 'x', '{:.1f}', '{:.1f}')
                + row('Y (km)', 'y', '{:.1f}', '{:.1f}')
                + row('Z depth (km)', 'z', '{:.1f}', '{:.1f}')
                + row('Centroid time (s)', 'time', '{:.2f}', '{:.2f}'))
    loc_table = (
        '<table style="border-collapse:collapse;margin:8px auto;max-width:760px;'
        'width:100%">'
        '<tr><th style="{th};text-align:left">Quantity</th>'
        '<th style="{th}">Median</th><th style="{th}">Mean &plusmn; std</th>'
        '<th style="{th}">95% CI</th></tr>{r}</table>').format(th=TH, r=loc_rows)
    loc_plot = ('<div style="text-align:center;margin-top:8px">'
                + img('hist_centroid_time.png', 'width:100%;max-width:880px;height:auto')
                + '</div>')

    # ---- section 3: Uncertainty Visualization (Hudson + Mo/Mw) ----
    unc = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;justify-content:center;'
        'align-items:flex-start;max-width:900px;margin:6px auto">'
        '<div style="flex:1 1 380px;min-width:300px;text-align:center">'
        + img('hudson_median.png', 'width:100%;max-width:440px;height:auto') + '</div>'
        '<div style="flex:1 1 380px;min-width:300px;text-align:center">'
        + img('hist_magnitudes.png', 'width:100%;max-width:440px;height:auto') + '</div>'
        '</div>')

    intro = ('<p style="text-align:center;font-size:14px;line-height:140%;margin:8px 0;'
             'color:#333;font-family:' + FONT + '">'
             + '{n} best-fitting solutions from {it} bootstrap resamplings '
               '(best VR {vr:.1f}%).'.format(
                   n=st['n_solutions'], it=st['n_iterations'], vr=st['vr_max'])
             + '</p>')

    section = (
        '<div id="bootstrap-results" style="max-width:900px;margin:0 auto;'
        'padding:10px 20px 20px;background:#ffffff;font-family:' + FONT + '">'
        + title('Bootstrap Uncertainty Analysis') + intro
        + title('Moment Tensor Statistics') + mt_panel
        + title('Location &amp; Time Statistics') + loc_table + loc_plot
        + title('Uncertainty Visualization') + unc
        + '</div>')

    # Insert right after the last content section and BEFORE the Gisola logo
    # footer (the final u-row-container in the page), so the logo stays last.
    anchor = html.rfind('<div class="u-row-container"')
    if anchor == -1:
        mm = list(re.finditer(r'</body\s*>', html, re.I))
        anchor = mm[-1].start() if mm else len(html)
    boot_html = html[:anchor] + section + html[anchor:]
    with open(idx_path, 'w', encoding='utf-8') as f:
        f.write(boot_html)
    config.logger.debug('Bootstrap section embedded into: %s', idx_path)


def calculateRevisedInversions(_cfg,logger,workdir,bestinvdir, evt_obj=None, restore=False):

    allstat='allstat'+bestinvdir.split('.')[0]+'.dat'+('.revise' if not restore else '')
    inpinv='inpinv'+bestinvdir.split('.')[1]+'.dat'
    igrdat=bestinvdir.split('.')[3]
    icrustal=bestinvdir.split('.')[2]
    igrid=bestinvdir.split('.')[4]
    grhes_pattern='gr.{}.{}.{}.*.hes'.format(icrustal,igrdat,igrid)

    # get origin from event object
    current_origin = None
    if evt_obj:
        try:
            current_origin = evt_obj.preferred_origin()
            if not current_origin and evt_obj.origins:  # Fallback if preferred_origin() is None but origins list exists
                current_origin = evt_obj.origins[0]
            
            if current_origin:
                 logger.info(f"Event Origin: Lat {current_origin.latitude:.4f}, Lon {current_origin.longitude:.4f}")
            else:
                logger.info("evt_obj was passed but no origin could be extracted. Keyinv decision might rely on default values.")
        except Exception as e:
            logger.info(f"Error extracting origin from passed evt_obj: {e}. Keyinv decision might rely on default values.")
    else:
        logger.info("Event origin data (evt_obj or config.evt) not available for IsotropicRule check in calculateRevisedInversions. Keyinv decision might rely on default values if geobox rule is present.")
    
    keyinv_value = config.get_keyinv(_cfg, current_origin)

    for grhes_file in sorted(glob.glob(os.path.join(workdir,'greens',grhes_pattern))):
        isource= grhes_file.split('.')[-2]

        command='{} {} {} {} {} {} {} {} {} {} \n'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)),_cfg['Inversion']['ExePath']),
                os.path.join('..', '..','allstat', allstat),
                os.path.join('..', '..','inpinv', inpinv),
                os.path.join('..', '..','grdat', 'grdat'+igrdat+'.hed'),
                os.path.join('..', '..','greens', os.path.basename(grhes_file)),
                os.path.join('..', '..','crustals', 'crustal'+icrustal+'.dat'),
                os.path.join('..', '..','sources', 'grid'+igrid, 'source'+isource+'.dat'),
                os.path.join('..', '..','station.dat'),
                os.path.join('..', '..','raw', igrdat),
                str(keyinv_value))

        invdir_path = os.path.join(workdir,'inversions','{}.{}.{}.{}.{}.{}'.format(allstat.split('.')[0][7:],
               inpinv.split('.')[0][6:], icrustal, igrdat, igrid, isource))
        os.makedirs(invdir_path, exist_ok=True)

        proc=subprocess.Popen(command, cwd=invdir_path, shell=True, universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out,err=proc.communicate()
        logger.info(err)
        logger.info(out)



@config.time
def gatherResults(cfg=None, evt=None, workdir=None, bestinvdir=None, revise=False):
    """
    Retrieve all the necessary info derived from Moment Tensor calculation
    """
    if revise:
        config.evt=evt
        config.cfg=cfg
        config.workdir=workdir
        config.bestinvdir=bestinvdir
        config.revise=True
    else:
        workdir=config.workdir
        evt=config.evt

        # gather all inv1.dat and find the best (based on correlation)
        corr=[]
        inversions=sorted(os.listdir(os.path.join(workdir,'inversions')))
        for inversion in inversions:
            with open(os.path.join(workdir,'inversions', inversion,'inv1.dat')) as _:
                content=_.readlines()
            for i,line in enumerate(content):
                if line.startswith(' Selected source position for subevent'):
                    corr.append(float(content[int(content[i+1].split()[1])+\
                    2].split()[2]))
                    break

        # open best corr inv1 file
        config.bestinvdir=inversions[corr.index(max(corr))]

    with open(os.path.join(workdir,'inversions',config.bestinvdir,'inv1.dat')) as _:
        content=_.readlines()

    # get tl value that was used for the best inversion
    with open(os.path.join(workdir,'grdat','grdat'+config.bestinvdir.split('.')[3]+\
    '.hed'),'r') as f:
        tl=float(f.readlines()[2].split('=')[1])

    config.besttl=tl

    # number of max sources per source file
    sourcefiles=sorted(glob.glob(os.path.join(workdir,'sources','grid'+\
                config.bestinvdir.split('.')[-2]+'/source*.dat')))

    with open(sourcefiles[0],'r') as f:
        maxsources=len(f.readlines())

    # read all source files from all inversions
    srctext=[]
    for i,src in enumerate(sourcefiles):
        with open(src,'r') as f:
            srctext+=f.readlines()

    # read all inv1.dat files from all inversions of the best inversion
    invtext=[]
    for i,inv1 in enumerate(sorted(glob.glob(os.path.join(workdir,\
    'inversions','.'.join(config.bestinvdir.split('.')[:-1])+'.*/inv1.dat')))):
        with open(inv1,'r') as f:
            temp=f.readlines()[3:]

        for i,line in enumerate(temp):
            if line.startswith(' Selected source position for subevent'):
                invtext+=temp[:i-1]
                break

    # merge source and inv1.dat files from all inversions
    # of the best inversion to one list
    # convert all elements to floats
    solutions=[]
    for i, source in enumerate(srctext):
        #print(source.split()+invtext[i].split()[1:])
        solutions+=[list(map(float,source.split()+invtext[i].split()[1:]))]

    # find best
    # ['3', '-2.0000', '0.0000', '8.6000', '89', '0.550889', '0.1180E+15', 
    # '84.198', '336', '68', '-162', '239', '74', '-22']
    best=max(solutions, key=lambda x: x[5])

    # read all time inversions from corr00.dat
    # 1   -4.0400    0.5474        115    46   -81        283    43   -99   
    # 81.22   -0.00 0.128037E-07 0.857937E+16
    timetext=[]
    for corr in sorted(glob.glob(os.path.join(workdir,\
    'inversions','.'.join(config.bestinvdir.split('.')[:-1])+'.*/corr00.dat'))):
        with open(corr,'r') as f:
            text=f.readlines()[2:]
        pos=int(os.path.basename(os.path.dirname(corr)).split('.')[-1])*maxsources
        text=list(map(lambda x: list(map(float,x.split())),text))
        timetext+=list(map(lambda x: [x[0]+pos]+x[1:],text))

    # calculate quality metrics
    # stvar
    thres=0.9*best[5]
    threstimetext=list(filter(lambda x: x[2]>=thres, timetext))
    stvar=len(threstimetext)/len(timetext)

    # fmvar
    threstimetext=list(map(lambda x: [*best[8:11]]+ [*x[6:9]], threstimetext))
    with multiprocessing.Pool() as p:
       res=p.map(kaganCalc, threstimetext)
    fmvar=np.mean(res)

    # merge all time inversions from best inversions to one dir
    srcs=[sol[0] for sol in solutions]
 
    # keep all correlations with the best's x,y
    correlations=list(filter(lambda x: x if \
    solutions[srcs.index(x[0])][1:3]==best[1:3] else None, timetext))
    # attach depth value instead of src id
    correlations=list(map(lambda x: solutions[srcs.index(x[0])][0:4]+x[1:], \
    correlations))

    # save all best time solutions to one file
    with open(os.path.join(workdir,'output',('solutions' if not revise else 'solutions.revise')),'w') as f:
        f.writelines('{}\n'.format(x)[1:-2]+'\n' for x in sorted(solutions))

    config.solutions=solutions

    # save all time solutions to one file
    with open(os.path.join(workdir,'output',('correlations' if not revise else 'correlations.revise')),'w') as f:
        f.writelines('{}\n'.format(x)[1:-2]+'\n' for x in sorted(correlations))

    config.correlations=correlations

    # save only the best inversion
    # create ObsPy objects based on QuakeML standard
    fm=FocalMechanism()
    mt=MomentTensor()
    org=Origin()
    mag=Magnitude()

    org.time=event.getOrigin(config.cfg,evt,config.cfg['Watcher']['Historical']).time+float(best[4])*(tl/NUM_OF_TIME_SAMPLES)
    _lat, _lon=cart2earth(event.getOrigin(config.cfg,evt,config.cfg['Watcher']['Historical']).latitude,
                          event.getOrigin(config.cfg,evt,config.cfg['Watcher']['Historical']).longitude, best[1], best[2])
    org.latitude=_lat
    org.longitude=_lon
    org.depth= best[3]*1000 # meters
    org.depth_type='from moment tensor inversion'
    org.time_fixed=False
    org.epicenter_fixed=False
    org.origin_type='centroid'
    org.creation_info=CreationInfo(agency_id=config.cfg['Citation']['Agency'], \
                     author=config.cfg['Citation']['Author'], \
                     version=config.cfg['Citation']['Version'], \
                     creation_time=UTCDateTime.now())
    mag.origin_id=org.resource_id
    mag.magnitude_type='Mw'
    mag.creation_info=org.creation_info
    mt.creation_info=org.creation_info
    mt.derived_origin_id=org.resource_id
    mt.moment_magnitude_id=mag.resource_id
    mt.category='regional'
    mt.inversion_type='zero trace' # aka deviatoric
    mt.iso=0
    org.evaluation_mode='automatic'
    org.evaluation_status=('preliminary' if not revise else 'reviewed')

    # fill Focal Mechanism and Moment Tensor values from best inv1.dat file
    for i,line in enumerate(content):
            if line.startswith(' SINGULAR values, incl. vardat'):
                minsn, maxsn, conum = content[i+1].split()

            elif line.startswith(' moment (Nm)'):
                mag.mag=float(content[i+1].split()[2])
                mt.scalar_moment=float(line.split()[2])
                mt.iso = float(content[i+2].split()[3])/100.0
                mt.double_couple=float(content[i+3].split()[3])/100.0
                mt.clvd=float(content[i+4].split()[3])/100.0
                nd=content[i+5].split() + content[i+6].split()
                fm.nodal_planes=NodalPlanes(nodal_plane_1=NodalPlane(\
                                strike=float(nd[1]), dip=float(nd[2]), 
                                rake=float(nd[3])), \
                                nodal_plane_2=NodalPlane(strike=float(nd[5]), \
                                dip=float(nd[6]), rake=float(nd[7])))
                ax=content[i+7].split()+content[i+8].split()+content[i+9].split()
                fm.principal_axes=PrincipalAxes(p_axis=Axis(azimuth=float(ax[4]),\
                                  plunge=float(ax[5])), \
                                  t_axis=Axis(azimuth=float(ax[10]), \
                                  plunge=float(ax[11])), \
                                  n_axis=Axis(azimuth=float(ax[16]), \
                                  plunge=float(ax[17])))
             
            elif line.startswith(' varred='):
                mt.variance=float(line.split()[1])
                mt.variance_reduction=float(line.split()[1])*100

    # open best corr inv3.dat file and get Tensor 6 values
    with open(os.path.join(workdir,'inversions', config.bestinvdir, 'inv3.dat')) as _:
        line=list(map(float,_.readlines()[0].split()))
    mt.tensor=Tensor(m_rr=float(line[2]), m_tt=float(line[3]), m_pp=float(line[4]),
                     m_rt=float(line[5]), m_rp=float(line[6]), m_tp=float(line[7]))

    # read best allstat.dat
    with open(os.path.join(workdir,'allstat','allstat'+ \
    config.bestinvdir.split('.')[0]+'.dat'+('.revise' if revise else '')), 'r') as _f:
        text=_f.readlines()
    mt.data_used=[DataUsed(wave_type='combined', station_count=len(text), \
                          component_count=sum([int(comp) for sta in text \
                          for comp in sta.split()[2:5]]), \
                          shortest_period=1/float(text[0].split()[-1]), \
                          longest_period=1/float(text[0].split()[-4]))]

    # add waveform info from Stream object
    st=read(os.path.join(workdir,'streams_corrected.mseed'), headonly=True)

    # read station, priority, distance, azimuth and back azimuth
    with open(os.path.join(workdir,'locations'), 'r') as f:
        stationinfo=f.readlines()
   
    # filter station names to those that are being used in the inversion
    text2=list(filter(lambda x: int(x.split()[1]) and (int(x.split()[2]) \
    or int(x.split()[3]) or int(x.split()[4])), text))
    # get only the station name
    text2=[_[0] for _ in text2]

    stationinfo=list(map(eval,stationinfo))

    # filter stationsinfo to those that are being used in the inversion
    stationinfo2=list(filter(lambda x: x[0] not in text2, stationinfo))

    # get all distances together only from used stations
    dist=[float(_[4][0])/1000 for _ in stationinfo2]
    azm=np.array([float(_[4][1]) for _ in stationinfo2])

    # find station azimuthal gap
    azm.sort()
    # circular shift and substract
    azm=azm-np.roll(azm,1)
    mag.azimuthal_gap=max(np.where(azm<=0, 360+azm, azm))

    components = AttribDict()
    components.namespace = 'custom'
    components.value = AttribDict()
    i=1
    # for all stations (used and unused)
    for line in text:
        w=[0,0,0]
        tr=st.select(station=line.split()[0])[0]
        stainfo=[_ for _ in stationinfo if _[0]==tr.stats.station][0]
        sta, sw, w[0], w[1], w[2], *freqs=line.split()
        tr.stats['frequencies']=freqs
        tr.stats['distance']=round(float(stainfo[4][0])/1000,1)
        tr.stats['latitude']=float(stainfo[2])
        tr.stats['longitude']=float(stainfo[3])
        file=os.path.join(workdir,'inversions',config.bestinvdir,sta+'fil.dat')
        obs={'Time':None, 'N':None, 'E':None, 'Z':None}
        syn={'Time':None, 'N':None, 'E':None, 'Z':None}
        # load observed data
        obs['Time'], obs['N'], obs['E'], obs['Z'] = np.loadtxt(file,unpack=True, usecols=[0,1,2,3])
        # load synthetic data
        syn['Time'], syn['N'], syn['E'], syn['Z'] = np.loadtxt(file[:-7]+'syn.dat',unpack=True, usecols=[0,1,2,3])

        for j, orient in enumerate(['N', 'E', 'Z']):
            tr.stats['channel']=tr.stats.channel[:2]+orient
            tr.stats['weight']=int(int(sw) and int(w[j]))
            tr.stats['variance']=calculateVariance(obs[orient], syn[orient], tl) if tr.stats['weight'] else 'None'

            if tr.stats['weight']:
                fm.waveform_id.append(WaveformStreamID(network_code=tr.stats.network, \
                station_code=tr.stats.station, location_code=tr.stats.location, \
                channel_code=tr.stats.channel))

            for element in ['network', 'station', 'location', 'channel', 'latitude', 'longitude', 'variance', 'weight', 'distance', 'frequencies']:
                if element=='network':
                    components.value['component_'+str(i)]=AttribDict()
                    components.value['component_'+str(i)].namespace='custom'
                    components.value['component_'+str(i)].value=AttribDict()
                components.value['component_'+str(i)].value[element]=AttribDict()
                components.value['component_'+str(i)].value[element].namespace='custom'
                components.value['component_'+str(i)].value[element].type='attribute'
                components.value['component_'+str(i)].value[element].value=tr.stats[element]
            i+=1

    fm.extra=AttribDict()
    fm.extra.components=components

    mag.station_count=mt.data_used[0].station_count
    mag.evaluation_mode=org.evaluation_mode
    mag.evaluation_status=org.evaluation_status
    fm.azimuthal_gap=mag.azimuthal_gap
    fm.triggering_origin_id= event.getOrigin(config.cfg,evt,config.cfg['Watcher']['Historical']).resource_id
    fm.evaluation_mode=org.evaluation_mode
    fm.evaluation_status=org.evaluation_status
    fm.creation_info=org.creation_info
    fm.moment_tensor=mt
    org.quality=OriginQuality(used_station_count=mt.data_used[0].station_count, \
                              azimuthal_gap=mag.azimuthal_gap, \
                              minimum_distance=kilometers2degrees(min(dist)), \
                              maximum_distance=kilometers2degrees(max(dist)), \
                              median_distance=kilometers2degrees(median(dist)))

    # define quality value
    if mt.variance >= 0.6 and mt.data_used[0].station_count > 4:
        quality='A'
    elif (mt.variance >= 0.4 and mt.variance < 0.6 and \
    mt.data_used[0].station_count >= 4) or (mt.variance >= 0.7 and \
    (mt.data_used[0].station_count==2 or mt.data_used[0].station_count==3)):
        quality='B'
    elif (mt.variance >= 0.15 and mt.variance < 0.4 and \
    mt.data_used[0].station_count > 4) or (mt.variance >= 0.2 and \
    mt.variance < 0.4 and mt.data_used[0].station_count == 4) or \
    (mt.variance >= 0.2 and mt.variance < 0.7 and \
    mt.data_used[0].station_count == 3) or (mt.variance >= 0.3 and \
    mt.variance < 0.7 and mt.data_used[0].station_count == 2):
        quality='C'
    else:
        quality='D'

    if mt.clvd <= 0.2:
        quality+='1'
    elif mt.clvd > 0.2 and mt.clvd <= 0.5:
        quality+='2'
    elif mt.clvd > 0.5 and mt.clvd <= 0.8:
        quality+='3'
    elif mt.clvd > 0.8:
        quality+='4'

    mt.extra = OrderedDict()
    mt.extra['correlation'] = {'namespace': 'custom', 'value': best[5]}
    mt.extra['quality'] = {'namespace': 'custom', 'value': quality}
    mt.extra['min_singular'] = {'namespace': 'custom', 'value': minsn}
    mt.extra['max_singular'] = {'namespace': 'custom', 'value': maxsn}
    mt.extra['condition_number'] = {'namespace': 'custom', 'value': conum}
    mt.extra['stvar'] = {'namespace': 'custom', 'value': stvar}
    mt.extra['fmvar'] = {'namespace': 'custom', 'value': fmvar}

    if revise:
            # check and keep only ONE (this) revision
        _fm=list(filter(lambda x: x.evaluation_status=='reviewed',evt.focal_mechanisms))
        if _fm:
            evt.focal_mechanisms=list(filter(lambda x: not x.resource_id==_fm[0].resource_id, evt.focal_mechanisms))
            evt.origins=list(filter(lambda x: not x.resource_id==_fm[0].moment_tensor.derived_origin_id,evt.origins))
            evt.magnitudes=list(filter(lambda x: not x.origin_id==_fm[0].moment_tensor.derived_origin_id,evt.magnitudes))

            # this double for loop is needed in order to maintain the components attribdict
            for _ in evt.focal_mechanisms:
                for _i in range(1,len(_.extra.components.value)+1):
                    _.extra['components']['value']['component_'+str(_i)]['value']=AttribDict()

    evt.focal_mechanisms.append(fm)
    evt.origins.append(org)
    evt.magnitudes.append(mag)

    #if revise:
    #    if evt.preferred_focal_mechanism().moment_tensor.variance <= mt.variance:
    #        evt.preferred_focal_mechanism_id=fm.resource_id

    #else:
    evt.preferred_focal_mechanism_id=fm.resource_id

    evt.write(os.path.join(workdir,'output','event.xml'), format="QUAKEML")
    evt.write(os.path.join(workdir,'output','event_sc.xml'), format="SC3ML") 

def getGreens():
    """
    All steps required for Greens' Functions calculation
    """
    config.logger.info('Creating sources files based on Grid rules')
    config.logger.info(config.dump(config.gridRules))
    createSources()

    config.logger.info('Selecting crustal model files based on Crustal rules')
    config.logger.info(config.dump(config.crustalRules))
    createCrustals()

    config.logger.info('Creating Greens\' Functions configuration files (grdat.hed)')
    createGrdat()

    config.logger.info('Creating Stations file (station.dat)')
    createStations()

    config.logger.info('Performing Greens\' Functions Computation')
    calculateGreens()

def getInversions():
    """
    All steps required for Inversions calculation.
    Returns a dict with keys 'numba warmup' and 'inversion (kernel)'
    so the caller can record them as separate pipeline stages.
    """
    config.logger.info('Creating Inversions configuration files (inpinv) ' + \
    'based on Time-shift rules')
    config.logger.info(config.dump(config.shiftRules))
    createInpinv()

    config.logger.info('Select Stations for inversion (allstat)')
    createAllstat()

    config.logger.info('Creating raw data files for ISOLA parsing')
    createRaw()

    config.logger.info('Calculating inversions')
    _t_inv = time.time()
    calculateInversions2()
    inv_time = time.time() - _t_inv

    return {'inversion (kernel)': inv_time}

