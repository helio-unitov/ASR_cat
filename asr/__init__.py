import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from astropy.visualization import time_support
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm.autonotebook import tqdm
from IPython.display import clear_output, display
import pandas as pd
from scipy.signal import argrelextrema
import time
import os
from decimal import Decimal
import gc
import seaborn as sns
import asr