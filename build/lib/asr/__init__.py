import matplotlib.pyplot as plt
import numpy as np
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm
import pandas as pd
from scipy.signal import argrelextrema
import os
import gc
import asr