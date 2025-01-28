import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import sounddevice as sd
import soundfile as sf
import os
import pickle
import time
from pathlib import Path
import warnings
import whisper 
from datetime import datetime
import json