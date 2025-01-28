import numpy as np
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import librosa
from pathlib import Path
from dotenv import load_dotenv


class FeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self._setup_environment()

    def _setup_environment(self):
        """Setup cache directories and environment variables"""
        # Load environment variables
        load_dotenv()

        # Create cache directory in project folder
        cache_dir = Path('./cache')
        cache_dir.mkdir(exist_ok=True)

        # Set environment variables for numba and librosa
        os.environ['NUMBA_CACHE_DIR'] = str(cache_dir)
        os.environ['LIBROSA_CACHE_DIR'] = str(cache_dir)

    def extract_features(self, audio):
        """Extract MFCC features from audio"""
        try:
            # Ensure audio is in the correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            # Extract MFCCs with error handling
            try:
                mfccs = librosa.feature.mfcc(
                    y=audio,
                    sr=self.sample_rate,
                    n_mfcc=20,
                    hop_length=int(self.sample_rate * 0.01)
                )
            except PermissionError:
                print("Warning: Cache permission error. Proceeding without caching.")
                # Disable caching temporarily
                os.environ['NUMBA_DISABLE_JIT'] = '1'
                mfccs = librosa.feature.mfcc(
                    y=audio,
                    sr=self.sample_rate,
                    n_mfcc=20,
                    hop_length=int(self.sample_rate * 0.01)
                )
                del os.environ['NUMBA_DISABLE_JIT']

            # Calculate delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
            return features.T
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None