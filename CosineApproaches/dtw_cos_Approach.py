import sounddevice as sd
import numpy as np
import os
import time
import wave
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import dct
from fastdtw import fastdtw
from numpy.linalg import norm

class EnhancedVoiceAuth:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 5
        self.channels = 1
        self.recordings_dir = "voice_samples"
        self.enrolled_dir = "enrolled_users"
        
        # MFCC parameters
        self.n_mfcc = 13
        self.n_mels = 40
        self.frame_length = 0.025  # 25ms
        self.frame_step = 0.01     # 10ms
        
        # Create directories
        for directory in [self.recordings_dir, self.enrolled_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def frame_signal(self, signal, frame_length, frame_step):
        """Create frames from signal"""
        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                 np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        indices = np.array(indices, dtype=np.int32)
        
        frames = pad_signal[indices]
        return frames

    def record_audio(self, filename):
        """Record audio and save to WAV file"""
        print("\nRecording will start in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print("🎤 Recording... Speak now!")
        
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16
        )
        sd.wait()
        print("✅ Recording complete!")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
        
        return recording

    def extract_mfcc(self, audio_data):
        """Extract MFCC features from audio"""
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_signal = np.append(
            audio_data[0],
            audio_data[1:] - pre_emphasis * audio_data[:-1]
        )

        # Framing
        frame_length = int(self.frame_length * self.sample_rate)
        frame_step = int(self.frame_step * self.sample_rate)
        
        frames = self.frame_signal(emphasized_signal, frame_length, frame_step)
        windowed_frames = frames * np.hamming(frame_length)
        
        # Compute FFT
        nfft = 512
        mag_frames = np.abs(np.fft.rfft(windowed_frames, nfft))
        pow_frames = ((1.0 / nfft) * (mag_frames ** 2))

        # Mel filterbank
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((nfft + 1) * hz_points / self.sample_rate)

        fbank = np.zeros((self.n_mels, int(np.floor(nfft / 2 + 1))))
        for m in range(1, self.n_mels + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # Apply DCT to get MFCCs
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        
        return mfcc

    def compare_features(self, features1, features2):
        """Compare features using DTW and cosine similarity with proper normalization"""
        # DTW distance calculation
        distance, _ = fastdtw(features1, features2)
    
        # Calculate the maximum possible distance between two feature sets
        # This is based on the maximum possible difference between corresponding points
        feature_dim = features1.shape[1]  # Number of MFCC coefficients
        max_point_distance = np.sqrt(feature_dim)  # Maximum distance between two normalized feature vectors
        max_length = max(len(features1), len(features2))
        theoretical_max_distance = max_length * max_point_distance
    
        # Normalize the distance to [0, 1] range
        normalized_distance = distance / theoretical_max_distance
    
        # Convert to similarity score (0 to 1)
        similarity = 1 - normalized_distance
    
        # Ensure the similarity score stays within [0, 1]
        similarity = np.clip(similarity, 0, 1)
    
        # Debug information
        print(f"Raw DTW distance: {distance:.2f}")
        print(f"Theoretical max distance: {theoretical_max_distance:.2f}")
        print(f"Normalized similarity score: {similarity:.2f}")
    
        return similarity

    def enroll_user(self, username):
        """Enroll a new user with 3 voice samples"""
        print(f"\n📝 Enrolling user: {username}")
        print("We'll record 3 samples of your voice.")
        print('Please read the following phrase each time:')
        print('"My voice is my secure password."')
        
        features_list = []
        
        for i in range(3):
            print(f"\nRecording sample {i+1} of 3")
            filename = os.path.join(self.recordings_dir, f"{username}_sample_{i+1}.wav")
            recording = self.record_audio(filename)
            features = self.extract_mfcc(recording.flatten())
            features_list.append(features)
        
        # Save all features (instead of average)
        np.save(os.path.join(self.enrolled_dir, f"{username}_enrolled.npy"), features_list)
        print(f"\n✅ Successfully enrolled {username}!")

    def verify_user(self, username):
        """Verify a user's identity"""
        enrolled_file = os.path.join(self.enrolled_dir, f"{username}_enrolled.npy")
        
        if not os.path.exists(enrolled_file):
            print("❌ User not found! Please enroll first.")
            return False
        
        print("\n🔒 Voice Verification Started")
        print('Please read the phrase:')
        print('"My voice is my secure password."')
        
        # Record verification attempt
        verify_file = os.path.join(self.recordings_dir, f"{username}_verify_{int(time.time())}.wav")
        recording = self.record_audio(verify_file)
        verify_features = self.extract_mfcc(recording.flatten())
        
        # Load enrolled features and compare with all samples
        enrolled_features_list = np.load(enrolled_file, allow_pickle=True)
        similarities = [self.compare_features(verify_features, ef) for ef in enrolled_features_list]
        best_similarity = max(similarities)
        
        threshold = 0.75
        
        if best_similarity > threshold:
            print(f"\n✅ Voice verified! Best similarity score: {best_similarity:.2f}")
            return True
        else:
            print(f"\n❌ Voice verification failed. Best similarity score: {best_similarity:.2f}")
            return False

def main():
    auth_system = EnhancedVoiceAuth()
    
    while True:
        print("\n=== Enhanced Voice Authentication System ===")
        print("1. Enroll new user")
        print("2. Verify user")
        print("3. Exit")
        
        try:
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == "1":
                username = input("Enter username to enroll: ").lower()
                auth_system.enroll_user(username)
            
            elif choice == "2":
                username = input("Enter username to verify: ").lower()
                auth_system.verify_user(username)
            
            elif choice == "3":
                print("\nGoodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()