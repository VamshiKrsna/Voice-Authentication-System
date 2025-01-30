# In this we try reading a paragraph to authorize our voice
# PERFECT - works so well.

import sounddevice as sd
import numpy as np
import os
import time
import wave
from scipy import signal
from scipy.io import wavfile

class SimpleVoiceAuth:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 10  # Increased duration for a paragraph
        self.channels = 1
        self.recordings_dir = "voice_samples"
        self.enrolled_dir = "enrolled_users"
        
        for directory in [self.recordings_dir, self.enrolled_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def record_audio(self, filename):
        """Record audio and save to WAV file"""
        print("\nRecording will start in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print("🎤 Recording... Speak now!")
        
        # audio recording
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16
        )
        sd.wait()
        print("✅ Recording complete!")
        
        # save as wav file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
    
    def extract_features(self, audio_data):
        """Extract simple audio features"""
        # normalizing the audio
        audio_norm = audio_data / np.max(np.abs(audio_data))
        
        # Calculate spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            audio_norm,
            fs=self.sample_rate,
            nperseg=1024,
            noverlap=512
        )
        
        # mean energy per frequency band
        feature_vector = np.mean(spectrogram, axis=1)
        return feature_vector
    
    def compare_features(self, features1, features2):
        """Compare two feature vectors"""
        # cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def enroll_user(self, username):
        """Enroll a new user with a paragraph"""
        print(f"\n📝 Enrolling user: {username}")
        print("Please read the following paragraph:")
        paragraph = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump! "
            "Sphinx of black quartz, judge my vow."
        )
        print(paragraph)
        
        filename = os.path.join(self.recordings_dir, f"{username}_enrollment.wav")
        self.record_audio(filename)
        
        # Extract features
        sample_rate, audio_data = wavfile.read(filename)
        enrollment_features = self.extract_features(audio_data)
        
        # Save features as enrollment template
        np.save(os.path.join(self.enrolled_dir, f"{username}_enrolled.npy"), enrollment_features)
        print(f"\n✅ Successfully enrolled {username}!")
    
    def verify_user(self, username):
        """Verify a user's identity using voice"""
        enrolled_file = os.path.join(self.enrolled_dir, f"{username}_enrolled.npy")
        
        if not os.path.exists(enrolled_file):
            print("❌ User not found! Please enroll first.")
            return False
        
        print("\n🔒 Voice Verification Started")
        print("Please speak any phrase for verification.")
        
        # recording the verification attempt
        verify_file = os.path.join(self.recordings_dir, f"{username}_verify_{int(time.time())}.wav")
        self.record_audio(verify_file)
        
        # loading and comparing features
        enrolled_features = np.load(enrolled_file)
        sample_rate, verify_audio = wavfile.read(verify_file)
        verify_features = self.extract_features(verify_audio)
        
        similarity = self.compare_features(enrolled_features, verify_features)
        threshold = 0.75  # adjustable threshold
        
        if similarity > threshold:
            print(f"\n✅ Voice verified! Similarity score: {similarity:.2f}")
            return True
        else:
            print(f"\n❌ Voice verification failed. Similarity score: {similarity:.2f}")
            return False

def main():
    auth_system = SimpleVoiceAuth()
    
    while True:
        print("\n=== Voice Authentication System ===")
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