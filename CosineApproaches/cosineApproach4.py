# PERFECT.

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
            
        print("\n🎤 Recording... Speak now!")
        
        recording = sd.rec(
            int(5 * self.sample_rate),  # 5 seconds per line
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
    
    def merge_audio_files(self, file_list, output_filename):
        """Merge multiple WAV files into one"""
        combined_audio = []
        for file in file_list:
            sample_rate, audio_data = wavfile.read(file)
            combined_audio.append(audio_data)
        
        merged_audio = np.concatenate(combined_audio, axis=0)
        
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(merged_audio.tobytes())
        
    def extract_features(self, audio_data):
        """Extract simple audio features"""
        audio_norm = audio_data / np.max(np.abs(audio_data))
        frequencies, times, spectrogram = signal.spectrogram(
            audio_norm,
            fs=self.sample_rate,
            nperseg=1024,
            noverlap=512
        )
        return np.mean(spectrogram, axis=1)
    
    def compare_features(self, features1, features2):
        """Compare two feature vectors"""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        return dot_product / (norm1 * norm2)
    
    def enroll_user(self, username):
        """Enroll a new user line by line"""
        print(f"\n📝 Enrolling user: {username}")
        
        paragraph_lines = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "Sphinx of black quartz, judge my vow."
        ]
        
        temp_files = []
        for i, line in enumerate(paragraph_lines):
            print(f"\n📜 Read this line: {line}")
            filename = os.path.join(self.recordings_dir, f"{username}_line{i+1}.wav")
            self.record_audio(filename)
            temp_files.append(filename)
        
        # Merge all recorded lines into one file
        merged_filename = os.path.join(self.recordings_dir, f"{username}_enrollment.wav")
        self.merge_audio_files(temp_files, merged_filename)
        
        # Extract and save features
        sample_rate, audio_data = wavfile.read(merged_filename)
        enrollment_features = self.extract_features(audio_data)
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
        
        verify_file = os.path.join(self.recordings_dir, f"{username}_verify_{int(time.time())}.wav")
        self.record_audio(verify_file)
        
        enrolled_features = np.load(enrolled_file)
        sample_rate, verify_audio = wavfile.read(verify_file)
        verify_features = self.extract_features(verify_audio)
        
        similarity = self.compare_features(enrolled_features, verify_features)
        threshold = 0.75
        
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
