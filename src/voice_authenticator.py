from .audio_utils import AudioRecorder
from .feature_extractor import FeatureExtractor
from .model_manager import ModelManager
from .transcriber import Transcriber
import os

class VoiceAuthenticator:
    def __init__(self, model_path="voice_models", sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording_duration = 5
        
        # Initialize components
        self.audio_recorder = AudioRecorder(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.model_manager = ModelManager(model_path)
        self.transcriber = Transcriber()

    def train_model(self, user_id, num_samples=3):
        """Train GMM model for a user"""
        features_list = []
        
        print(f"Starting enrollment for user {user_id}")
        print("We'll record multiple samples of your voice.")
        
        for i in range(num_samples):
            input(f"Press Enter to record sample {i+1}/{num_samples}...")
            audio = self.audio_recorder.record_audio(self.recording_duration)
            features = self.feature_extractor.extract_features(audio)
            
            if features is not None:
                features_list.append(features)
            else:
                print(f"Failed to extract features from sample {i+1}")
                return False
        
        # Combine all features and train
        all_features = np.vstack(features_list)
        return self.model_manager.train_model(user_id, all_features)

    def verify_voice(self, user_id, audio=None):
        """Verify a voice against a user's model"""
        if not self.model_manager.load_user_model(user_id):
            return False
            
        if audio is None:
            audio = self.audio_recorder.record_audio(3)
            
        features = self.feature_extractor.extract_features(audio)
        if features is None:
            return False
            
        return self.model_manager.verify_features(features)

    def transcribe_with_auth(self, user_id):
        """Continuously transcribe audio with voice authentication"""
        if not self.model_manager.load_user_model(user_id):
            print("Please enroll first.")
            return
            
        print("Starting authenticated transcription session...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Record audio chunk
                audio = self.audio_recorder.record_audio(5)
                
                # Verify voice
                if not self.verify_voice(user_id, audio):
                    print("⚠️ Unauthorized voice detected!")
                    continue
                
                # Save audio temporarily for whisper
                temp_file = "temp_audio.wav"
                self.audio_recorder.save_audio(audio, temp_file)
                
                # Transcribe
                result = self.transcriber.transcribe_audio(temp_file)
                if result:
                    print(result)
                
                # Clean up
                os.remove(temp_file)
                
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        except Exception as e:
            print(f"Error during transcription: {e}")
