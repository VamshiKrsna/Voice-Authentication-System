import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
import numpy as np

class ModelManager:
    def __init__(self, model_path="voice_models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.gmm = None
        self.threshold = 0.8

    def train_model(self, user_id, features):
        """Train GMM model for a user"""
        self.gmm = GaussianMixture(n_components=16, covariance_type='diag')
        self.gmm.fit(features)
        
        model_file = self.model_path / f"{user_id}_voice_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.gmm, f)
        
        return True

    def load_user_model(self, user_id):
        """Load a user's voice model"""
        model_file = self.model_path / f"{user_id}_voice_model.pkl"
        try:
            with open(model_file, 'rb') as f:
                self.gmm = pickle.load(f)
            return True
        except FileNotFoundError:
            print(f"No voice model found for user {user_id}")
            return False

    def verify_features(self, features):
        """Verify features against loaded model"""
        if self.gmm is None:
            return False
            
        score = np.mean(self.gmm.score_samples(features))
        normalized_score = 1 / (1 + np.exp(-score))
        
        print(f"Voice verification score: {normalized_score:.3f}")
        return normalized_score > self.threshold