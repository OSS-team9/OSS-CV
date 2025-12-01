import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_scaler(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean_"], dtype=np.float32)
    scaler.scale_ = np.array(data["scale_"], dtype=np.float32)
    scaler.var_ = np.array(data["var_"], dtype=np.float32)
    scaler.n_features_in_ = int(data["n_features_in_"])
    scaler.n_samples_seen_ = int(data.get("n_samples_seen_", 0))
    
    return scaler
