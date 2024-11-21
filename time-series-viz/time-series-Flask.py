from flask import Flask, render_template, request, jsonify
import h5py
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

app = Flask(__name__)

# Load the HDF5 file once on startup for efficiency
proj_dir = r"D:\data\TAMU\hallway-tests_analysis"  # Update to your actual directory
recording_name = "2024-07-31_15-19-11_7917-1-sec-dwell-training-sci"
features_h5 = os.path.join(proj_dir, recording_name, "features.h5")

with h5py.File(features_h5, "r") as f:
    recording_id = list(f.keys())[0]
    feature_cols = [feature for feature in f[recording_id].keys() if f[recording_id][feature].shape != ()]
    features_data = {feature: f[recording_id][feature][:] for feature in feature_cols}
    recording_fps = f[recording_id]["fps"][()]
    recording_frame_count = f[recording_id]["frame_count"][()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_features', methods=['GET'])
def get_features():
    return jsonify(feature_cols)

@app.route('/get_data', methods=['POST'])
def get_data():
    features = request.json['features']
    sigma = request.json.get('sigma', 1) * recording_fps
    data = {}

    for feature in features:
        raw_data = features_data[feature]
        smoothed_data = gaussian_filter1d(raw_data, sigma=sigma)
        data[feature] = {
            'raw': raw_data.tolist(),
            'smoothed': smoothed_data.tolist()
        }

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
