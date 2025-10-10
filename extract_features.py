import os
import pandas as pd
import librosa
import numpy as np

# Paths
DATASET_PATH = 'UrbanSound8K'  # change if needed
METADATA_FILENAME = os.path.join(DATASET_PATH, 'metadata', 'UrbanSound8K.csv')

# Load metadata
metadata = pd.read_csv(METADATA_FILENAME)

features, labels = [], []

for index, row in metadata.iterrows():
    fold = row["fold"]
    file_name = row["slice_file_name"]
    class_id = row["classID"]
    file_path = os.path.join(DATASET_PATH, 'audio', f'fold{fold}', file_name)

    # Load and extract MFCC features
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        features.append(mfccs)
        labels.append(class_id)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

print(f"Extracted features from {len(features)} audio files.")

# Optionally save features for later use
np.save('features.npy', X)
np.save('labels.npy', y)
