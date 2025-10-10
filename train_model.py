import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load features & labels saved earlier
X = np.load('features.npy')
y = np.load('labels.npy')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Accuracy check
acc = clf.score(X_test, y_test)
print(f"Model accuracy: {acc:.2f}")

# Save model for deployment
joblib.dump(clf, 'urban_noise_model.pkl')
