import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('creditcard_2023.csv')

# Drop ID column if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Separate features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
y_pred = iso_forest.fit_predict(X_scaled)

# Convert predictions: -1 = anomaly → 1 (fraud), 1 = normal → 0
df['Anomaly'] = np.where(y_pred == -1, 1, 0)

# Show confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:\n", confusion_matrix(y, df['Anomaly']))
print("\nClassification Report:\n", classification_report(y, df['Anomaly']))

# Visualize anomalies
sns.countplot(data=df, x='Anomaly')
plt.title('Detected Anomalies')
plt.show()
