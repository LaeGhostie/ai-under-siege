import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("🧠 AI Under Siege: Adversarial Attacks on ML")
st.markdown("""
Interact with a basic machine learning malware classifier and try to evade detection.
Modify the feature inputs to simulate adversarial behavior.
""")

# Generate synthetic data (you can replace this with real malware dataset later)
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sidebar for input manipulation
st.sidebar.header("✏️ Feature Tweaks")

feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
input_features = []

for i, name in enumerate(feature_names):
    val = st.sidebar.slider(name, float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
    input_features.append(val)

input_array = np.array(input_features).reshape(1, -1)

# Prediction
prediction = model.predict(input_array)[0]
confidence = model.predict_proba(input_array)[0][prediction] * 100

# Display results
label = "🛡️ Benign" if prediction == 0 else "☠️ Malicious"
st.metric(label="Prediction", value=label, delta=f"Confidence: {confidence:.2f}%")

# Accuracy of model (just for transparency)
acc = accuracy_score(y_test, model.predict(X_test))
st.caption(f"Model Accuracy: {acc*100:.2f}% (on synthetic data)")

# Show a chart of feature importance
importances = model.feature_importances_
st.bar_chart({"Feature Importance": importances})
