# 🛡️FDIA Medical Security System
## 🚀 Overview

The FDIA Medical Security System is a cybersecurity-driven application designed to detect False Data Injection Attacks (FDIA) in medical systems. It combines machine learning with explainable AI to identify anomalies and ensure the integrity of healthcare data.

## 🌐 Live Demo
🔗 Application: https://fdia-medical-security-system.onrender.com/login
---
🎯 Problem Statement

Modern healthcare systems depend on accurate data. FDIA attacks can:

Corrupt patient records
Lead to incorrect diagnoses
Compromise critical decision-making systems

This project focuses on detecting and analyzing such attacks in real time.

✨ Key Features
🔍 FDIA Detection Engine
📊 Real-Time Attack Simulation
🧠 Explainable AI using SHAP
🚨 Alert & Notification System
📜 Attack History Logging
🎨 Interactive Web Interface
🧠 Machine Learning Models

This system uses a dual-model approach to improve detection accuracy:

Model 1: Random Forest Classifier
Handles structured medical data effectively
Robust against overfitting
Provides high accuracy for anomaly detection
Model 2: Logistic Regression
Lightweight and fast
Helps in baseline comparison
Improves interpretability of predictions

👉 Predictions from both models are used to:

Cross-validate suspicious activity
Improve reliability of FDIA detection
🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript
Backend: Python (Flask)
Machine Learning: Scikit-learn
Explainability: SHAP
Deployment: Render
⚙️ System Architecture
User inputs / simulated medical data
Data preprocessing
Dual-model prediction (Random Forest + Logistic Regression)
SHAP-based explanation
Alert generation & logging
📦 Installation & Setup
# Clone the repository
git clone https://github.com/shanmugamani45/FDIA-Medical-Security-System.git

cd FDIA-Medical-Security-System

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
▶️ Usage
Login to the system
Input or simulate medical data
View detection results
Analyze SHAP explanations
Monitor alerts and attack logs
📊 Core Capabilities
Detects anomalies in healthcare datasets
Provides interpretable AI decisions
Simulates real-world cyberattack scenarios
Enhances trust in medical systems
📸 Screenshots

(Add these — don’t skip if you want impact)

Login Page
Dashboard
Detection Results
SHAP Output
🔮 Future Enhancements
IoT-based real-time medical device integration
Deep learning models (LSTM / Autoencoders)
Real-time streaming analytics
Role-based access control
API integration with hospital systems
🤝 Contributing

Fork → Modify → Pull Request

📄 License

MIT License
