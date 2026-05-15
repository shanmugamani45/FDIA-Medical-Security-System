# FDIA Medical Security System

## Overview

The FDIA (False Data Injection Attack) Medical Security System is a machine learning-based solution designed to detect and classify malicious data injections in medical sensor data.

The system simulates real-world healthcare monitoring scenarios and identifies anomalies that may indicate cyber attacks on patient data.

---

## Problem Statement

Modern healthcare systems rely on continuous sensor data such as heart rate, temperature, and oxygen levels. These systems are vulnerable to False Data Injection Attacks (FDIA), where attackers manipulate sensor data, leading to incorrect medical decisions.

---

## Solution

This project implements an intelligent detection system that:

* Detects abnormal patterns in medical sensor data
* Classifies different types of attacks
* Simulates real-time monitoring
* Provides insights into feature importance

---

## Features

* Machine Learning-based attack detection
* Multi-class attack classification (Additive, Scaling, Replay)
* Synthetic dataset simulation for healthcare data
* Real-time data stream simulation
* Feature importance analysis (Explainability)
* Attack history logging
* Web-based interface for interaction

---

## Tech Stack

* Python
* Flask (Backend)
* Scikit-learn
* XGBoost
* Pandas / NumPy
* Matplotlib (Visualization)

---

## Machine Learning Models Used

* Random Forest Classifier
* XGBoost Classifier
* SMOTE for handling imbalanced data
* MinMaxScaler for normalization

---

## System Architecture

Data Source (Synthetic Medical Data)
↓
Data Preprocessing (Scaling, Cleaning)
↓
ML Model (RF / XGBoost)
↓
Attack Detection
↓
Attack Classification
↓
Output + Logging + Visualization

---

## Real-Time Simulation

The system simulates continuous monitoring by processing data in a sequential manner, mimicking real-time healthcare sensor streams.

---

## Explainability

The model provides feature importance to identify which parameters contribute most to attack detection.

Example:

* Heart Rate deviation → High impact
* SpO2 variation → Medium impact
* Temperature fluctuation → Low impact

---

## Dataset

Due to the lack of real-world IoT medical datasets, synthetic data is generated to simulate realistic healthcare sensor behavior and attack scenarios.

---

## Future Enhancements

* Integration with real IoT medical devices
* Deployment as a cloud-based monitoring system
* Advanced anomaly detection using deep learning
* Real-time alert system for hospitals

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/fdia-medical-security-system.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

---

## Live Demo

https://fdia-medical-security-system.onrender.com/

---

## Screen Shot
### Login page
<img width="1918" height="969" alt="image" src="https://github.com/user-attachments/assets/4132621f-79ef-4d35-be74-b1e30b0a20ab" />

### Admin view
<img width="1914" height="967" alt="image" src="https://github.com/user-attachments/assets/e042bc66-6abd-42ef-8486-a327733add32" />

<img width="1919" height="961" alt="image" src="https://github.com/user-attachments/assets/5c5b2b0b-d056-4b15-95b5-33cd2f530c18" />

### Doctor view
<img width="1918" height="969" alt="image" src="https://github.com/user-attachments/assets/64a00a2e-1e87-4a18-b940-e1a1484f0b6d" />

### hacker view
<img width="1916" height="964" alt="image" src="https://github.com/user-attachments/assets/d677c5b8-1a7b-486b-8602-9488a8c3db8e" />

## Conclusion

This project demonstrates how machine learning can be applied to detect cybersecurity threats in healthcare systems. It provides a scalable foundation for building secure and intelligent medical monitoring systems.

---
Created with ❤️ by [shanmugamani45](https://github.com/shanmugamani45)
---
