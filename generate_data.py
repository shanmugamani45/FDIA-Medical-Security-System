import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

NUM_PATIENTS = 10
READINGS_PER_PATIENT = 500

def generate_patient_vitals(patient_id):
    timestamps = [datetime(2024, 1, 1) + timedelta(seconds=i*5)
                  for i in range(READINGS_PER_PATIENT)]
    hr_base = np.random.randint(65, 85)
    bp_sys_base = np.random.randint(110, 130)
    bp_dia_base = np.random.randint(70, 85)
    spo2_base = np.random.uniform(96, 99)
    temp_base = np.random.uniform(36.4, 37.2)
    rr_base = np.random.randint(14, 18)

    hr = hr_base + np.random.normal(0, 2, READINGS_PER_PATIENT)
    bp_sys = bp_sys_base + np.random.normal(0, 3, READINGS_PER_PATIENT)
    bp_dia = bp_dia_base + np.random.normal(0, 2, READINGS_PER_PATIENT)
    spo2 = np.clip(spo2_base + np.random.normal(0, 0.3, READINGS_PER_PATIENT), 94, 100)
    temp = temp_base + np.random.normal(0, 0.1, READINGS_PER_PATIENT)
    rr = rr_base + np.random.normal(0, 1, READINGS_PER_PATIENT)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'patient_id': f'P{patient_id:03d}',
        'heart_rate': np.round(hr, 1),
        'bp_systolic': np.round(bp_sys, 1),
        'bp_diastolic': np.round(bp_dia, 1),
        'spo2': np.round(spo2, 1),
        'temperature': np.round(temp, 1),
        'resp_rate': np.round(rr, 1),
        'label': 0,
        'attack_type': 'Normal'
    })
    return df

def inject_attacks(df):
    df = df.copy()
    n = len(df)
    vitals = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temperature']

    # Additive Bias Attack — 10% of data
    bias_idx = np.random.choice(n, int(n * 0.10), replace=False)
    for idx in bias_idx:
        col = np.random.choice(vitals)
        df.at[idx, col] += np.random.uniform(8, 15)
        df.at[idx, 'label'] = 1
        df.at[idx, 'attack_type'] = 'Additive Bias'

    # Scaling Attack — 8% of data
    scale_idx = np.random.choice(
        [i for i in range(n) if i not in bias_idx], int(n * 0.08), replace=False)
    for idx in scale_idx:
        col = np.random.choice(vitals)
        df.at[idx, col] *= np.random.uniform(1.15, 1.25)
        df.at[idx, 'label'] = 1
        df.at[idx, 'attack_type'] = 'Scaling'

    # Replay Attack — 7% of data
    used = set(bias_idx) | set(scale_idx)
    replay_idx = np.random.choice(
        [i for i in range(30, n) if i not in used], int(n * 0.07), replace=False)
    for idx in replay_idx:
        past_idx = idx - np.random.randint(20, 30)
        for col in vitals:
            df.at[idx, col] = df.at[past_idx, col]
        df.at[idx, 'label'] = 1
        df.at[idx, 'attack_type'] = 'Replay'

    return df

all_patients = []
for pid in range(1, NUM_PATIENTS + 1):
    patient_df = generate_patient_vitals(pid)
    patient_df = inject_attacks(patient_df)
    all_patients.append(patient_df)

dataset = pd.concat(all_patients, ignore_index=True)
dataset.to_csv('data/patient_vitals.csv', index=False)
print(f"Dataset created: {len(dataset)} records")
print(f"Normal: {(dataset['label']==0).sum()} | Attacks: {(dataset['label']==1).sum()}")
print(f"Attack types:\n{dataset[dataset['label']==1]['attack_type'].value_counts()}")
