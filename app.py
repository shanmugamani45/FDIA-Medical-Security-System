from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import json
import random
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import threading
import time
import shap

app = Flask(__name__)
app.secret_key = 'fdia_medical_secret_key'

USERS = {
    'admin': {'password': 'ad123', 'role': 'admin'},
    'doctor': {'password': 'doc123', 'role': 'doctor'},
    'hacker': {'password': '1or1=1', 'role': 'hacker'}
}

# ── Load Models ───────────────────────────────────────────────
scaler     = pickle.load(open('models/scaler.pkl', 'rb'))
rf_model   = pickle.load(open('models/random_forest.pkl', 'rb'))
type_clf   = pickle.load(open('models/attack_type_clf.pkl', 'rb'))
type_map   = pickle.load(open('models/type_map.pkl', 'rb'))
inv_type   = {v: k for k, v in type_map.items()}

explainer = shap.TreeExplainer(rf_model)

df_all = pd.read_csv('data/patient_vitals.csv')
FEATURES   = ['heart_rate','bp_systolic','bp_diastolic','spo2','temperature','resp_rate']
PATIENTS   = df_all['patient_id'].unique().tolist()

# ── Alert Log (in memory) ─────────────────────────────────────
alert_log = []
history_log = []
patient_cursors = {pid: 0 for pid in PATIENTS}
live_patient_data = {pid: None for pid in PATIENTS}
injected_overrides = {pid: {} for pid in PATIENTS}

def simulate_live_data():
    """Background thread simulating data every 2 seconds"""
    while True:
        for pid in PATIENTS:
            pdata = df_all[df_all['patient_id'] == pid]
            idx = patient_cursors[pid] % len(pdata)
            row = pdata.iloc[idx]
            reading_raw = [row[f] for f in FEATURES]
            
            if pid in injected_overrides:
                for i, f in enumerate(FEATURES):
                    if f in injected_overrides[pid]:
                        reading_raw[i] = injected_overrides[pid][f]
                        
            live_patient_data[pid] = reading_raw
            patient_cursors[pid] += 1
        time.sleep(2)

threading.Thread(target=simulate_live_data, daemon=True).start()

def get_severity(prob):
    if prob < 0.5:   return "Normal",   "success", 0
    elif prob < 0.70: return "Low",     "warning", 1
    elif prob < 0.85: return "Medium",  "orange",  2
    else:             return "High",    "danger",  3

def analyze_reading(reading_raw, patient_id):
    reading_scaled = scaler.transform([reading_raw])[0]
    prob = rf_model.predict_proba([reading_scaled])[0][1]
    is_attack = prob >= 0.5
    severity_label, severity_class, severity_level = get_severity(prob)

    attack_type = "Normal"
    explanation = "All vitals within expected range."
    shap_dict = {}
    if is_attack:
        type_pred = type_clf.predict([reading_scaled])[0]
        attack_type = inv_type.get(type_pred, "Unknown")
        
        try:
            shap_vals = explainer.shap_values(np.array([reading_scaled]))
            if isinstance(shap_vals, list):
                attack_shap = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                attack_shap = shap_vals[0, :, 1]
            else:
                attack_shap = shap_vals[0]
                
            top_feature_idx = np.argmax(np.abs(attack_shap))
            top_feature = FEATURES[top_feature_idx].replace('_', ' ').title()
            effect_val = attack_shap[top_feature_idx]
            explanation = (f"{attack_type} attack detected. "
                           f"Top anomaly indicator: {top_feature} (SHAP effect: {effect_val:.2f})")
            
            for i, f in enumerate(FEATURES):
                shap_dict[f.replace('_', ' ').title()] = round(float(attack_shap[i]), 4)
                
        except Exception as e:
            explanation = f"{attack_type} attack detected. (Explainability failed: {e})"

    return {
        'patient_id': patient_id,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'heart_rate': round(reading_raw[0], 1),
        'bp_systolic': round(reading_raw[1], 1),
        'bp_diastolic': round(reading_raw[2], 1),
        'spo2': round(reading_raw[3], 1),
        'temperature': round(reading_raw[4], 1),
        'resp_rate': round(reading_raw[5], 1),
        'attack_prob': round(prob * 100, 1),
        'is_attack': bool(is_attack),
        'attack_type': attack_type,
        'severity': severity_label,
        'severity_class': severity_class,
        'severity_level': severity_level,
        'explanation': explanation,
        'shap_values': shap_dict
    }

# ── Routes ────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username]['password'] == password:
            session['user'] = username
            session['role'] = USERS[username]['role']
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', patients=PATIENTS, role=session['role'])

@app.route('/api/stream')
def stream():
    results = []
    for pid in PATIENTS:
        if live_patient_data[pid] is not None:
            reading_raw = live_patient_data[pid]
            result = analyze_reading(reading_raw, pid)
            results.append(result)
            
            history_record = {
                'timestamp': result['timestamp'],
                'patient_id': result['patient_id'],
                'attack_prob': result['attack_prob'],
                'severity': result['severity']
            }
            history_log.insert(0, history_record)
            if len(history_log) > 500:
                history_log.pop()
                
            if result['is_attack']:
                alert_log.insert(0, result)
                if len(alert_log) > 50:
                    alert_log.pop()
    return jsonify(results)

@app.route('/api/history')
def get_history():
    return jsonify(history_log[:50])

@app.route('/api/alerts')
def get_alerts():
    return jsonify(alert_log[:20])

@app.route('/api/inject', methods=['POST'])
def inject_attack():
    data = request.json
    pid = data.get('patient_id')
    feature = data.get('feature')
    val = float(data.get('value', 0))
    duration = int(data.get('duration', 10))
    
    if pid and feature:
        if pid not in injected_overrides:
            injected_overrides[pid] = {}
        injected_overrides[pid][feature] = val
        
        def clear_override():
            time.sleep(duration)
            if feature in injected_overrides.get(pid, {}):
                del injected_overrides[pid][feature]
                
        threading.Thread(target=clear_override).start()
        return jsonify({"status": "injected", "patient": pid, "feature": feature, "value": val})
    return jsonify({"error": "Invalid request"}), 400

@app.route('/api/report/<patient_id>')
def generate_report(patient_id):
    patient_alerts = [a for a in alert_log if a['patient_id'] == patient_id]
    if not patient_alerts:
        patient_alerts = [{
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'attack_type': 'Additive Bias', 'severity': 'High',
            'heart_rate': 92, 'bp_systolic': 145, 'spo2': 94,
            'temperature': 37.8, 'explanation': 'Sample alert for report demo.'
        }]

    os.makedirs('reports', exist_ok=True)
    path = f'reports/report_{patient_id}.pdf'
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("FDIA Detection System — Patient Alert Report", styles['Title']))
    elems.append(Paragraph(f"Patient ID: {patient_id}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elems.append(Spacer(1, 20))
    elems.append(Paragraph(f"Total Alerts: {len(patient_alerts)}", styles['Heading2']))
    elems.append(Spacer(1, 10))

    table_data = [['Time', 'Attack Type', 'Severity', 'HR', 'BP Sys', 'SpO2', 'Temp', 'Explanation']]
    for a in patient_alerts[:15]:
        table_data.append([
            a['timestamp'], a['attack_type'], a['severity'],
            str(a.get('heart_rate','-')), str(a.get('bp_systolic','-')),
            str(a.get('spo2','-')), str(a.get('temperature','-')),
            a['explanation'][:40]+'...' if len(a.get('explanation',''))>40 else a.get('explanation','-')
        ])

    t = Table(table_data, colWidths=[55,75,55,35,45,40,40,130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F3864')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTSIZE',   (0,0), (-1,-1), 7),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#EBF3FB')]),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 20))
    elems.append(Paragraph("Disclaimer: This report is generated by an ML-based proof-of-concept system. Clinical decisions must be verified by qualified medical personnel.", styles['Italic']))
    doc.build(elems)
    return jsonify({'status': 'ok', 'path': path})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
