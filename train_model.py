import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FDIA DETECTION SYSTEM — MODEL TRAINING")
print("="*60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('data/patient_vitals.csv')
FEATURES = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temperature', 'resp_rate']
print(f"\n[1] Dataset loaded: {len(df)} records, {df['label'].sum()} attacks")

# ── 2. Normalize ──────────────────────────────────────────────
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
print("[2] Features normalized and scaler saved")

# ── 3. Train/Test Split ───────────────────────────────────────
X = df[FEATURES].values
y = df['label'].values
attack_types = df['attack_type'].values

X_train, X_test, y_train, y_test, at_train, at_test = train_test_split(
    X, y, attack_types, test_size=0.2, random_state=42, stratify=y)

# ── 4. SMOTE ──────────────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print(f"[3] SMOTE applied: {len(X_train_bal)} training samples")

# ── 5. Baseline Models ────────────────────────────────────────
print("\n[4] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]
print(f"    RF AUC-ROC: {roc_auc_score(y_test, rf_proba):.4f}")
pickle.dump(rf, open('models/random_forest.pkl', 'wb'))

print("[5] Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train_bal, y_train_bal)
xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:,1]
print(f"    XGB AUC-ROC: {roc_auc_score(y_test, xgb_proba):.4f}")
pickle.dump(xgb, open('models/xgboost.pkl', 'wb'))

# ── 6. Attack Type Classifier ─────────────────────────────────
print("[6] Training Attack Type Classifier...")
orig_attack_mask = y_train == 1
X_attack = X_train[orig_attack_mask]
at_attack = at_train[orig_attack_mask]
type_map = {'Additive Bias': 0, 'Scaling': 1, 'Replay': 2}
at_encoded = np.array([type_map[t] for t in at_attack])
type_clf = XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss')
type_clf.fit(X_attack, at_encoded)
pickle.dump(type_clf, open('models/attack_type_clf.pkl', 'wb'))
pickle.dump(type_map, open('models/type_map.pkl', 'wb'))
print("    Attack type classifier saved")

# ── 7. Results Summary ────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print("\nRandom Forest:")
print(classification_report(y_test, rf_pred, target_names=['Normal','Attack']))
print("XGBoost:")
print(classification_report(y_test, xgb_pred, target_names=['Normal','Attack']))
print(f"\nAUC-ROC Comparison:")
print(f"  Random Forest : {roc_auc_score(y_test, rf_proba):.4f}")
print(f"  XGBoost       : {roc_auc_score(y_test, xgb_proba):.4f}")
print("\n✅ All models trained and saved successfully!")
