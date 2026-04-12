====================================================
FDIA MEDICAL SECURITY SYSTEM — RUN INSTRUCTIONS
====================================================

STEP 1 — Install dependencies:
    pip install -r requirements.txt

STEP 2 — Generate dataset:
    python generate_data.py

STEP 3 — Train models:
    python train_model.py

STEP 4 — Launch dashboard:
    python app.py

STEP 5 — Open browser:
    http://localhost:5000

====================================================
FEATURES:
- 10 patients monitored simultaneously
- 3 attack types: Additive Bias, Scaling, Replay
- Severity levels: Low / Medium / High
- Explainability: WHY each attack was flagged
- PDF report generation per patient
- Live attack timeline chart
- Attack type distribution chart
====================================================
