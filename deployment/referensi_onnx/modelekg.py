import os
import logging
from datetime import datetime

import numpy as np
import onnxruntime as ort
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1) Load Environment Variables (lebih robust)
# -----------------------------------------------------------------------------
# Pastikan load .env dari lokasi file ini (bukan tergantung cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# -----------------------------------------------------------------------------
# 2) Logger
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -----------------------------------------------------------------------------
# 3) Konfigurasi Gemini
# -----------------------------------------------------------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")

gemini_model = None
if GEMINI_KEY and GEMINI_KEY.strip() and "REPLACE YOUR TOKEN" not in GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel(MODEL_NAME)
        logger.info("✓ Gemini AI initialized (model=%s)", MODEL_NAME)
    except Exception as e:
        logger.error("✗ Failed to init Gemini: %s", e)
        gemini_model = None
else:
    logger.warning("✗ GEMINI_API_KEY not found / placeholder. AI advice will be disabled.")

# -----------------------------------------------------------------------------
# 4) Load ONNX Model (robust, server tetap jalan)
# -----------------------------------------------------------------------------
session = None
input_name_signal = None
input_name_rr = None
output_name = None

ONNX_PATH = os.path.join(BASE_DIR, "ecgmodel.onnx")
try:
    session = ort.InferenceSession(ONNX_PATH)
    input_name_signal = session.get_inputs()[0].name
    input_name_rr = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    logger.info("✓ ONNX model loaded (%s)", ONNX_PATH)
except Exception as e:
    logger.error("✗ Failed to load ONNX (%s): %s", ONNX_PATH, e)
    session = None

CLASS_NAMES = ["N", "L", "R", "A", "V", "F", "/", "Other"]
CLASS_DESCRIPTIONS = {
    "N": "Normal Sinus Rhythm",
    "L": "Left Bundle Branch Block",
    "R": "Right Bundle Branch Block",
    "A": "Atrial Premature Beat",
    "V": "Ventricular Premature Beat",
    "F": "Fusion beat",
    "/": "Paced rhythm",
    "Other": "Other abnormality"
}

def preprocess_signal(data):
    sig = np.array(data, dtype=np.float32)
    mu, sd = float(np.mean(sig)), float(np.std(sig))
    if sd == 0:
        sd = 1e-8
    sig = (sig - mu) / sd

    # shape expected: (216, 2) setelah concat
    sig = sig.reshape(216, 1)
    sig = np.concatenate([sig, sig], axis=1)
    return np.expand_dims(sig, axis=0)

def make_gemini_prompt(result_label: str, description: str, confidence: float, patient_context: str = "") -> str:
    # prompt singkat & aman untuk UI
    ctx = (patient_context or "").strip()
    if ctx:
        ctx = f" Patient context: {ctx}"
    return (
        f"Act as a cardiologist. Patient ECG classification: {result_label} ({description}). "
        f"Confidence: {confidence*100:.1f}%.{ctx} "
        "Give a SINGLE SENTENCE (max 150 chars) recommendation. "
        "Do not say 'I am an AI'. Be direct."
    )

def generate_ai_advice(result_label: str, confidence: float, description: str, patient_context: str = "") -> str:
    if gemini_model is None:
        return ""

    prompt = make_gemini_prompt(result_label, description, confidence, patient_context)
    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", "")
        return (text or "").strip()
    except Exception as e:
        logger.error("Gemini API Error: %s", e)
        return ""  # biarkan kosong, Java fallback

# -----------------------------------------------------------------------------
# Health endpoint (debug koneksi dari Java)
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "modelekg-backend",
        "onnx_loaded": session is not None,
        "gemini_enabled": gemini_model is not None,
        "time": datetime.utcnow().isoformat() + "Z"
    })

# -----------------------------------------------------------------------------
# Endpoint 1: /predict  (ONNX + optional Gemini)
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if session is None:
            return jsonify({
                "ok": False,
                "error": "ONNX model not loaded. Check ecgmodel.onnx path/logs."
            }), 500

        req_data = request.get_json(silent=True) or {}
        if "signal" not in req_data:
            return jsonify({"ok": False, "error": "Missing 'signal' data"}), 400
        if "rr" not in req_data:
            return jsonify({"ok": False, "error": "Missing 'rr' data"}), 400

        # --- A. Prediksi ONNX ---
        processed_signal = preprocess_signal(req_data["signal"])
        processed_rr = np.array(req_data["rr"], dtype=np.float32).reshape(1, -1)

        outputs = session.run([output_name], {
            input_name_signal: processed_signal,
            input_name_rr: processed_rr
        })

        prediction = outputs[0][0]
        class_idx = int(np.argmax(prediction))
        result_label = CLASS_NAMES[class_idx]
        confidence = float(prediction[class_idx])
        description = CLASS_DESCRIPTIONS.get(result_label, "Unknown")

        # --- B. Generate Saran AI (Gemini) ---
        patient_context = req_data.get("patientContext", "")
        ai_advice = generate_ai_advice(result_label, confidence, description, patient_context)

        return jsonify({
            "ok": True,
            "classification": result_label,
            "confidence": confidence,
            "class_label": description,
            "advice": ai_advice,     # kosong jika Gemini error/disabled
            "mode": "hybrid_cloud"
        })

    except Exception as e:
        logger.error("Prediction Error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# -----------------------------------------------------------------------------
# Endpoint 2: /gemini/advice (Gemini-only, cocok untuk Java fallback strategy)
# Java bisa panggil ini tanpa kirim signal
# -----------------------------------------------------------------------------
@app.route("/gemini/advice", methods=["POST"])
def gemini_advice():
    logger.info("HIT /gemini/advice with payload: %s", request.get_json(silent=True))
    try:
        req_data = request.get_json(silent=True) or {}
        classification = req_data.get("classification", "")
        confidence = float(req_data.get("confidence", 0.0))
        patient_context = req_data.get("patientContext", "")

        description = CLASS_DESCRIPTIONS.get(classification, "Unknown")
        advice = generate_ai_advice(classification, confidence, description, patient_context)

        # kalau advice kosong, tetap 200 tapi ok=false biar Java tau fallback
        if not advice:
            return jsonify({
                "ok": False,
                "error": "Gemini advice unavailable (disabled or error).",
                "advice": ""
            }), 200

        return jsonify({
            "ok": True,
            "advice": advice,
            "model": MODEL_NAME
        }), 200

    except Exception as e:
        logger.error("Gemini Advice Error: %s", e)
        return jsonify({"ok": False, "error": str(e), "advice": ""}), 500

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("REGISTERED ROUTES:")
    for r in app.url_map.iter_rules():
        print(r, r.methods)

    app.run(host="127.0.0.1", port=5000, debug=True)