from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import types

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
CORS(app)

# ============================================================
# 🔧 FIX per l'errore "No module named 'numpy._core'" (Railway)
# ============================================================

try:
    import numpy.core as npcore
    module = types.ModuleType("numpy._core")
    module.__dict__.update(npcore.__dict__)
    sys.modules["numpy._core"] = module
except Exception as e:
    print(f"⚠️ Impossibile creare alias numpy._core → numpy.core: {e}")


# ============================================================
# 🔥 CARICAMENTO MODELLO PT_AI + PREPROCESSING
# ============================================================

model = None
numeric_columns = None
categorical_columns = None
scaler = None
label_encoder = None

try:
    print("📥 Caricamento modello PT_AI in corso...")
    model = load_model("pt_ai_nn_model.keras")

    with open("pt_ai_preprocessing_nn.pkl", "rb") as f:
        prep = pickle.load(f)

    # Assumo che nel pickle ci sia un dizionario con queste chiavi:
    # "numeric_columns", "categorical_columns", "scaler", "label_encoder"
    numeric_columns = prep["numeric_columns"]
    categorical_columns = prep["categorical_columns"]
    scaler = prep["scaler"]
    label_encoder = prep["label_encoder"]

    print("✅ Modello e preprocessing caricati correttamente!")

except Exception as e:
    print(f"❌ Errore durante il caricamento del modello o del preprocessing: {e}")
    model = None


# ============================================================
# 🔍 FUNZIONE DI PREDIZIONE DELL’ESERCIZIO (PT_AI)
# ============================================================

def predict_exercise(example_dict: dict) -> str:
    """
    example_dict deve contenere gli stessi campi del CSV originale
    (tutti gli angoli + eventuali feature categoriali come 'Side').

    Esempio JSON dal client:
    {
        "Angle1": 30.5,
        "Angle2": 45.2,
        ...
        "Side": "Left"
    }
    """

    if model is None:
        raise RuntimeError("Modello non caricato.")

    # 1. DataFrame con una sola riga
    df_example = pd.DataFrame([example_dict])

    # 2. Estrazione delle colonne numeriche
    df_num = df_example[numeric_columns]

    # 3. Estrazione + one-hot delle categoriche
    df_cat = pd.get_dummies(df_example.drop(columns=numeric_columns))

    # 4. Allineamento colonne categoriche a quelle di training
    for col in categorical_columns:
        if col not in df_cat.columns:
            df_cat[col] = 0

    df_cat = df_cat[categorical_columns]

    # 5. Concatenazione numeriche + categoriche
    X = np.hstack([df_num.values, df_cat.values])

    # 6. Scaling
    X_scaled = scaler.transform(X)

    # 7. Predizione
    preds = model.predict(X_scaled)
    idx = np.argmax(preds, axis=1)[0]

    # 8. Indice → etichetta esercizio
    predicted_label = label_encoder.inverse_transform([idx])[0]

    return predicted_label


# ============================================================
# 🌐 ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'PT_AI API is running! 🏋️',
        'version': '1.0.0'
    })


@app.route('/api/analyze-exercise', methods=['POST'])
def analyze_exercise():
    """
    Esempio di body JSON atteso:

    {
        "Angle1": 30,
        "Angle2": 45,
        ...,
        "Side": "Left"
    }

    Risposta:

    {
        "success": true,
        "predicted_exercise": "Squats"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'Nessun dato ricevuto'}), 400

        if model is None:
            return jsonify({'success': False, 'error': 'Modello non caricato sul server'}), 500

        predicted_label = predict_exercise(data)

        return jsonify({
            "success": True,
            "predicted_exercise": predicted_label
        }), 200

    except Exception as e:
        print("❌ Errore durante analisi:", e)
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# AVVIO SERVER (solo in locale)
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 PT_AI API Server Starting...")
    print("=" * 50)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
