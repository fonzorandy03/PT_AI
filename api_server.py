from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
CORS(app)


# ============================================================
# 🔥 CARICAMENTO DEL MODELLO PT_AI (Keras) + PREPROCESSING
# ============================================================

try:
    print("📥 Caricamento modello PT_AI in corso...")
    pt_model = load_model("pt_ai_nn_model.keras")

    with open("pt_ai_preprocessing_nn.pkl", "rb") as f:
        prep = pickle.load(f)

    scaler = prep["scaler"]
    label_encoder = prep["label_encoder"]
    numeric_columns = prep["numeric_columns"]
    categorical_columns = prep["categorical_columns"]

    print("✅ Modello e preprocessing caricati correttamente!")

except Exception as e:
    print("❌ Errore durante il caricamento del modello:", e)
    raise e


# ============================================================
# 🔍 FUNZIONE DI PREDIZIONE DELL’ESERCIZIO (PT_AI)
# ============================================================

def predict_exercise(example_dict):
    """
    example_dict deve contenere gli stessi campi del CSV originale
    es: {"Angle1": 30, "Angle2": 45, ..., "Side": "Left"}
    """

    # 1. DataFrame con una sola riga
    df_example = pd.DataFrame([example_dict])

    # 2. Dati numerici
    num_data = df_example[numeric_columns]

    # 3. Dati categorici → get_dummies
    cat_data = pd.get_dummies(
        df_example.drop(columns=numeric_columns),
    )

    # 4. Allineo le colonne categoriche al training
    for col in categorical_columns:
        if col not in cat_data.columns:
            cat_data[col] = 0

    cat_data = cat_data[categorical_columns]

    # 5. Combino numeriche + categoriche
    processed = np.hstack([num_data.values, cat_data.values])

    # 6. Applico scaler
    processed_scaled = scaler.transform(processed)

    # 7. Inferenza
    preds = pt_model.predict(processed_scaled)
    idx = np.argmax(preds, axis=1)[0]

    # 8. Converto indice → label originale
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
    Body JSON esempio:
    {
        "Angle1": 30,
        "Angle2": 45,
        ...,
        "Side": "Left"
    }

    Ritorna:
    {
        "success": true,
        "predicted_exercise": "Squats"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'Nessun dato ricevuto'}), 400

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
# AVVIO SERVER
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
