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

    numeric_columns = prep["numeric_columns"]
    categorical_columns = prep["categorical_columns"]
    scaler = prep["scaler"]
    label_encoder = prep["label_encoder"]

    print("✅ Modello e preprocessing caricati correttamente!")
    print(f"📊 Colonne numeriche attese: {numeric_columns}")
    print(f"📊 Colonne categoriche attese: {categorical_columns}")

except Exception as e:
    print(f"❌ Errore durante il caricamento del modello o del preprocessing: {e}")
    model = None


# ============================================================
# 🔍 FUNZIONE DI PREDIZIONE DELL'ESERCIZIO (PT_AI) - FIXED
# ============================================================

def predict_exercise(example_dict: dict) -> str:
    """
    example_dict deve contenere gli angoli articolari.
    
    Esempio JSON dal client:
    {
        "Shoulder_Angle": 160.0,
        "Elbow_Angle": 140.0,
        "Hip_Angle": 170.0,
        "Knee_Angle": 90.0,
        "Ankle_Angle": 110.0,
        ...
    }
    """

    if model is None:
        raise RuntimeError("Modello non caricato.")

    # 1. DataFrame con una sola riga
    df_example = pd.DataFrame([example_dict])
    
    print(f"📥 Dati ricevuti: {list(example_dict.keys())}")
    print(f"📥 Colonne numeriche attese: {numeric_columns}")

    # 2. Verifica che tutte le colonne numeriche siano presenti
    missing_cols = [col for col in numeric_columns if col not in df_example.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti: {missing_cols}")

    # 3. Estrazione delle colonne numeriche
    df_num = df_example[numeric_columns]
    print(f"✅ Colonne numeriche estratte: {df_num.shape}")

    # 4. Gestione colonne categoriche (se presenti)
    if categorical_columns and len(categorical_columns) > 0:
        print("🔄 Elaborazione colonne categoriche...")
        
        # Trova colonne non numeriche
        remaining_cols = [col for col in df_example.columns if col not in numeric_columns]
        
        if len(remaining_cols) > 0:
            # Crea one-hot encoding
            df_cat = pd.get_dummies(df_example[remaining_cols])
            
            # Allineamento colonne categoriche a quelle di training
            for col in categorical_columns:
                if col not in df_cat.columns:
                    df_cat[col] = 0
            
            # Seleziona solo le colonne del training
            df_cat = df_cat[categorical_columns]
            
            # Concatenazione numeriche + categoriche
            X = np.hstack([df_num.values, df_cat.values])
            print(f"✅ Features totali (numeriche + categoriche): {X.shape}")
        else:
            # Solo colonne numeriche
            X = df_num.values
            print(f"✅ Solo features numeriche: {X.shape}")
    else:
        # Nessuna colonna categorica
        X = df_num.values
        print(f"✅ Solo features numeriche: {X.shape}")

    # 5. Scaling
    X_scaled = scaler.transform(X)
    print(f"✅ Features scalate: {X_scaled.shape}")

    # 6. Predizione
    preds = model.predict(X_scaled, verbose=0)
    idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds, axis=1)[0])

    # 7. Indice → etichetta esercizio
    predicted_label = label_encoder.inverse_transform([idx])[0]
    
    print(f"✅ Predizione: {predicted_label} (confidenza: {confidence:.2%})")

    return predicted_label, confidence


# ============================================================
# 🌐 ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'PT_AI API is running! 🏋️',
        'version': '2.0.0',
        'model_loaded': model is not None,
        'expected_features': {
            'numeric': numeric_columns.tolist() if numeric_columns is not None else [],
            'categorical': categorical_columns.tolist() if categorical_columns is not None else []
        }
    })


@app.route('/api/analyze-exercise', methods=['POST'])
def analyze_exercise():
    """
    Esempio di body JSON atteso:

    {
        "Shoulder_Angle": 160.0,
        "Elbow_Angle": 140.0,
        "Hip_Angle": 170.0,
        "Knee_Angle": 90.0,
        "Ankle_Angle": 110.0,
        "Shoulder_Ground_Angle": 80.0,
        "Elbow_Ground_Angle": 45.0,
        "Hip_Ground_Angle": 90.0,
        "Knee_Ground_Angle": 90.0,
        "Ankle_Ground_Angle": 90.0
    }

    Risposta:

    {
        "success": true,
        "predicted_exercise": "Squats",
        "confidence": 0.95
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'Nessun dato ricevuto'}), 400

        if model is None:
            return jsonify({'success': False, 'error': 'Modello non caricato sul server'}), 500

        print(f"📥 Richiesta ricevuta con {len(data)} features")

        predicted_label, confidence = predict_exercise(data)

        return jsonify({
            "success": True,
            "predicted_exercise": predicted_label,
            "confidence": confidence
        }), 200

    except ValueError as ve:
        print(f"❌ Errore validazione dati: {ve}")
        return jsonify({
            'success': False, 
            'error': str(ve),
            'expected_columns': numeric_columns.tolist() if numeric_columns is not None else []
        }), 400

    except Exception as e:
        print("❌ Errore durante analisi:", e)
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint per verificare lo stato del server"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(numeric_columns) + len(categorical_columns) if numeric_columns is not None else 0
    })


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