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
    
    import tensorflow as tf
    from tensorflow.keras.models import model_from_json
    
    # Carica preprocessing
    with open("pt_ai_preprocessing_nn.pkl", "rb") as f:
        prep = pickle.load(f)

    numeric_columns = prep["numeric_columns"]
    categorical_columns = prep["categorical_columns"]
    scaler = prep["scaler"]
    label_encoder = prep["label_encoder"]
    
    # Carica architettura dal JSON
    print("📥 Caricamento architettura modello...")
    with open("pt_ai_nn_architecture.json", "r") as json_file:
        model_json = json_file.read()
    
    model = model_from_json(model_json)
    
    # Carica i pesi
    print("📥 Caricamento pesi modello...")
    model.load_weights("pt_ai_nn_model.weights.h5")
    
    # Compila il modello
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ Modello e preprocessing caricati correttamente!")
    print(f"📊 Colonne numeriche attese: {numeric_columns}")
    print(f"📊 Colonne categoriche attese: {categorical_columns}")

except Exception as e:
    print(f"❌ Errore durante il caricamento del modello o del preprocessing: {e}")
    print(traceback.format_exc())
    model = None


# ============================================================
# 🔍 FUNZIONE DI PREDIZIONE DELL'ESERCIZIO (PT_AI) - FIXED
# ============================================================

def predict_exercise(example_dict: dict):
    """
    Predice l'esercizio basandosi su angoli articolari e lato del corpo.
    
    Input atteso:
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
        "Ankle_Ground_Angle": 90.0,
        "Side": "Left"  # opzionale
    }
    """

    if model is None:
        raise RuntimeError("Modello non caricato.")

    # 1. Crea DataFrame
    df_example = pd.DataFrame([example_dict])
    
    print(f"📥 Dati ricevuti: {list(example_dict.keys())}")
    print(f"📥 Colonne numeriche attese: {numeric_columns}")

    # 2. Verifica colonne numeriche
    missing_cols = [col for col in numeric_columns if col not in df_example.columns]
    if missing_cols:
        raise ValueError(f"Colonne numeriche mancanti: {missing_cols}")

    # 3. Estrai colonne numeriche
    df_num = df_example[numeric_columns].values
    print(f"✅ Colonne numeriche estratte: {df_num.shape}")

    # 4. Scala SOLO le colonne numeriche
    X_scaled = scaler.transform(df_num)
    print(f"✅ Features numeriche scalate: {X_scaled.shape}")

    # 5. Gestione colonne categoriche
    if categorical_columns and len(categorical_columns) > 0:
        print("🔄 Elaborazione colonne categoriche...")
        
        # Trova colonne categoriche nei dati ricevuti
        cat_cols_present = [col for col in df_example.columns if col not in numeric_columns]
        
        if len(cat_cols_present) > 0:
            # Crea one-hot encoding
            df_cat = pd.get_dummies(df_example[cat_cols_present])
            
            # Crea un DataFrame vuoto con tutte le colonne categoriche del training
            df_cat_aligned = pd.DataFrame(0, index=[0], columns=categorical_columns)
            
            # Riempi le colonne presenti
            for col in df_cat.columns:
                if col in df_cat_aligned.columns:
                    df_cat_aligned[col] = df_cat[col].values
            
            print(f"✅ Colonne categoriche allineate: {df_cat_aligned.shape}")
            
            # Concatena numeriche scalate + categoriche
            X_final = np.hstack([X_scaled, df_cat_aligned.values])
            print(f"✅ Features finali (num + cat): {X_final.shape}")
        else:
            # Nessuna colonna categorica nei dati → usa solo numeriche
            X_final = X_scaled
            print(f"✅ Solo features numeriche: {X_final.shape}")
    else:
        # Modello senza colonne categoriche
        X_final = X_scaled
        print(f"✅ Solo features numeriche: {X_final.shape}")

    # 6. Predizione
    try:
        preds = model.predict(X_final, verbose=0)
        idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds, axis=1)[0])

        # 7. Decodifica label
        predicted_label = label_encoder.inverse_transform([idx])[0]
        
        print(f"✅ Predizione: {predicted_label} (confidenza: {confidence:.2%})")
        
        return predicted_label, confidence
        
    except Exception as e:
        print(f"❌ Errore durante predizione: {e}")
        print(f"Shape features finali: {X_final.shape}")
        print(f"Shape atteso dal modello: {model.input_shape}")
        raise


# ============================================================
# 🌐 ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'PT_AI API is running! 🏋️',
        'version': '2.0.1',
        'model_loaded': model is not None,
        'expected_features': {
            'numeric': numeric_columns.tolist() if numeric_columns is not None else [],
            'categorical': categorical_columns if categorical_columns is not None else []
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
        "Ankle_Ground_Angle": 90.0,
        "Side": "Left"
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
            'expected_numeric': numeric_columns.tolist() if numeric_columns is not None else [],
            'expected_categorical': categorical_columns if categorical_columns is not None else []
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
        'numeric_features': len(numeric_columns) if numeric_columns is not None else 0,
        'categorical_features': len(categorical_columns) if categorical_columns is not None else 0,
        'total_features': (len(numeric_columns) if numeric_columns is not None else 0) + 
                         (len(categorical_columns) if categorical_columns is not None else 0)
    })


# ============================================================
# AVVIO SERVER
# ============================================================

if __name__ == '__main__':
    import os
    
    print("=" * 50)
    print("🚀 PT_AI API Server Starting...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📂 Files: {os.listdir('.')}")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # ← IMPORTANTE per produzione
    )