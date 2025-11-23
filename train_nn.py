"""
PT AI Model Training & Saving Script
Training + Salvataggio completo del modello per FitBot AI
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime

# ============================================
# CONFIGURAZIONE
# ============================================

DATASET_PATH = "exercise_angles.csv"
OUTPUT_DIR = "pt_ai_model"
MODEL_NAME = "pt_ai_nn_model"
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Crea directory output
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("🚀 PT AI MODEL TRAINING - FitBot AI")
print("="*60)
print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📂 Dataset: {DATASET_PATH}")
print(f"💾 Output: {OUTPUT_DIR}/")
print("="*60)

# ============================================
# CARICAMENTO DATASET
# ============================================

print("\n📊 Caricamento dataset...")
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"📋 Esercizi rilevati: {df['Label'].nunique()}")
    print(f"   {list(df['Label'].unique())}")
except FileNotFoundError:
    print(f"❌ ERRORE: File '{DATASET_PATH}' non trovato!")
    print("   Scarica il dataset da:")
    print("   https://www.kaggle.com/datasets/mrigaankjaswal/exercise-detection-dataset")
    exit(1)

# ============================================
# PREPROCESSING
# ============================================

print("\n🔧 Preprocessing dati...")

# Separa la colonna delle etichette (Label)
labels = df['Label']

# Seleziona le colonne numeriche
numeric_columns = df.select_dtypes(include=[np.number])
print(f"   Colonne numeriche: {len(numeric_columns.columns)}")

# Seleziona le colonne categoriche (escludi Label)
categorical_columns = df.select_dtypes(include=['object']).drop(columns=['Label'])
print(f"   Colonne categoriche: {len(categorical_columns.columns)}")

# Trasforma le colonne categoriche con get_dummies
categorical_data = pd.get_dummies(categorical_columns)

# Normalizza le colonne numeriche
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_columns)
print(f"✅ Normalizzazione completata")

# Codifica le etichette con LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
print(f"✅ Label encoding: {num_classes} classi")

# Converte le etichette in formato categorico
categorical_labels = to_categorical(encoded_labels)

# Combina i dati numerici e categorici
processed_data = np.hstack([numeric_data_scaled, categorical_data])
print(f"✅ Dati processati: shape {processed_data.shape}")

# ============================================
# SPLIT DATASET
# ============================================

print(f"\n📊 Split dataset...")
print(f"   Training: {70}%")
print(f"   Validation: {15}%")
print(f"   Test: {15}%")

# Dividi il dataset in training, validation e test
X_train, X_temp, y_train, y_temp = train_test_split(
    processed_data, 
    categorical_labels, 
    test_size=0.30, 
    random_state=RANDOM_STATE,
    stratify=encoded_labels  # Mantieni distribuzione classi
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.50, 
    random_state=RANDOM_STATE
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Validation set: {X_val.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# ============================================
# COSTRUZIONE MODELLO
# ============================================

print(f"\n🏗️  Costruzione modello...")

model = Sequential([
    Dense(128, input_shape=(processed_data.shape[1],), activation='relu', name='dense_1'),
    Dropout(0.3),
    Dense(64, activation='relu', name='dense_2'),
    Dropout(0.2),
    Dense(32, activation='relu', name='dense_3'),
    Dense(num_classes, activation='softmax', name='output')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ============================================
# CALLBACKS
# ============================================

print(f"\n⚙️  Configurazione callbacks...")

# Early stopping per prevenire overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Riduzione learning rate quando plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# ============================================
# TRAINING
# ============================================

print(f"\n🚀 Inizio training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print("-"*60)

history = model.fit(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    validation_data=(X_val, y_val), 
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("-"*60)
print("✅ Training completato!")

# ============================================
# VALUTAZIONE
# ============================================

print(f"\n📊 Valutazione sul test set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Valutazione dettagliata
predictions = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Calcola accuracy per classe
from sklearn.metrics import classification_report, confusion_matrix

print(f"\n📈 Report dettagliato:")
print(classification_report(
    true_labels, 
    predicted_labels, 
    target_names=label_encoder.classes_,
    zero_division=0
))

# Confusion Matrix
print(f"\n🔢 Confusion Matrix:")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print(conf_matrix)

# ============================================
# VISUALIZZAZIONE
# ============================================

print(f"\n📊 Generazione grafici...")

# Plot 1: Loss
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Loss over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Accuracy over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
plt.subplot(1, 3, 3)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45, ha='right')
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_metrics.png', dpi=300, bbox_inches='tight')
print(f"✅ Grafici salvati: {OUTPUT_DIR}/training_metrics.png")
plt.close()

# ============================================
# SALVATAGGIO MODELLO
# ============================================

print(f"\n💾 Salvataggio modello...")

# 1. Salva i PESI del modello
weights_path = f'{OUTPUT_DIR}/{MODEL_NAME}.weights.h5'
model.save_weights(weights_path)
print(f"✅ Pesi salvati: {weights_path}")

# 2. Salva l'ARCHITETTURA in JSON
architecture_path = f'{OUTPUT_DIR}/{MODEL_NAME}_architecture.json'
model_json = model.to_json()
with open(architecture_path, 'w') as json_file:
    json_file.write(model_json)
print(f"✅ Architettura salvata: {architecture_path}")

# 3. Salva PREPROCESSING (scaler, encoder, colonne)
preprocessing_path = f'{OUTPUT_DIR}/{MODEL_NAME}_preprocessing.pkl'
preprocessing_info = {
    'scaler': scaler,
    'label_encoder': label_encoder,
    'numeric_columns': list(numeric_columns.columns),
    'categorical_columns': list(categorical_data.columns),
    'model_architecture': model_json,
    'input_shape': processed_data.shape[1],
    'output_shape': num_classes,
    'classes': list(label_encoder.classes_),
    'num_classes': num_classes
}

with open(preprocessing_path, 'wb') as f:
    pickle.dump(preprocessing_info, f)
print(f"✅ Preprocessing salvato: {preprocessing_path}")

# 4. Salva METADATA e METRICHE
metadata_path = f'{OUTPUT_DIR}/{MODEL_NAME}_metadata.json'
metadata = {
    'model_name': MODEL_NAME,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_path': DATASET_PATH,
    'dataset_shape': df.shape,
    'num_samples': len(df),
    'num_classes': num_classes,
    'classes': list(label_encoder.classes_),
    'training': {
        'epochs': len(history.history['loss']),
        'batch_size': BATCH_SIZE,
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0]
    },
    'metrics': {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    },
    'architecture': {
        'input_shape': processed_data.shape[1],
        'layers': [
            {'type': 'Dense', 'units': 128, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.3},
            {'type': 'Dense', 'units': 64, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.2},
            {'type': 'Dense', 'units': 32, 'activation': 'relu'},
            {'type': 'Dense', 'units': num_classes, 'activation': 'softmax'}
        ]
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata salvati: {metadata_path}")

# 5. Salva ESEMPI di predizione
examples_path = f'{OUTPUT_DIR}/{MODEL_NAME}_examples.txt'
with open(examples_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("ESEMPI DI PREDIZIONI - PT AI MODEL\n")
    f.write("="*60 + "\n\n")
    
    for i in range(min(20, len(predicted_labels))):
        pred_label = label_encoder.inverse_transform([predicted_labels[i]])[0]
        true_label = label_encoder.inverse_transform([true_labels[i]])[0]
        confidence = predictions[i][predicted_labels[i]] * 100
        
        status = "✅" if pred_label == true_label else "❌"
        f.write(f"{status} Sample {i+1}:\n")
        f.write(f"   Predicted: {pred_label} ({confidence:.1f}%)\n")
        f.write(f"   True:      {true_label}\n\n")

print(f"✅ Esempi salvati: {examples_path}")

# ============================================
# TEST CARICAMENTO
# ============================================

print(f"\n🧪 Test caricamento modello...")

try:
    from tensorflow.keras.models import model_from_json
    
    # Ricarica architettura
    with open(architecture_path, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    
    # Ricarica pesi
    loaded_model.load_weights(weights_path)
    
    # Test predizione
    test_sample = X_test[0:1]
    original_pred = model.predict(test_sample, verbose=0)
    loaded_pred = loaded_model.predict(test_sample, verbose=0)
    
    if np.allclose(original_pred, loaded_pred):
        print("✅ Test caricamento: SUCCESS")
        print("   Il modello salvato funziona correttamente!")
    else:
        print("⚠️  Test caricamento: ATTENZIONE")
        print("   Le predizioni differiscono leggermente")
        
except Exception as e:
    print(f"❌ Errore test caricamento: {e}")

# ============================================
# SUMMARY FINALE
# ============================================

print("\n" + "="*60)
print("🎉 TRAINING COMPLETATO CON SUCCESSO!")
print("="*60)
print(f"\n📁 File generati in '{OUTPUT_DIR}/':")
print(f"   1. {MODEL_NAME}.weights.h5          - Pesi del modello")
print(f"   2. {MODEL_NAME}_architecture.json   - Architettura")
print(f"   3. {MODEL_NAME}_preprocessing.pkl   - Scaler + Encoder")
print(f"   4. {MODEL_NAME}_metadata.json       - Metriche e info")
print(f"   5. {MODEL_NAME}_examples.txt        - Esempi predizioni")
print(f"   6. training_metrics.png             - Grafici training")

print(f"\n📊 Metriche finali:")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")

if test_accuracy >= 0.90:
    print(f"\n🏆 ECCELLENTE! Accuracy > 90%")
elif test_accuracy >= 0.85:
    print(f"\n✅ OTTIMO! Accuracy > 85% - Pronto per produzione")
elif test_accuracy >= 0.75:
    print(f"\n👍 BUONO! Considera di migliorare il modello")
else:
    print(f"\n⚠️  Accuracy < 75% - Training aggiuntivo consigliato")

print(f"\n🚀 Next Steps:")
print(f"   1. Copia i file in 'backend/' della tua app")
print(f"   2. Avvia il server: python pt_ai_server.py")
print(f"   3. Testa con l'app Flutter!")

print("\n" + "="*60)
print("💪 Buon allenamento con PT AI!")
print("="*60 + "\n")