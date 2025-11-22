from tensorflow.keras.models import load_model

# Carica il modello .keras
print("📥 Caricamento modello .keras...")
model = load_model('pt_ai_nn_model.keras')

# Salva in formato .h5
print("💾 Salvataggio in formato .h5...")
model.save('pt_ai_nn_model.h5', save_format='h5')

print("✅ Conversione completata!")
print("📁 File creato: pt_ai_nn_model.h5")