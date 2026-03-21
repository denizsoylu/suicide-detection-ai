import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from deep_translator import GoogleTranslator  # Çeviri için bunu ekledik
import os


# 1. MODEL MİMARİSİ (BiLSTM Hibrit Yapısı)
class BertBiLSTM(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(BertBiLSTM, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256 * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = lstm_out[:, -1, :]
        logits = self.classifier(self.dropout(lstm_out))
        return logits


# 2. AYARLAR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/suicide_bert/bert_bilstm_model.pth"
TOKENIZER_PATH = "models/suicide_bert/"

# Global değişkenler
tokenizer = None
model = None

# 3. YÜKLEME İŞLEMİ
try:
    print(f"🧠 BiLSTM Modeli Yükleniyor... Cihaz: {DEVICE}")

    # Tokenizer Yükle
    if os.path.exists(os.path.join(TOKENIZER_PATH, "vocab.txt")):
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Model Yükle
    model = BertBiLSTM(model_name="bert-base-uncased", num_labels=2)

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model ağırlıkları başarıyla yüklendi!")
    else:
        print(f"⚠️ UYARI: Model dosyası ({MODEL_PATH}) bulunamadı. Boş modelle devam ediliyor.")

    model.to(DEVICE)
    model.eval()

except Exception as e:
    print(f"❌ Başlatma hatası: {e}")
    # En kötü senaryoda çalışması için varsayılan tokenizer
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 4. TAHMİN VE ÇEVİRİ FONKSİYONU (Düzeltilen Kısım)
def predict(text: str):
    global tokenizer, model

    translated_text = text  # Varsayılan olarak aynısı kalsın

    # A) Çeviri Adımı (Türkçe -> İngilizce)
    try:
        # Türkçe karakterleri İngilizce modele uygun hale getiriyoruz
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Çeviri hatası: {e}")
        translated_text = text  # Çeviri yapılamazsa orijinali kullan

    if tokenizer is None or model is None:
        return {"label": "ERROR", "score": 0.0, "translated_text": text}

    # B) Model Tahmini
    try:
        inputs = tokenizer(translated_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(
            DEVICE)

        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        suicide_prob = probabilities[0][1].item()

        # Eşik değeri (Threshold)
        label = "SUICIDE" if suicide_prob > 0.5 else "NON-SUICIDE"

        return {
            "label": label,
            "score": suicide_prob,
            "translated_text": translated_text,  # <-- İŞTE BU EKSİKTİ, EKLENDİ!
            "details": {"suicide": suicide_prob, "non-suicide": 1 - suicide_prob}
        }
    except Exception as e:
        return {"label": "ERROR", "score": 0.0, "translated_text": translated_text, "details": str(e)}