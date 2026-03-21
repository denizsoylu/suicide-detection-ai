import torch
import speech_recognition as sr
import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect

# 1. AYARLAR VE CİHAZ SEÇİMİ
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSLATION_MODEL_NAME = "facebook/m2m100_418M"

print("Modeller yükleniyor, lütfen bekleyin...")
# Çeviri Modeli Yükleme
tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
translation_model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_NAME).to(DEVICE)


# Not: Ana BERT+BiLSTM modelinizi burada yüklediğinizi varsayıyoruz
# suicide_model = torch.load("model_yolu.pth")

# 2. SESİ METNE ÇEVİRME (STT) FONKSİYONU
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    if not os.path.exists(audio_file_path):
        return "Hata: Ses dosyası bulunamadı."

    try:
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            # Google API üzerinden Türkçe transkripsiyon
            text = recognizer.recognize_google(audio_data, language="tr-TR")
            return text
    except sr.UnknownValueError:
        return "Hata: Ses anlaşılamadı."
    except Exception as e:
        return f"Hata: {str(e)}"


# 3. ÇOK DİLLİ ÇEVİRİ FONKSİYONU
def translate_to_english(text):
    if not text or "Hata" in text:
        return text

    try:
        detected_lang = detect(text)  # Dil tespiti
    except:
        detected_lang = "tr"  # Varsayılan Türkçe kabul et

    tokenizer.src_lang = detected_lang
    encoded = tokenizer(text, return_tensors="pt").to(DEVICE)

    # İngilizceye çevir (en)
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id("en")
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation


# 4. TAM ENTEGRE ANALİZ AKIŞI (PIPELINE)
def full_audio_analysis_pipeline(audio_path):
    print("--- Analiz Başlatıldı ---")

    # Adım 1: Sesi Türkçe Metne Çevir
    tr_text = transcribe_audio(audio_path)
    print(f"1. Algılanan Metin (TR): {tr_text}")

    if "Hata" in tr_text:
        return tr_text

    # Adım 2: Metni İngilizceye Çevir
    en_text = translate_to_english(tr_text)
    print(f"2. Çevrilen Metin (EN): {en_text}")

    # Adım 3: Ana BERT+BiLSTM Modeli ile Tahmin Yap
    # Bu aşama ÜNÜ.docx dosyasında belirtilen %94 doğruluklu modeldir.
    # Örnek: result = suicide_model.predict(en_text)

    print("3. Analiz Tamamlandı.")
    return en_text  # Veya model sonucunu dönün


# TEST ÇALIŞTIRMASI
if __name__ == "__main__":
    audio_path = "test_kaydi.wav"
    # test_kaydi.wav dosyasının var olduğunu varsayalım
    if os.path.exists(audio_path):
        final_result = full_audio_analysis_pipeline(audio_path)
        print(f"Sonuç: {final_result}")
    else:
        print("Lütfen test için bir 'test_kaydi.wav' dosyası ekleyin.")