from speech_to_text import speech_to_text
from translator import translate_to_english


def voice_to_english_text():
    # 1. Sesten metne
    text = speech_to_text()

    if not text:
        print("Metin alınamadı")
        return

    # 2. İngilizceye çevir
    translated = translate_to_english(text)

    print("\n✅ Nihai Çıktı")
    print("Türkçe:", text)
    print("İngilizce:", translated)


if __name__ == "__main__":
    voice_to_english_text()
