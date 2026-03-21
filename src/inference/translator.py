import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect

DEVICE = torch.device("cpu")

MODEL_NAME = "facebook/m2m100_418M"

print("Çeviri modeli yükleniyor...")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)

print("Model hazır.")


def translate_to_english(text: str) -> str:
    # 1️⃣ DİLİ ALGILA
    try:
        detected_lang = detect(text)
    except:
        detected_lang = "en"

    print(f"Algılanan dil: {detected_lang}")

    # 2️⃣ M2M100 DİL KODU AYARLA
    tokenizer.src_lang = detected_lang
    encoded = tokenizer(text, return_tensors="pt").to(DEVICE)

    # 3️⃣ HEDEF DİL: İNGİLİZCE
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id("en")
    )

    # 4️⃣ ÇÖZ
    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return translation


# TEST
if __name__ == "__main__":
    sample_text = "hayat çok zor"
    print("Girdi:", sample_text)
    print("Çeviri:", translate_to_english(sample_text))
