import pandas as pd
import nltk
import os
import re
# 'dont' -> 'do not' gibi dönüşümler için bu kütüphaneyi kullanacağız

import contractions
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from collections import Counter

# NLP işlemleri için ihtiyacım olan temel paketleri indiriyorum
nltk.download("punkt", quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Konsol çıktılarında veriyi daha rahat görebilmek için ayarlarımı yapıyorum
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def safe_contractions_fix(text):
    """
    Bazı satırlarda contractions kütüphanesi 'IndexError' hatası verebiliyor.
    Bu fonksiyon hata aldığında metni olduğu gibi döndürerek kodun çökmesini engelliyor.
    """
    try:
        if pd.isna(text) or str(text).strip() == "":
            return str(text)
        return contractions.fix(str(text))
    except Exception:
        # Eğer kütüphane hata verirse, orijinal metni bozmadan geri döndür
        return str(text)


def preprocess_data():
    # 1. Veri Yükleme Aşaması
    # Projenin farklı bilgisayarlarda da çalışabilmesi için dosya yolunu dinamik hale getiriyorum
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    # Ana dizindeki data/raw klasöründen orijinal veri setimin yolunu oluşturuyorum
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'Suicide_Detection.csv')

    # Hata yapmamak için veriyi hangi yolda aradığımı ekrana yazdırıyorum
    print(f"Veri şu adreste aranıyor: {raw_data_path}")

    if not os.path.exists(raw_data_path):
        print(f"Hata: Dosyayı bulamadım! Lütfen veri setinin doğru klasörde olduğundan emin ol.")
        return

    print("Veri yükleniyor, lütfen bekleyin...")
    df = pd.read_csv(raw_data_path)
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    # 2. Metin Düzeltme ve Temizlik
    print("Metinler normalize ediliyor (kısaltmalar düzeltiliyor, temizlik yapılıyor)...")

    # Hata aldığımız 'contractions.fix' işlemini artık 'safe' fonksiyonumuzla yapıyoruz
    df['text'] = df['text'].apply(safe_contractions_fix)

    df['text'] = df['text'].str.lower()
    # Noktalama işaretlerini ve sayıları temizliyorum
    df['text'] = df['text'].str.replace('[^\\w\\s]', '', regex=True)
    df['text'] = df['text'].str.replace('\\d', '', regex=True)

    # 3. Etkisiz Kelimelerin (Stop Words) Temizlenmesi
    print("Anlam taşımayan gereksiz kelimeleri temizliyorum...")
    sw = stopwords.words('english')
    # Kütüphanenin kaçırabileceği çok kısa internet jargonlarını hala manuel ekleyebilirim
    sw.extend(['u', 'ur', 'id', 'get', 'know'])

    df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    # 4. Nadir Kelimelerin (Rare Words) Ayıklanması
    print("Sadece bir kez geçen ve modelin kafasını karıştırabilecek nadir kelimeleri buluyorum...")
    # 232 bin satırda hızlı işlem yapabilmek için Counter kullanıyorum
    all_words = Counter(" ".join(df['text']).split())
    drops = {word for word, count in all_words.items() if count == 1}

    print(f"{len(drops)} adet nadir kelimeyi veri setinden çıkarıyorum...")
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

    # 5. Kelime Köklerine İnme (Lemmatization)
    print("Kelimeleri köklerine indirgeyerek anlamsal bütünlük sağlıyorum (Bu işlem biraz vakit alabilir)...")
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # 6. İşlenmiş Veriyi Kaydetme
    # Model eğitiminde kullanacağım tertemiz veriyi 'processed' klasörüne kaydediyorum
    output_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_suicide_data.csv')

    # Hedef klasör yoksa otomatik olarak oluşturuyorum
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = df.dropna(subset=['text', 'class'])
    df = df[df['text'].str.strip() != ""]
    df.to_csv(output_path, index=False)
    print(f"İşlem başarıyla tamamlandı! Temizlenmiş verimi şuraya kaydettim: {output_path}")
    print("Verinin son halinden bir kesit:")
    print(df.head())


if __name__ == "__main__":
    preprocess_data()
