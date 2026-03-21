import streamlit as st
import os
from src.inference.model_inference import predict
from src.inference.speech_to_text import transcribe_audio

# Sayfa Ayarları
st.set_page_config(page_title="Suicide Detection AI", page_icon="🧠", layout="wide")

# --- SOL MENÜ (SADE) ---
# Sadece logoyu bıraktım, yazıların hepsini sildim.
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)

# --- ANA BAŞLIK ---
st.title("🧠 Suicide Detection AI")
st.write("Yapay zeka destekli intihar riski tespit sistemi.")
st.markdown("---")

# --- SEKMELER ---
tab1, tab2 = st.tabs(["📝 Metin Analizi", "🎙️ Sesli Analiz"])

# --- TAB 1: METİN GİRİŞİ ---
with tab1:
    user_input = st.text_area("Analiz edilecek metni girin:", height=150,
                              placeholder="Bugün kendimi nasıl hissediyorum...")

    if st.button("Analiz Et", key="text_btn"):
        if user_input:
            with st.spinner('Yapay zeka düşünüyor...'):
                result = predict(user_input)

            st.divider()

            # Sonuçları Göster
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sonuç:")
                if result["label"] == "SUICIDE":
                    st.error("⚠️ YÜKSEK RİSK TESPİT EDİLDİ")
                elif result["label"] == "ERROR":
                    st.warning("Bir hata oluştu.")
                else:
                    st.success("✅ RİSK ALGILANMADI (GÜVENLİ)")

            with col2:
                st.subheader("Güven Skoru:")
                score = result["score"]
                st.progress(score)
                st.caption(f"Modelin Emin Olma Oranı: %{score * 100:.1f}")

            with st.expander("Detaylı Raporu Gör"):
                if "translated_text" in result:
                    st.write(f"**İşlenen Metin (EN):** {result['translated_text']}")
                st.json(result.get("details", {}))

# --- TAB 2: SESLİ GİRİŞ ---
with tab2:
    st.write("Mikrofon butonuna basarak konuşabilirsiniz.")

    # Streamlit ses kaydedicisi
    audio_value = st.audio_input("Sesinizi Kaydedin")

    if audio_value:
        st.audio(audio_value)

        if st.button("Sesi Analiz Et", key="audio_btn"):
            with st.spinner("Ses metne dönüştürülüyor..."):
                # 1. Sesi kaydet
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_value.read())

                # 2. Yazıya çevir (Türkçe)
                transcribed_text = transcribe_audio("temp_audio.wav")

                # 3. Dosyayı temizle
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")

            # Hata kontrolü
            if "Hata" in transcribed_text:
                st.error(transcribed_text)
            else:
                st.success(f"🗣️ Algılanan: **{transcribed_text}**")

                # 4. Çeviri ve Analiz (Otomatik)
                with st.spinner("Yapay zeka analiz ediyor..."):
                    result = predict(transcribed_text)

                st.divider()
                if result["label"] == "SUICIDE":
                    st.error(f"⚠️ YÜKSEK RİSK")
                else:
                    st.success(f"✅ GÜVENLİ")

                st.progress(result["score"])
                st.progress(result["score"])
                st.caption(f"Risk Skoru: %{result['score'] * 100:.1f}")