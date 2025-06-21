
import soundfile as sf
import streamlit as st



st.title("English Accent Classifier 🎙️🌍")
st.markdown("Upload a video link (e.g. YouTube, Loom, etc.) and get accent prediction.")

url = st.text_input("Enter public video URL:")

if url:
    with st.spinner("Downloading and analyzing..."):
        audio_path = download_audio(url)
        agent = Agent_classifier()
        result = agent.predict_whole(audio_path)

    st.success("Done!")
    st.markdown(f"**Accent:** `{result['label']}`")
    st.markdown(f"**Confidence Score:** `{round(result['score']*100, 2)}%`")

    st.markdown("**Optional Explanation:**")
    st.markdown(
        f"The model detected this as a `{result['label']}` English accent "
        f"with a confidence of `{round(result['score']*100, 2)}%`. "
        "It uses a fine-tuned wav2vec2 model trained on common English accents."
    )