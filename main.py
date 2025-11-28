import base64
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from load_env import load_dotenv
from text_to_speech import synthesize_speech
from vsl_recognition import (
    SignLanguageRecognizer,
    create_holistic,
    extract_keypoints,
    mediapipe_detection,
    sequence_frames,
)

st.set_page_config(page_title="VSL Prediction", layout="centered")
st.title("DỰ ĐOÁN NGÔN NGỮ KÝ HIỆU")


@st.cache_resource
def get_recognizer():
    return SignLanguageRecognizer()


load_dotenv()
recognizer = get_recognizer()
holistic = create_holistic()
tts_output_dir = Path("Outputs/app_predictions")
tts_output_dir.mkdir(parents=True, exist_ok=True)

def process_webcam_to_sequence():
    cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định
    st.write("⏳ Đang chuẩn bị... Bắt đầu trong 1.5 giây...")
    time.sleep(1.5)  # Hiển thị thông báo trong 1.5 giây
    
    # Đọc video từ webcam trong 4 giây
    st.write("🎥 Đang ghi hình trong 4 giây...")
    sequence = []
    start_time = time.time()

    # Khởi tạo Mediapipe Holistic model
    holistic = create_holistic()
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập webcam")
            break
        elapsed_time = time.time() - start_time
        if elapsed_time > 4:  # Sau 4 giây thì dừng
            break
        # Chuyển đổi frame từ BGR (OpenCV) sang RGB (Mediapipe)
        image, results = mediapipe_detection(frame, holistic)

        # Trích xuất keypoints từ kết quả của Mediapipe
        keypoints = extract_keypoints(results)
        
        # Thêm keypoints vào chuỗi (có thể dừng sau 60 frames hoặc khi người dùng nhấn nút)
        if keypoints is not None:
            sequence.append(keypoints)

        # Hiển thị webcam feed trên Streamlit
        stframe.image(image, channels="BGR", caption="Webcam feed", use_container_width=True)

    cap.release()
    holistic.close()
    
    return sequence

# Streamlit App

def autoplay_audio(audio_path: Path):
    mime = "audio/mpeg"
    if audio_path.suffix.lower() == ".wav":
        mime = "audio/wav"
    elif audio_path.suffix.lower() == ".ogg":
        mime = "audio/ogg"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:{mime};base64,{b64}" type="{mime}">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

input_mode = st.radio("Chọn nguồn đầu vào:", ["🎞️ Video file", "📷 Webcam"])

sequence = None
if input_mode == "🎞️ Video file":
    uploaded_file = st.file_uploader("Tải lên video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        if st.button("🔍 Dự đoán từ video"):
            sequence = sequence_frames(tmp_path, holistic)

elif input_mode == "📷 Webcam":
    st.warning("Nhấn nút bên dưới để bắt đầu ghi hình từ webcam.")
    if st.button("📸 Ghi và dự đoán"):
        sequence = process_webcam_to_sequence()

# Dự đoán
if sequence is not None:
    try:
        result = recognizer.predict_from_sequence(sequence)
    except ValueError:
        st.error("Không thu được dữ liệu đầu vào hợp lệ. Vui lòng thử lại.")
    else:
        confidence_pct = result.confidence * 100
        st.success(f"✅ Nhãn dự đoán: **{result.label}** ({confidence_pct:.2f}%)")

        recognized_text = result.label
        try:
            audio_file = tts_output_dir / f"prediction_{int(time.time())}.mp3"
            synthesize_speech(recognized_text, audio_file, voice="coral")
            autoplay_audio(audio_file)
            st.info("🔊 Văn bản đã được phát ngay lập tức.")
        except Exception as tts_error:
            st.warning(f"Không thể phát âm thanh TTS: {tts_error}")
