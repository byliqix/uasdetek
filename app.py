import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from collections import deque

st.set_page_config(
    page_title="Deteksi Kelelahan",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CSS =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}
h1, h2, h3 {
    color: #00d4ff !important;
}
.metric-card {
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
}
.metric-label {
    font-size: 0.9rem;
    text-align: center;
    color: #ccc;
}
.alert { color: #00ff88; font-weight: bold; }
.slightly-drowsy { color: #ffff00; font-weight: bold; }
.drowsy { color: #ff9900; font-weight: bold; }
.danger { color: #ff3333; font-weight: bold; animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ===================== SESSION STATE =====================
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None

# ===================== DETEKSI WAJAH =====================
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_eye_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

face_cascade = load_face_cascade()
eye_cascade = load_eye_cascade()

# ===================== EAR CALCULATOR =====================
def calculate_ear(eye_landmarks):
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0

def estimate_ear_from_eye_roi(eye_roi):
    h, w = eye_roi.shape[:2]
    if h == 0 or w == 0:
        return 0.25
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x_, y_, w_, h_ = cv2.boundingRect(largest)
        aspect = h_ / (w_ + 1e-6)
        ear = aspect / 2.0
        return min(ear, 0.5)
    return 0.25

# ===================== PERCLOS =====================
class PERCLOSCalculator:
    def __init__(self, ear_threshold=0.2, window_size=60):
        self.ear_threshold = ear_threshold
        self.window_size = window_size
        self.eye_states = deque(maxlen=window_size)

    def update(self, ear_value):
        is_closed = ear_value < self.ear_threshold
        self.eye_states.append(is_closed)
        return is_closed

    def get_perclos(self):
        if len(self.eye_states) == 0:
            return 0
        return sum(self.eye_states) / len(self.eye_states) * 100

    def get_fatigue_level(self):
        perclos = self.get_perclos()
        if perclos < 20:
            return "Alert", (0, 255, 0)
        elif perclos < 40:
            return "Slightly Drowsy", (0, 255, 255)
        elif perclos < 60:
            return "Drowsy", (255, 165, 0)
        else:
            return "Very Drowsy - DANGER!", (0, 0, 255)

# ===================== MAR (Yawn) =====================
class YawnDetector:
    def __init__(self, mar_threshold=0.6, consecutive_frames=3):
        self.mar_threshold = mar_threshold
        self.consecutive_frames = consecutive_frames
        self.yawn_count = 0
        self.frame_count = 0
        self.is_yawning = False

    def detect(self, mar_value):
        if mar_value > self.mar_threshold:
            self.frame_count += 1
            if self.frame_count >= self.consecutive_frames:
                if not self.is_yawning:
                    self.yawn_count += 1
                    self.is_yawning = True
        else:
            self.frame_count = 0
            self.is_yawning = False
        return self.is_yawning

    def reset(self):
        self.yawn_count = 0
        self.frame_count = 0
        self.is_yawning = False

def estimate_mar_from_mouth_roi(mouth_roi):
    h, w = mouth_roi.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY) if len(mouth_roi.shape) == 3 else mouth_roi
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x_, y_, w_, h_ = cv2.boundingRect(largest)
        aspect = h_ / (w_ + 1e-6)
        mar = aspect * 2.0
        return min(mar, 1.0)
    return 0.0

# ===================== HEAD POSE =====================
class HeadPoseDetector:
    def __init__(self, pitch_threshold=20):
        self.pitch_threshold = pitch_threshold
        self.nodding_count = 0
        self.is_nodding = False

    def detect(self, pitch_angle):
        if pitch_angle > self.pitch_threshold:
            if not self.is_nodding:
                self.nodding_count += 1
                self.is_nodding = True
        else:
            self.is_nodding = False
        return self.is_nodding

    def reset(self):
        self.nodding_count = 0
        self.is_nodding = False

def estimate_head_pitch(face_rect):
    x, y, w, h = face_rect
    nose_y = y + h * 0.6
    chin_y = y + h
    eye_center_y = y + h * 0.35
    offset = nose_y - eye_center_y
    pitch = (offset / (h + 1e-6)) * 90
    return pitch

# ===================== MAIN DETECTOR =====================
class FatigueDetector:
    def __init__(self):
        self.perclos = PERCLOSCalculator()
        self.yawn = YawnDetector()
        self.head = HeadPoseDetector()
        self.blink_count = 0
        self.prev_eye_state = False
        self.start_time = time.time()
        self.weights = {"perclos": 0.4, "yawn": 0.25, "head_pose": 0.2, "blink_rate": 0.15}

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        result = {
            "face_detected": len(faces) > 0,
            "ear": 0.25,
            "mar": 0.0,
            "perclos": 0.0,
            "is_yawning": False,
            "is_nodding": False,
            "fatigue_score": 0,
            "fatigue_level": "Alert",
            "color": (0, 255, 0),
            "blink_rate": 0,
            "faces": faces,
        }

        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3)
            ear_values = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_roi = face_roi_color[ey:ey+eh, ex:ex+ew]
                ear = estimate_ear_from_eye_roi(eye_roi)
                ear_values.append(ear)
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)

            ear = np.mean(ear_values) if ear_values else 0.25
            result["ear"] = ear

            is_closed = self.perclos.update(ear)
            if is_closed and not self.prev_eye_state:
                self.blink_count += 1
            self.prev_eye_state = is_closed

            result["perclos"] = self.perclos.get_perclos()

            elapsed = time.time() - self.start_time
            blink_rate = self.blink_count / (elapsed / 60 + 0.01)
            result["blink_rate"] = blink_rate

            mouth_h = int(h * 0.15)
            mouth_y = y + int(h * 0.7)
            mouth_roi = frame[mouth_y:mouth_y+mouth_h, x:x+w]
            if mouth_roi.size > 0:
                mar = estimate_mar_from_mouth_roi(mouth_roi)
                result["mar"] = mar
                result["is_yawning"] = self.yawn.detect(mar)
                cv2.rectangle(frame, (x, mouth_y), (x+w, mouth_y+mouth_h), (0, 255, 255), 1)

            pitch = estimate_head_pitch((x, y, w, h))
            result["is_nodding"] = self.head.detect(pitch)

            perclos_score = result["perclos"] / 100
            yawn_score = 1.0 if result["is_yawning"] else 0.0
            head_score = 1.0 if result["is_nodding"] else 0.0
            blink_score = min(blink_rate / 30, 1.0) if (blink_rate > 25 or blink_rate < 8) else 0

            fatigue_score = (
                self.weights["perclos"] * perclos_score
                + self.weights["yawn"] * yawn_score
                + self.weights["head_pose"] * head_score
                + self.weights["blink_rate"] * blink_score
            )
            result["fatigue_score"] = fatigue_score * 100

            if fatigue_score < 0.25:
                result["fatigue_level"] = "Alert"
                result["color"] = (0, 255, 0)
            elif fatigue_score < 0.5:
                result["fatigue_level"] = "Slightly Drowsy"
                result["color"] = (0, 255, 255)
            elif fatigue_score < 0.75:
                result["fatigue_level"] = "Drowsy"
                result["color"] = (0, 165, 255)
            else:
                result["fatigue_level"] = "Very Drowsy - DANGER!"
                result["color"] = (0, 0, 255)

            label = f"{result['fatigue_level']} | Score: {result['fatigue_score']:.1f}%"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, result["color"], 2)

        return frame, result

# ===================== SIDEBAR =====================
st.sidebar.title("😴 Deteksi Kelelahan")
st.sidebar.markdown("### Menu")
menu = st.sidebar.radio("", ["Dashboard", "Deteksi Real-time", "Tentang"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Info Sistem**")
st.sidebar.info(
    "Mendeteksi kelelahan menggunakan:\n"
    "- PERCLOS (Eye Closure Ratio)\n"
    "- Yawn Detection (MAR)\n"
    "- Head Pose Estimation\n"
    "- Multi-modal Fusion"
)

# ===================== DASHBOARD =====================
if menu == "Dashboard":
    st.title("🚗 Sistem Deteksi Kelelahan")
    st.markdown("### Multi-Modal Fatigue Detection System")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">👁️ PERCLOS</div>
            <div style="text-align:center;font-size:1rem;">Mendeteksi persentase mata tertutup dalam window waktu</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">👄 Yawn Detection</div>
            <div style="text-align:center;font-size:1rem;">Mendeteksi menguap berdasarkan MAR</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🤖 Head Pose</div>
            <div style="text-align:center;font-size:1rem;">Mendeteksi anggukan kepala (nodding)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Cara Kerja")
    st.markdown("""
    1. **Face Detection** - Mendeteksi wajah menggunakan Haar Cascade
    2. **Eye Tracking** - Menghitung EAR (Eye Aspect Ratio) untuk deteksi mata tertutup
    3. **PERCLOS** - Menghitung persentase waktu mata tertutup dalam 60 frame terakhir
    4. **Yawn Detection** - Mendeteksi menguap berdasarkan rasio aspek mulut
    5. **Head Pose** - Mendeteksi kepala menunduk (nodding)
    6. **Fusion** - Menggabungkan semua sinyal dengan bobot yang telah ditentukan
    """)

    st.markdown("### Bobot Fusion")
    weights_data = {
        "Metrik": ["PERCLOS", "Yawn", "Head Pose", "Blink Rate"],
        "Bobot": [0.40, 0.25, 0.20, 0.15]
    }
    st.table(pd.DataFrame(weights_data))

    st.markdown("### Tingkat Kelelahan")
    level_data = {
        "Skor Fatigue": ["0-25%", "25-50%", "50-75%", "75-100%"],
        "Level": ["Alert 🟢", "Slightly Drowsy 🟡", "Drowsy 🟠", "Very Drowsy - DANGER! 🔴"],
    }
    st.table(pd.DataFrame(level_data))

# ===================== REAL-TIME DETECTION =====================
elif menu == "Deteksi Real-time":
    st.title("📷 Deteksi Kelelahan Real-time")

    col1, col2 = st.columns([2, 1])

    with col1:
        run = st.checkbox("Mulai Deteksi", key="run_detection")
        FRAME_WINDOW = st.image([], use_container_width=True)

    with col2:
        st.markdown("### Status")
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if run:
        st.session_state.running = True
        detector = FatigueDetector()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Tidak dapat mengakses kamera. Pastikan kamera terhubung.")
            st.stop()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        status_container = st.empty()

        while st.session_state.running and run:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca frame dari kamera.")
                break

            processed_frame, result = detector.detect(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(processed_frame)

            level_class = {
                "Alert": "alert",
                "Slightly Drowsy": "slightly-drowsy",
                "Drowsy": "drowsy",
                "Very Drowsy - DANGER!": "danger",
            }.get(result["fatigue_level"], "alert")

            with status_placeholder.container():
                st.markdown(
                    f'<h2 class="{level_class}">{result["fatigue_level"]}</h2>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<h1 class="{level_class}">{result["fatigue_score"]:.1f}%</h1>',
                    unsafe_allow_html=True,
                )

            with metrics_placeholder.container():
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("EAR", f"{result['ear']:.3f}")
                    st.metric("PERCLOS", f"{result['perclos']:.1f}%")
                    st.metric("Blink Rate", f"{result['blink_rate']:.1f}/min")
                with mcol2:
                    st.metric("MAR", f"{result['mar']:.3f}")
                    st.metric("Yawning", "Ya ✅" if result["is_yawning"] else "Tidak ❌")
                    st.metric("Nodding", "Ya ✅" if result["is_nodding"] else "Tidak ❌")

            time.sleep(0.03)
        else:
            cap.release()
    else:
        st.info("Centang 'Mulai Deteksi' untuk mengaktifkan kamera dan memulai deteksi.")

# ===================== TENTANG =====================
elif menu == "Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    ### Deteksi Kelelahan Multi-Modal

    Aplikasi ini mengimplementasikan sistem deteksi kelelahan berbasis **computer vision** 
    dengan menggabungkan beberapa sinyal fisiologis:

    #### Metrik yang Digunakan:
    - **PERCLOS** (Percentage of Eye Closure): Persentase waktu mata tertutup dalam periode tertentu
    - **EAR** (Eye Aspect Ratio): Rasio aspek mata untuk mendeteksi kondisi mata terbuka/tertutup
    - **MAR** (Mouth Aspect Ratio): Rasio aspek mulut untuk mendeteksi menguap
    - **Head Pose**: Deteksi kepala menunduk sebagai indikasi kelelahan
    - **Blink Rate**: Frekuensi kedipan mata per menit

    #### Teknologi:
    - OpenCV untuk pemrosesan citra
    - Haar Cascade untuk deteksi wajah dan mata
    - Streamlit untuk antarmuka web
    """)

# ===================== RUN =====================
if __name__ == "__main__":
    pass
