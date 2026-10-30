# =========================================================
# Project: AI Smart Sign Language Vision System Pro
# Author: Girase Farjana Babu
# Degree: Master of Computer Applications (MCA)
# Copyright (c) 2026 Girase Farjana Babu. All Rights Reserved.
# =========================================================

import threading
import time
import datetime
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import tensorflow as tf
import customtkinter as ctk
from PIL import Image

# =========================================================
# 1. APPEARANCE & THEME SETTINGS
# =========================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# =========================================================
# 2. VOICE SYSTEM (NON-BLOCKING THREADED)
# =========================================================
last_spoken_sign = ""
last_speech_time = 0
voice_enabled = True
draw_landmarks = True
auto_brightness_enabled = True

def speak_text(text):
    global voice_enabled
    if not voice_enabled or not text or text in ["NO HAND", "NO FACE"]:
        return

    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 140)
            voices = engine.getProperty("voices")
            if len(voices) > 1:
                engine.setProperty("voice", voices[1].id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()

# =========================================================
# 3. LOAD MODELS
# =========================================================
print("Loading sign language model...")
with open("model.pkl", "rb") as f:
    sign_model = pickle.load(f)

print("Loading emotion model...")
emotion_model = tf.keras.models.load_model("emotion_model.keras")
EMOTION_LABELS = ["ANGRY", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]

print("Loading age model...")
age_model = tf.keras.models.load_model("age_model.keras")
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Warm-up models
try:
    dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
    _ = emotion_model(dummy_input, training=False)
    _ = age_model(dummy_input, training=False)
except Exception:
    pass

# =========================================================
# 4. MEDIAPIPE INITIALIZATION (OPTIMIZED FOR HIGH FPS)
# =========================================================
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# =========================================================
# 5. ROBUST CAMERA INITIALIZATION (NO-LAG ZERO BUFFER)
# =========================================================
def get_working_camera(target_index=1):
    for backend in [None, cv2.CAP_DSHOW]:
        c = cv2.VideoCapture(target_index) if backend is None else cv2.VideoCapture(target_index, backend)
        if c.isOpened():
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(5):
                ret, frame = c.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✓ Connected to Camera {target_index} successfully!")
                    return c
            c.release()

    print("⚠️ Camera 1 stream not ready. Switching to Camera 0 (Laptop Cam)...")
    c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return c

cap = get_working_camera(target_index=1)

# Instant low-light lookup table
gamma_lut = np.array([((i / 255.0) ** (1.0 / 1.35)) * 255 for i in range(256)]).astype("uint8")

# =========================================================
# 6. GLOBAL STATE VARIABLES
# =========================================================
current_sign = "NO HAND"
sign_accuracy = "0.00 %"
current_emotion = "NO FACE"
emotion_accuracy = "0.00 %"
current_age = "NO FACE"
age_accuracy = "0.00 %"

hand_detected_str = "Not Detected"
face_detected_str = "Not Detected"

cached_face_box = None
frame_counter = 0
prev_time = time.time()
fps_val = 30
emotion_probs_dict = {label: 0.0 for label in EMOTION_LABELS}

# =========================================================
# 7. FIXED ALL 4-CORNER CAMERA BRACKETS [ ┌ ┐ └ ┘ ]
# =========================================================
def draw_camera_focus_bbox(img, bbox, color=(255, 255, 255)):
    x, y, x2, y2 = bbox
    w_box = x2 - x
    h_box = y2 - y
    arm = max(16, int(min(w_box, h_box) * 0.22))
    t = 2

    # 1. Top-Left Corner ┌
    cv2.line(img, (x, y), (x + arm, y), color, t)
    cv2.line(img, (x, y), (x, y + arm), color, t)

    # 2. Top-Right Corner ┐
    cv2.line(img, (x2, y), (x2 - arm, y), color, t)
    cv2.line(img, (x2, y), (x2, y + arm), color, t)

    # 3. Bottom-Left Corner └
    cv2.line(img, (x, y2), (x + arm, y2), color, t)
    cv2.line(img, (x, y2), (x, y2 - arm), color, t)

    # 4. Bottom-Right Corner ┘
    cv2.line(img, (x2, y2), (x2 - arm, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - arm), color, t)

def draw_live_badge(img):
    cv2.rectangle(img, (18, 16), (92, 44), (15, 18, 26), -1)
    cv2.rectangle(img, (18, 16), (92, 44), (55, 60, 75), 1)

    is_blink_on = int(time.time() * 2) % 2 == 0
    dot_color = (0, 0, 255) if is_blink_on else (0, 0, 160)
    cv2.circle(img, (32, 30), 5, dot_color, -1)
    cv2.putText(img, "LIVE", (44, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

# =========================================================
# 8. MAIN DASHBOARD APPLICATION
# =========================================================
class AISmartVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI SMART SIGN LANGUAGE SYSTEM - PRO EDITION")
        self.geometry("1320x760")
        self.configure(fg_color="#0B0D14")

        # Control States
        self.is_running = True
        self.current_frame = None
        self.current_mode = "Live Detection"
        self.nav_buttons = {}
        self.emotion_bars = {}

        # Anti-Jitter Image Sizing Cache
        self.prev_cam_w = 640
        self.prev_cam_h = 480
        self.cam_disp_w = 640
        self.cam_disp_h = 480

        # Fixed Column Weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=210)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=290)

        self.setup_sidebar()
        self.setup_center_panel()
        self.setup_right_panel()

        # Start Camera Loop
        self.process_camera()

    # --- 1. SIDEBAR ---
    def setup_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color="#131622", corner_radius=0, width=210)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="AI VISION PRO", font=ctk.CTkFont(size=20, weight="bold"), text_color="#A5B4FC").pack(pady=20)

        nav_items = ["Live Detection", "Sign Language", "Emotion", "Age Estimation", "Settings", "About", "Instructions"]
        for item in nav_items:
            btn = ctk.CTkButton(
                sidebar, 
                text=item, 
                fg_color="#5B46E8" if item == "Live Detection" else "transparent",
                hover_color="#3B2DB0", 
                anchor="w", 
                height=40,
                command=lambda name=item: self.switch_mode(name)
            )
            btn.pack(fill="x", padx=10, pady=5)
            self.nav_buttons[item] = btn

        # Telemetry Box
        status_box = ctk.CTkFrame(sidebar, fg_color="#181C2B", corner_radius=10)
        status_box.pack(side="bottom", fill="x", padx=10, pady=20)
        
        ctk.CTkLabel(status_box, text="● System Status", text_color="#00E676", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 2))
        self.lbl_fps = ctk.CTkLabel(status_box, text="⚡ 30 FPS | 15ms Latency", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38BDF8")
        self.lbl_fps.pack(anchor="w", padx=10, pady=(0, 2))

        self.lbl_system_running = ctk.CTkLabel(status_box, text="Hardware Boost: Active", font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_system_running.pack(anchor="w", padx=10, pady=(0, 10))

    # --- 2. CENTER PANEL ---
    def setup_center_panel(self):
        self.center = ctk.CTkFrame(self, fg_color="transparent")
        self.center.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

        # Header Titles (Top)
        self.header_title = ctk.CTkLabel(self.center, text="AI SMART SIGN LANGUAGE SYSTEM", font=ctk.CTkFont(size=22, weight="bold"))
        self.header_title.pack(side="top", fill="x", pady=(5, 0))
        
        self.header_sub = ctk.CTkLabel(self.center, text="Active Mode: LIVE DETECTION | Real-time AI Pipeline", font=ctk.CTkFont(size=12), text_color="#818CF8")
        self.header_sub.pack(side="top", fill="x", pady=(0, 8))

        # Bottom Control Buttons Bar
        self.btn_frame = ctk.CTkFrame(self.center, fg_color="transparent")
        self.btn_frame.pack(side="bottom", fill="x", pady=5)

        ctk.CTkButton(self.btn_frame, text="Start", fg_color="#5B46E8", hover_color="#4532C7", width=110, height=40, command=self.start_detection).pack(side="left", expand=True, padx=4)
        ctk.CTkButton(self.btn_frame, text="Stop", fg_color="#D93838", hover_color="#B22B2B", width=110, height=40, command=self.stop_detection).pack(side="left", expand=True, padx=4)
        ctk.CTkButton(self.btn_frame, text="Screenshot", fg_color="#007ACC", hover_color="#005A9E", width=110, height=40, command=self.take_screenshot).pack(side="left", expand=True, padx=4)
        
        # Voice Mute/Unmute Switch
        self.voice_switch = ctk.CTkSwitch(self.btn_frame, text="🔊 Voice", font=ctk.CTkFont(size=12, weight="bold"), command=self.toggle_voice)
        self.voice_switch.select()
        self.voice_switch.pack(side="left", expand=True, padx=6)

        # Bottom Status Message Bar
        self.status_bar = ctk.CTkFrame(self.center, fg_color="#131622", height=35)
        self.status_bar.pack(side="bottom", fill="x", pady=5)
        self.lbl_status_msg = ctk.CTkLabel(self.status_bar, text="✓  All systems running smoothly", text_color="#00E676")
        self.lbl_status_msg.pack(pady=5)

        # Center Display Slots
        self.display_container = ctk.CTkFrame(self.center, fg_color="transparent")
        self.display_container.pack(side="top", fill="both", expand=True, pady=5)

        self.camera_label = ctk.CTkLabel(self.display_container, text="", fg_color="#181C2B", corner_radius=12)
        self.camera_label.pack(fill="both", expand=True)

        self.info_panel = ctk.CTkFrame(self.display_container, fg_color="#181C2B", corner_radius=12)

    # --- 3. RIGHT PANEL ---
    def setup_right_panel(self):
        right = ctk.CTkFrame(self, fg_color="transparent", width=290)
        right.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        right.grid_propagate(False)

        # 1. Sign Card
        card1 = ctk.CTkFrame(right, fg_color="#131622", corner_radius=12)
        card1.pack(fill="x", pady=4)
        ctk.CTkLabel(card1, text="SIGN LANGUAGE RESULT", font=ctk.CTkFont(size=11, weight="bold"), text_color="#A5B4FC").pack(anchor="w", padx=15, pady=(8, 2))
        self.sign_val = ctk.CTkLabel(card1, text="NO HAND", font=ctk.CTkFont(size=22, weight="bold"), text_color="#C084FC")
        self.sign_val.pack(anchor="w", padx=15)
        self.sign_conf = ctk.CTkLabel(card1, text="Confidence: 0.00%", font=ctk.CTkFont(size=12), text_color="gray")
        self.sign_conf.pack(anchor="w", padx=15, pady=(0, 8))

        # 2. Emotion Card + Live Probability Meters
        card2 = ctk.CTkFrame(right, fg_color="#131622", corner_radius=12)
        card2.pack(fill="x", pady=4)
        ctk.CTkLabel(card2, text="EMOTION RESULT", font=ctk.CTkFont(size=11, weight="bold"), text_color="#A5B4FC").pack(anchor="w", padx=15, pady=(8, 2))
        self.emotion_val = ctk.CTkLabel(card2, text="NO FACE", font=ctk.CTkFont(size=22, weight="bold"), text_color="#10B981")
        self.emotion_val.pack(anchor="w", padx=15)
        self.emotion_conf = ctk.CTkLabel(card2, text="Confidence: 0.00%", font=ctk.CTkFont(size=12), text_color="gray")
        self.emotion_conf.pack(anchor="w", padx=15, pady=(0, 4))

        # Live Mini Emotion Progress Bars
        meters_frame = ctk.CTkFrame(card2, fg_color="#181C2B", corner_radius=8)
        meters_frame.pack(fill="x", padx=12, pady=(0, 8))

        for emo in EMOTION_LABELS:
            row = ctk.CTkFrame(meters_frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            
            display_name = emo.capitalize()
            lbl = ctk.CTkLabel(
                row, 
                text=display_name, 
                font=ctk.CTkFont(size=11, weight="bold"), 
                text_color="#94A3B8", 
                width=68, 
                anchor="w"
            )
            lbl.pack(side="left")
            
            pbar = ctk.CTkProgressBar(row, height=7, corner_radius=3, fg_color="#0F111A", progress_color="#10B981")
            pbar.set(0.0)
            pbar.pack(side="left", fill="x", expand=True, padx=(2, 6))
            
            pct_lbl = ctk.CTkLabel(row, text="0%", font=ctk.CTkFont(size=11), text_color="gray", width=34, anchor="e")
            pct_lbl.pack(side="left")
            self.emotion_bars[emo] = (pbar, pct_lbl)

        # 3. Age Card
        card3 = ctk.CTkFrame(right, fg_color="#131622", corner_radius=12)
        card3.pack(fill="x", pady=4)
        ctk.CTkLabel(card3, text="AGE ESTIMATION RESULT", font=ctk.CTkFont(size=11, weight="bold"), text_color="#A5B4FC").pack(anchor="w", padx=15, pady=(8, 2))
        self.age_val = ctk.CTkLabel(card3, text="NO FACE", font=ctk.CTkFont(size=22, weight="bold"), text_color="#F59E0B")
        self.age_val.pack(anchor="w", padx=15)
        self.age_conf = ctk.CTkLabel(card3, text="Confidence: 0.00%", font=ctk.CTkFont(size=12), text_color="gray")
        self.age_conf.pack(anchor="w", padx=15, pady=(0, 8))

        # 4. Info Box
        info = ctk.CTkFrame(right, fg_color="#131622", corner_radius=12)
        info.pack(fill="both", expand=True, pady=4)
        ctk.CTkLabel(info, text="DETECTION TELEMETRY", font=ctk.CTkFont(size=11, weight="bold"), text_color="#A5B4FC").pack(anchor="w", padx=15, pady=(8, 4))

        self.lbl_hand_status = ctk.CTkLabel(info, text="• Hand: Not Detected", font=ctk.CTkFont(size=12))
        self.lbl_hand_status.pack(anchor="w", padx=15, pady=1)

        self.lbl_face_status = ctk.CTkLabel(info, text="• Face: Not Detected", font=ctk.CTkFont(size=12))
        self.lbl_face_status.pack(anchor="w", padx=15, pady=1)

        ctk.CTkLabel(info, text="• Sign Engine: Active (RF)", font=ctk.CTkFont(size=12), text_color="#00E676").pack(anchor="w", padx=15, pady=1)
        ctk.CTkLabel(info, text="• Emotion Engine: Active (CNN)", font=ctk.CTkFont(size=12), text_color="#00E676").pack(anchor="w", padx=15, pady=1)
        ctk.CTkLabel(info, text="• Age Engine: Active (CNN)", font=ctk.CTkFont(size=12), text_color="#00E676").pack(anchor="w", padx=15, pady=1)

    # --- SIDEBAR NAVIGATION SWITCH LOGIC ---
    def switch_mode(self, mode_name):
        self.current_mode = mode_name

        for name, btn in self.nav_buttons.items():
            btn.configure(fg_color="#5B46E8" if name == mode_name else "transparent")

        self.header_sub.configure(text=f"Active Mode: {mode_name.upper()} | Real-time AI Pipeline")

        if mode_name in ["Live Detection", "Sign Language", "Emotion", "Age Estimation"]:
            self.info_panel.pack_forget()
            self.camera_label.pack(fill="both", expand=True)
        elif mode_name == "Settings":
            self.camera_label.pack_forget()
            self.render_settings_view()
        elif mode_name == "About":
            self.camera_label.pack_forget()
            self.render_about_view()
        elif mode_name == "Instructions":
            self.camera_label.pack_forget()
            self.render_instructions_view()

    def render_settings_view(self):
        for widget in self.info_panel.winfo_children():
            widget.destroy()
        self.info_panel.pack(fill="both", expand=True)

        ctk.CTkLabel(self.info_panel, text="SYSTEM SETTINGS", font=ctk.CTkFont(size=18, weight="bold"), text_color="#A5B4FC").pack(pady=20)
        
        switch_speech = ctk.CTkSwitch(self.info_panel, text="Enable Voice Feedback (TTS)", command=self.toggle_voice_from_settings)
        if voice_enabled:
            switch_speech.select()
        else:
            switch_speech.deselect()
        switch_speech.pack(pady=10)
        self.settings_speech_switch = switch_speech

        switch_landmarks = ctk.CTkSwitch(self.info_panel, text="Draw MediaPipe Skeleton Overlays", command=self.toggle_landmarks)
        if draw_landmarks:
            switch_landmarks.select()
        else:
            switch_landmarks.deselect()
        switch_landmarks.pack(pady=10)
        self.settings_landmarks_switch = switch_landmarks

        switch_light = ctk.CTkSwitch(self.info_panel, text="Auto Low-Light / Contrast Booster", command=self.toggle_brightness)
        if auto_brightness_enabled:
            switch_light.select()
        else:
            switch_light.deselect()
        switch_light.pack(pady=10)
        self.settings_light_switch = switch_light

    def render_about_view(self):
        for widget in self.info_panel.winfo_children():
            widget.destroy()
        self.info_panel.pack(fill="both", expand=True)

        ctk.CTkLabel(self.info_panel, text="ABOUT PROJECT", font=ctk.CTkFont(size=20, weight="bold"), text_color="#A5B4FC").pack(pady=(30, 8))
        ctk.CTkLabel(self.info_panel, text="AI Smart Sign Language System Pro", font=ctk.CTkFont(size=15, weight="bold"), text_color="#38BDF8").pack(pady=(0, 15))

        desc = (
            "Integrated Real-time Computer Vision Pipeline\n\n"
            "• Sign Language Detection (MediaPipe + Scikit-Learn)\n"
            "• Facial Emotion Recognition (Keras CNN)\n"
            "• Age Estimation Model (Deep Learning)\n\n"
            "Developed for MCA Project Presentation.\n\n"
            "© 2026 Girase Farjana Babu. All Rights Reserved."
        )
        ctk.CTkLabel(self.info_panel, text=desc, font=ctk.CTkFont(size=13), justify="center", text_color="#CBD5E1").pack(pady=15)

    def render_instructions_view(self):
        for widget in self.info_panel.winfo_children():
            widget.destroy()
        self.info_panel.pack(fill="both", expand=True)

        ctk.CTkLabel(self.info_panel, text="📖 HOW TO USE / INSTRUCTIONS", font=ctk.CTkFont(size=18, weight="bold"), text_color="#A5B4FC").pack(pady=(15, 8))

        scroll_box = ctk.CTkScrollableFrame(self.info_panel, fg_color="transparent")
        scroll_box.pack(fill="both", expand=True, padx=15, pady=5)

        instructions_data = [
            {"title": "1. Live Detection Mode", "color": "#8B5CF6", "desc": "Runs all AI models simultaneously. It tracks your hand gestures for Sign Language while estimating Facial Emotion and Age in real-time."},
            {"title": "2. Sign Language Recognition", "color": "#3B82F6", "desc": "Position your hand clearly in front of the webcam. The skeleton tracks 21 hand landmarks and announces recognized signs with Voice TTS."},
            {"title": "3. Facial Emotion & Age Estimation", "color": "#10B981", "desc": "Target brackets lock onto your face. Real-time probability meters show live confidence across all 5 emotional states."},
            {"title": "4. Audio / Voice Feedback", "color": "#F59E0B", "desc": "Toggle the '🔊 Voice' switch at the bottom panel anytime to mute or un-mute spoken audio predictions."},
            {"title": "5. Start, Stop & Screenshot Controls", "color": "#007ACC", "desc": "• Start: Resumes detection.\n• Stop: Pauses processing.\n• Screenshot: Saves camera frame snapshot locally."},
            {"title": "6. Best Accuracy Tips", "color": "#EC4899", "desc": "• Sit 1.5 to 3 feet from the webcam with good lighting.\n• Auto low-light booster handles dimmer environments automatically."}
        ]

        for item in instructions_data:
            card = ctk.CTkFrame(scroll_box, fg_color="#131622", corner_radius=10)
            card.pack(fill="x", pady=5, padx=5)

            ctk.CTkLabel(card, text=item["title"], font=ctk.CTkFont(size=13, weight="bold"), text_color=item["color"]).pack(anchor="w", padx=15, pady=(8, 2))
            ctk.CTkLabel(card, text=item["desc"], font=ctk.CTkFont(size=12), text_color="#CBD5E1", wraplength=620, justify="left").pack(anchor="w", padx=15, pady=(0, 8))

    # --- BUTTON & SETTING TOGGLES ---
    def toggle_voice(self):
        global voice_enabled
        voice_enabled = bool(self.voice_switch.get())
        status = "Voice Audio Enabled" if voice_enabled else "Voice Audio Muted"
        self.lbl_status_msg.configure(text=f"🔊 {status}", text_color="#38BDF8" if voice_enabled else "#FFB300")

    def toggle_voice_from_settings(self):
        global voice_enabled
        voice_enabled = bool(self.settings_speech_switch.get())
        if voice_enabled:
            self.voice_switch.select()
        else:
            self.voice_switch.deselect()
        self.toggle_voice()

    def toggle_landmarks(self):
        global draw_landmarks
        draw_landmarks = bool(self.settings_landmarks_switch.get())

    def toggle_brightness(self):
        global auto_brightness_enabled
        auto_brightness_enabled = bool(self.settings_light_switch.get())

    def start_detection(self):
        self.is_running = True
        self.lbl_status_msg.configure(text="✓  All systems are running smoothly", text_color="#00E676")
        self.lbl_system_running.configure(text="Hardware Boost: Active", text_color="gray")

    def stop_detection(self):
        self.is_running = False
        self.lbl_status_msg.configure(text="⏸ Detection Paused", text_color="#FFB300")
        self.lbl_system_running.configure(text="System Paused", text_color="#FFB300")

    def take_screenshot(self):
        if self.current_frame is not None:
            filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.lbl_status_msg.configure(text=f"📷 Screenshot Saved as {filename}", text_color="#007ACC")

    def close_application(self):
        try:
            cap.release()
            hands.close()
            face_detector.close()
        except Exception:
            pass
        self.destroy()

    # --- ULTRA OPTIMIZED REAL-TIME CAMERA LOOP ---
    def process_camera(self):
        global current_sign, sign_accuracy, current_emotion, emotion_accuracy
        global current_age, age_accuracy, last_spoken_sign, last_speech_time
        global hand_detected_str, face_detected_str, cached_face_box, frame_counter
        global prev_time, fps_val, emotion_probs_dict

        if self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frame_counter += 1
                
                # Instant downscale for high-res mobile cams (1080p -> 640p)
                h_orig, w_orig = frame.shape[:2]
                if w_orig > 640:
                    new_h = int(640 * h_orig / w_orig)
                    frame = cv2.resize(frame, (640, new_h))

                frame = cv2.flip(frame, 1)

                # 1. Instant Auto Low-Light Boost
                if auto_brightness_enabled:
                    mean_lum = np.mean(frame[::10, ::10, 0])
                    if mean_lum < 70:
                        frame = cv2.LUT(frame, gamma_lut)

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate Live FPS
                cur_t = time.time()
                dt = cur_t - prev_time
                prev_time = cur_t
                if dt > 0:
                    fps_val = int(0.9 * fps_val + 0.1 * (1.0 / dt))

                # --- 2. DRAW "● LIVE" CAMERA BADGE ---
                draw_live_badge(frame)

                # --- 3. HAND GESTURE DETECTION ---
                if self.current_mode in ["Live Detection", "Sign Language"]:
                    hand_result = hands.process(rgb)
                    if hand_result.multi_hand_landmarks:
                        hand_detected_str = "Detected"
                        for hand_landmarks in hand_result.multi_hand_landmarks:
                            if draw_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, 
                                    hand_landmarks, 
                                    mp_hands.HAND_CONNECTIONS
                                )
                            
                            landmarks = []
                            for point in hand_landmarks.landmark:
                                landmarks.extend([point.x, point.y])

                            if len(landmarks) == 42:
                                prediction = sign_model.predict([landmarks])
                                current_sign = str(prediction[0]).upper()

                                if hasattr(sign_model, "predict_proba"):
                                    probabilities = sign_model.predict_proba([landmarks])
                                    max_probability = np.max(probabilities) * 100
                                    sign_accuracy = f"{max_probability:.2f} %"

                                current_time = time.time()
                                if current_sign != last_spoken_sign or (current_time - last_speech_time > 2.5):
                                    speak_text(current_sign)
                                    last_spoken_sign = current_sign
                                    last_speech_time = current_time
                    else:
                        hand_detected_str = "Not Detected"
                        current_sign = "NO HAND"
                        sign_accuracy = "0.00 %"
                        last_spoken_sign = ""

                # --- 4. FACE & FAST DL INFERENCE ---
                if self.current_mode in ["Live Detection", "Emotion", "Age Estimation"]:
                    if frame_counter % 2 == 0:
                        face_result = face_detector.process(rgb)
                        if face_result.detections:
                            face_detected_str = "Detected"
                            bbox = face_result.detections[0].location_data.relative_bounding_box
                            x, y = max(0, int(bbox.xmin * w)), max(0, int(bbox.ymin * h))
                            x2, y2 = min(w, x + int(bbox.width * w)), min(h, y + int(bbox.height * h))
                            cached_face_box = (x, y, x2, y2)
                        else:
                            cached_face_box = None
                            face_detected_str = "Not Detected"
                            current_emotion = "NO FACE"
                            emotion_accuracy = "0.00 %"
                            current_age = "NO FACE"
                            age_accuracy = "0.00 %"
                            for emo in EMOTION_LABELS:
                                emotion_probs_dict[emo] = 0.0

                    if cached_face_box is not None:
                        x, y, x2, y2 = cached_face_box
                        
                        # Real Camera Style Crisp White Bracket [ ]
                        draw_camera_focus_bbox(frame, (x, y, x2, y2), color=(255, 255, 255))

                        # Deep Learning Inference every 4th frame
                        if frame_counter % 4 == 0 and x2 > x and y2 > y:
                            face = frame[y:y2, x:x2]
                            if face.size > 0:
                                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                gray_face = cv2.resize(gray_face, (48, 48)).astype(np.float32) / 255.0
                                face_input = np.expand_dims(gray_face, axis=(0, -1))

                                # Fast direct tensor execution
                                emotion_preds = emotion_model(face_input, training=False).numpy()[0]
                                emotion_idx = np.argmax(emotion_preds)
                                current_emotion = EMOTION_LABELS[emotion_idx]
                                emotion_accuracy = f"{(emotion_preds[emotion_idx] * 100):.2f} %"

                                for idx, emo in enumerate(EMOTION_LABELS):
                                    emotion_probs_dict[emo] = float(emotion_preds[idx])

                                age_preds = age_model(face_input, training=False).numpy()[0]
                                age_idx = np.argmax(age_preds)
                                current_age = AGE_LABELS[age_idx]
                                age_accuracy = f"{(age_preds[age_idx] * 100):.2f} %"

                self.current_frame = frame.copy()

                # --- 5. RENDER CAMERA WITH ANTI-JITTER STABILITY ---
                if self.current_mode in ["Live Detection", "Sign Language", "Emotion", "Age Estimation"]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)

                    cur_w = self.camera_label.winfo_width()
                    cur_h = self.camera_label.winfo_height()

                    if abs(cur_w - self.prev_cam_w) > 6 or abs(cur_h - self.prev_cam_h) > 6:
                        self.cam_disp_w = max(100, cur_w - 6)
                        self.cam_disp_h = max(100, cur_h - 6)
                        self.prev_cam_w = cur_w
                        self.prev_cam_h = cur_h

                    ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=(self.cam_disp_w, self.cam_disp_h))
                    self.camera_label.configure(image=ctk_img, text="")

                # --- 6. UPDATE UI CARDS & TELEMETRY ---
                self.sign_val.configure(text=current_sign)
                self.sign_conf.configure(text=f"Confidence: {sign_accuracy}")

                self.emotion_val.configure(text=current_emotion)
                self.emotion_conf.configure(text=f"Confidence: {emotion_accuracy}")

                # Update live mini emotion meters
                for emo, (pbar, pct_lbl) in self.emotion_bars.items():
                    val = emotion_probs_dict.get(emo, 0.0)
                    pbar.set(val)
                    pct_lbl.configure(text=f"{int(val*100)}%")

                self.age_val.configure(text=current_age)
                self.age_conf.configure(text=f"Confidence: {age_accuracy}")

                self.lbl_hand_status.configure(text=f"• Hand: {hand_detected_str}")
                self.lbl_face_status.configure(text=f"• Face: {face_detected_str}")
                self.lbl_fps.configure(text=f"⚡ {fps_val} FPS | Latency: {int(dt*1000)}ms")

        if self.winfo_exists():
            self.after(1, self.process_camera)

# =========================================================
# 9. APPLICATION ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = AISmartVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.close_application)
    app.mainloop()