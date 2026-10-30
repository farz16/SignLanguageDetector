import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import tensorflow as tf
import customtkinter as ctk
from PIL import Image, ImageTk


# =========================================================
# 1. APPEARANCE
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# =========================================================
# 2. VOICE SYSTEM
# =========================================================

last_spoken_sign = ""
last_speech_time = 0


def speak_text(text):

    def run():

        try:

            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                140
            )

            voices = engine.getProperty(
                "voices"
            )

            if len(voices) > 1:

                engine.setProperty(
                    "voice",
                    voices[1].id
                )

            engine.say(text)

            engine.runAndWait()

            engine.stop()

        except Exception:
            pass

    threading.Thread(
        target=run,
        daemon=True
    ).start()


# =========================================================
# 3. LOAD SIGN MODEL
# =========================================================

print("Loading sign language model...")

with open(
    "model.pkl",
    "rb"
) as f:

    sign_model = pickle.load(f)

print(
    "Sign classes:",
    sign_model.classes_
)


# =========================================================
# 4. LOAD EMOTION MODEL
# =========================================================

print("Loading emotion model...")

emotion_model = tf.keras.models.load_model(
    "emotion_model.keras"
)

EMOTION_LABELS = [
    "ANGRY",
    "HAPPY",
    "NEUTRAL",
    "SAD",
    "SURPRISE"
]

print("Emotion model loaded.")


# =========================================================
# 5. LOAD AGE MODEL
# =========================================================

print("Loading age model...")

age_model = tf.keras.models.load_model(
    "age_model.keras"
)

AGE_LABELS = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55+"
]

print("Age model loaded.")


# =========================================================
# 6. MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# =========================================================
# 7. CAMERA
# =========================================================

print("Opening Camera 0...")

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    raise SystemExit


print("Camera started successfully.")


# =========================================================
# 8. GLOBAL DATA
# =========================================================

current_sign = "NO HAND"
sign_accuracy = "0.00 %"

current_emotion = "NO FACE"
emotion_accuracy = "0.00 %"

current_age = "NO FACE"
age_accuracy = "0.00 %"

status_text = "Searching..."


# =========================================================
# 9. MAIN WINDOW
# =========================================================

app = ctk.CTk()

app.title(
    "AI Sign Language • Emotion • Age Analyzer"
)

app.geometry(
    "1450x850"
)

app.minsize(
    1200,
    700
)


# =========================================================
# 10. HEADER
# =========================================================

header = ctk.CTkFrame(
    app,
    height=80,
    corner_radius=0
)

header.pack(
    fill="x",
    padx=0,
    pady=0
)


title = ctk.CTkLabel(
    header,
    text="AI HUMAN ANALYSIS SYSTEM",
    font=ctk.CTkFont(
        size=27,
        weight="bold"
    )
)

title.pack(
    side="left",
    padx=30,
    pady=20
)


subtitle = ctk.CTkLabel(
    header,
    text="SIGN LANGUAGE  •  EMOTION  •  AGE",
    font=ctk.CTkFont(
        size=13
    ),
    text_color="gray70"
)

subtitle.pack(
    side="left",
    padx=10
)


system_status = ctk.CTkLabel(
    header,
    text="● SYSTEM ONLINE",
    text_color="#00E676",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

system_status.pack(
    side="right",
    padx=30
)


# =========================================================
# 11. MAIN AREA
# =========================================================

main_area = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

main_area.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=15
)


# =========================================================
# 12. CAMERA FRAME
# =========================================================

camera_frame = ctk.CTkFrame(
    main_area,
    corner_radius=15
)

camera_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


camera_title = ctk.CTkLabel(
    camera_frame,
    text="LIVE CAMERA",
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    )
)

camera_title.pack(
    anchor="w",
    padx=20,
    pady=(15, 5)
)


camera_label = ctk.CTkLabel(
    camera_frame,
    text="Starting camera..."
)

camera_label.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


# =========================================================
# 13. RIGHT DASHBOARD
# =========================================================

dashboard = ctk.CTkScrollableFrame(
    main_area,
    width=390,
    corner_radius=15
)

dashboard.pack(
    side="right",
    fill="y",
    padx=(10, 0)
)


dashboard_title = ctk.CTkLabel(
    dashboard,
    text="AI DETECTION",
    font=ctk.CTkFont(
        size=21,
        weight="bold"
    )
)

dashboard_title.pack(
    anchor="w",
    padx=15,
    pady=(15, 20)
)


# =========================================================
# 14. CARD FUNCTION
# =========================================================

def create_card(
    parent,
    title_text,
    value_text,
    confidence_text
):

    card = ctk.CTkFrame(
        parent,
        corner_radius=12
    )

    card.pack(
        fill="x",
        padx=5,
        pady=8
    )


    title_label = ctk.CTkLabel(
        card,
        text=title_text,
        font=ctk.CTkFont(
            size=12,
            weight="bold"
        ),
        text_color="gray65"
    )

    title_label.pack(
        anchor="w",
        padx=18,
        pady=(14, 2)
    )


    value_label = ctk.CTkLabel(
        card,
        text=value_text,
        font=ctk.CTkFont(
            size=25,
            weight="bold"
        )
    )

    value_label.pack(
        anchor="w",
        padx=18,
        pady=2
    )


    confidence_label = ctk.CTkLabel(
        card,
        text=confidence_text,
        font=ctk.CTkFont(
            size=11
        ),
        text_color="gray60"
    )

    confidence_label.pack(
        anchor="w",
        padx=18,
        pady=(0, 14)
    )


    return (
        value_label,
        confidence_label
    )


# =========================================================
# 15. SIGN CARD
# =========================================================

sign_value, sign_confidence = create_card(
    dashboard,
    "SIGN LANGUAGE",
    "NO HAND",
    "Confidence: 0.00 %"
)


# =========================================================
# 16. EMOTION CARD
# =========================================================

emotion_value, emotion_confidence = create_card(
    dashboard,
    "FACIAL EMOTION",
    "NO FACE",
    "Confidence: 0.00 %"
)


# =========================================================
# 17. AGE CARD
# =========================================================

age_value, age_confidence = create_card(
    dashboard,
    "AGE RANGE",
    "NO FACE",
    "Confidence: 0.00 %"
)


# =========================================================
# 18. STATUS CARD
# =========================================================

status_card = ctk.CTkFrame(
    dashboard,
    corner_radius=12
)

status_card.pack(
    fill="x",
    padx=5,
    pady=15
)


status_heading = ctk.CTkLabel(
    status_card,
    text="SYSTEM STATUS",
    font=ctk.CTkFont(
        size=12,
        weight="bold"
    ),
    text_color="gray65"
)

status_heading.pack(
    anchor="w",
    padx=18,
    pady=(14, 5)
)


status_value = ctk.CTkLabel(
    status_card,
    text="● Searching...",
    font=ctk.CTkFont(
        size=15,
        weight="bold"
    ),
    text_color="#FFB300"
)

status_value.pack(
    anchor="w",
    padx=18,
    pady=(0, 15)
)


# =========================================================
# 19. INSTRUCTIONS
# =========================================================

instruction_title = ctk.CTkLabel(
    dashboard,
    text="INSTRUCTIONS",
    font=ctk.CTkFont(
        size=13,
        weight="bold"
    ),
    text_color="gray65"
)

instruction_title.pack(
    anchor="w",
    padx=10,
    pady=(10, 5)
)


instructions = [
    "• Show one hand sign at a time.",
    "• Keep your face clearly visible.",
    "• Use good lighting.",
    "• Keep your hand inside the camera.",
    "• Voice output is enabled.",
    "• Press EXIT to close the system."
]


for instruction in instructions:

    label = ctk.CTkLabel(
        dashboard,
        text=instruction,
        font=ctk.CTkFont(
            size=12
        ),
        text_color="gray65",
        anchor="w"
    )

    label.pack(
        fill="x",
        padx=10,
        pady=3
    )


# =========================================================
# 20. FOOTER
# =========================================================

footer = ctk.CTkFrame(
    app,
    height=50,
    corner_radius=0
)

footer.pack(
    fill="x"
)


camera_status_footer = ctk.CTkLabel(
    footer,
    text="● Camera: ON",
    text_color="#00E676",
    font=ctk.CTkFont(
        size=12,
        weight="bold"
    )
)

camera_status_footer.pack(
    side="left",
    padx=25
)


ai_status_footer = ctk.CTkLabel(
    footer,
    text="AI Engine: ACTIVE",
    text_color="#00E676",
    font=ctk.CTkFont(
        size=12,
        weight="bold"
    )
)

ai_status_footer.pack(
    side="left",
    padx=20
)


voice_status_footer = ctk.CTkLabel(
    footer,
    text="Voice: ENABLED",
    font=ctk.CTkFont(
        size=12
    )
)

voice_status_footer.pack(
    side="left",
    padx=20
)


exit_button = ctk.CTkButton(
    footer,
    text="EXIT",
    width=100,
    height=32,
    command=lambda: close_application()
)

exit_button.pack(
    side="right",
    padx=20
)


# =========================================================
# 21. CLOSE FUNCTION
# =========================================================

def close_application():

    try:

        cap.release()

        hands.close()

        face_detector.close()

    except Exception:
        pass

    app.destroy()


# =========================================================
# 22. CAMERA PROCESSING
# =========================================================

def process_camera():

    global current_sign
    global sign_accuracy
    global current_emotion
    global emotion_accuracy
    global current_age
    global age_accuracy
    global status_text
    global last_spoken_sign
    global last_speech_time


    ret, frame = cap.read()


    if not ret:

        status_text = "Camera Error"

        app.after(
            30,
            process_camera
        )

        return


    # Mirror
    frame = cv2.flip(
        frame,
        1
    )


    # Resize
    frame = cv2.resize(
        frame,
        (900, 650)
    )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # HAND DETECTION
    # =====================================================

    hand_result = hands.process(
        rgb
    )


    if hand_result.multi_hand_landmarks:

        status_text = "Hand Detected"

        for hand_landmarks in hand_result.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            landmarks = []


            for point in hand_landmarks.landmark:

                landmarks.append(
                    point.x
                )

                landmarks.append(
                    point.y
                )


            if len(landmarks) == 42:

                prediction = sign_model.predict(
                    [landmarks]
                )


                current_sign = str(
                    prediction[0]
                ).upper()


                if hasattr(
                    sign_model,
                    "predict_proba"
                ):

                    probabilities = (
                        sign_model.predict_proba(
                            [landmarks]
                        )
                    )

                    max_probability = (
                        np.max(
                            probabilities
                        ) * 100
                    )

                    sign_accuracy = (
                        f"{max_probability:.2f} %"
                    )


                current_time = time.time()


                if (
                    current_sign != last_spoken_sign
                    or
                    (
                        current_time
                        - last_speech_time
                        > 2.5
                    )
                ):

                    speak_text(
                        current_sign
                    )

                    last_spoken_sign = (
                        current_sign
                    )

                    last_speech_time = (
                        current_time
                    )


    else:

        current_sign = "NO HAND"

        sign_accuracy = "0.00 %"

        last_spoken_sign = ""


    # =====================================================
    # FACE DETECTION
    # =====================================================

    face_result = face_detector.process(
        rgb
    )


    if face_result.detections:

        for detection in face_result.detections:

            h, w, _ = frame.shape


            bbox = (
                detection
                .location_data
                .relative_bounding_box
            )


            x = int(
                bbox.xmin * w
            )

            y = int(
                bbox.ymin * h
            )

            box_w = int(
                bbox.width * w
            )

            box_h = int(
                bbox.height * h
            )


            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )


            x2 = min(
                w,
                x + box_w
            )

            y2 = min(
                h,
                y + box_h
            )


            if x2 <= x or y2 <= y:

                continue


            # Face box
            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 255),
                2
            )


            face = frame[
                y:y2,
                x:x2
            ]


            if face.size == 0:

                continue


            # =================================================
            # FACE PREPROCESSING
            # =================================================

            gray_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )


            gray_face = cv2.resize(
                gray_face,
                (48, 48)
            )


            gray_face = (
                gray_face.astype(
                    np.float32
                ) / 255.0
            )


            face_input = np.expand_dims(
                gray_face,
                axis=(0, -1)
            )


            # =================================================
            # EMOTION
            # =================================================

            emotion_prediction = (
                emotion_model.predict(
                    face_input,
                    verbose=0
                )[0]
            )


            emotion_index = np.argmax(
                emotion_prediction
            )


            current_emotion = (
                EMOTION_LABELS[
                    emotion_index
                ]
            )


            emotion_probability = (
                emotion_prediction[
                    emotion_index
                ] * 100
            )


            emotion_accuracy = (
                f"{emotion_probability:.2f} %"
            )


            # =================================================
            # AGE
            # =================================================

            age_prediction = (
                age_model.predict(
                    face_input,
                    verbose=0
                )[0]
            )


            age_index = np.argmax(
                age_prediction
            )


            current_age = (
                AGE_LABELS[
                    age_index
                ]
            )


            age_probability = (
                age_prediction[
                    age_index
                ] * 100
            )


            age_accuracy = (
                f"{age_probability:.2f} %"
            )


            break


    else:

        current_emotion = "NO FACE"

        emotion_accuracy = "0.00 %"

        current_age = "NO FACE"

        age_accuracy = "0.00 %"


    # =====================================================
    # CAMERA IMAGE
    # =====================================================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(
        frame_rgb
    )


    # Fit camera image
    camera_width = camera_label.winfo_width()
    camera_height = camera_label.winfo_height()


    if (
        camera_width > 100
        and
        camera_height > 100
    ):

        image.thumbnail(
            (
                camera_width - 20,
                camera_height - 20
            ),
            Image.Resampling.LANCZOS
        )


    photo = ImageTk.PhotoImage(
        image
    )


    camera_label.configure(
        image=photo,
        text=""
    )


    camera_label.image = photo


    # =====================================================
    # UPDATE DASHBOARD
    # =====================================================

    sign_value.configure(
        text=current_sign
    )


    sign_confidence.configure(
        text=(
            "Confidence: "
            + sign_accuracy
        )
    )


    emotion_value.configure(
        text=current_emotion
    )


    emotion_confidence.configure(
        text=(
            "Confidence: "
            + emotion_accuracy
        )
    )


    age_value.configure(
        text=current_age
    )


    age_confidence.configure(
        text=(
            "Confidence: "
            + age_accuracy
        )
    )


    status_value.configure(
        text="● " + status_text
    )


    # =====================================================
    # NEXT FRAME
    # =====================================================

    if app.winfo_exists():

        app.after(
            10,
            process_camera
        )


# =========================================================
# 23. START
# =========================================================

app.protocol(
    "WM_DELETE_WINDOW",
    close_application
)


app.after(
    100,
    process_camera
)


print(
    "Professional AI Dashboard started."
)


app.mainloop()


# =========================================================
# 24. FINAL CLEANUP
# =========================================================

try:

    cap.release()

    hands.close()

    face_detector.close()

except Exception:
    pass