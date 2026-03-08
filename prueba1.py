# --- 0. Imports ---
import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# MediaPipe (modern)
import mediapipe as mp

# --- 1. MediaPipe helpers (modern API) ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    """Converts BGR->RGB, runs model.process, returns BGR image and results"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def draw_styled_landmarks(image, results):
    """Draws face mesh, pose, left and right hands if present"""
    # Face mesh (use FACEMESH_TESSELATION for dense mesh)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_face.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )

    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

# --- 2. Keypoint extraction (robusta) ---
def extract_keypoints(results):
    """Return a 1D array concatenating pose(33*4), face(468*3), left hand(21*3), right hand(21*3)"""
    # pose: 33 landmarks each [x,y,z,visibility] => 33*4 = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face: 468 landmarks each [x,y,z] => 468*3 = 1404
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # left hand: 21 landmarks each [x,y,z] => 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # right hand
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# --- 3. Data folders and params ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30
start_folder = 1   # we'll create sequences starting from 1 (más intuitivo)

# Create folders if missing
os.makedirs(DATA_PATH, exist_ok=True)
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    # create sequence folders inside each action if not exist
    for seq in range(start_folder, start_folder + no_sequences):
        seq_path = os.path.join(action_path, str(seq))
        os.makedirs(seq_path, exist_ok=True)

# --- 4. Collect data (run this cell interactively) ---
# Nota: Ejecuta la recolección solo cuando estés listo: abre cámara y pulsa 'q' para salir.
def collect_data():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(start_folder, start_folder + no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("No frame captured, check camera.")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Wait / messaging only at first frame
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    # Save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()

# Para ejecutar recolección: llama collect_data()
# collect_data()

# --- 5. Preprocess (load sequences into X,y) ---
def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        # list sequence folders (only directories with numeric names)
        seq_dirs = [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d)) and d.isdigit()]
        for seq in sorted(seq_dirs, key=lambda x: int(x)):
            window = []
            for frame_num in range(sequence_length):
                npy_file = os.path.join(action_dir, seq, f"{frame_num}.npy")
                if os.path.exists(npy_file):
                    res = np.load(npy_file)
                else:
                    # if missing frame, use zeros
                    res = np.zeros(33*4 + 468*3 + 21*3 + 21*3)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y

# Ejecuta load_data() después de recolectar datos
# X, y = load_data()

# --- 6. Train-test split example ---
# Si ya tienes X,y:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 7. Model (LSTM) ---
def build_model(input_shape, actions_count):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions_count, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Ejemplo de entrenamiento (descomenta y ejecuta cuando tengas X_train, y_train)
# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)
# input_shape = (sequence_length, X_train.shape[2])
# model = build_model(input_shape, actions.shape[0])
# model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])  # epochs ajustable

# --- 8. Guardar y cargar modelo correctamente ---
# model.save('action.h5')
# from tensorflow.keras.models import load_model
# model = load_model('action.h5')

# --- 9. Evaluación ejemplo ---
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat_labels = np.argmax(yhat, axis=1).tolist()
# print(multilabel_confusion_matrix(ytrue, yhat_labels))
# print("Accuracy:", accuracy_score(ytrue, yhat_labels))

# --- 10. Real-time prediction (usa modelo cargado) ---
def prob_viz(res_probs, actions_list, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res_probs):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions_list[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def realtime_predict(model, threshold=0.5):
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    sequence = []
    sentence = []
    predictions = []

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                # smoothing + threshold logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Para usar realtime_predict: primero carga el modelo (model = load_model('action.h5')) y llama:
# realtime_predict(model)
