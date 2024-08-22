import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
from PIL import Image

# Load the model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Load emojis
emojis = {
    "neutral":cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\neutral.png", cv2.IMREAD_UNCHANGED),
    "happy": cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\happy.png", cv2.IMREAD_UNCHANGED),
    "surprised":cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\surprised.png", cv2.IMREAD_UNCHANGED),
    "hello":cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\hello.png", cv2.IMREAD_UNCHANGED),
    "sad":cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\sad.png", cv2.IMREAD_UNCHANGED),
    "angry":cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\angry.png", cv2.IMREAD_UNCHANGED),
    "peace":cv2.imread("D:\\FYP\\New one (simple neural network)\\liveEmoji-main\\liveEmoji-main\\emojis\\angry.png", cv2.IMREAD_UNCHANGED)
    # Add more emojis as needed
}

# Define function to overlay emoji on the frame
def overlay_emoji(frame, emoji, x, y, emoji_scale=0.2):
    h, w, _ = emoji.shape
    roi = frame[y:y+h, x:x+w]

    # Resize the emoji
    emoji_resized = cv2.resize(emoji, None, fx=emoji_scale, fy=emoji_scale)

    # Overlay emoji on the frame
    emoji_h, emoji_w, _ = emoji_resized.shape
    alpha_s = emoji_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        roi[:emoji_h, :emoji_w, c] = (alpha_s * emoji_resized[:, :, c] +
                                       alpha_l * roi[:emoji_h, :emoji_w, c])


# Initialize mediapipe solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1,-1)

        # Predict emotion label
        pred = label[np.argmax(model.predict(lst))]

        # Get the corresponding emoji
        emoji = emojis.get(pred, None)

        if emoji is not None:
            # Overlay emoji on the frame
            overlay_emoji(frm, emoji, 50, 100)

        # Draw text indicating the emotion detected
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
