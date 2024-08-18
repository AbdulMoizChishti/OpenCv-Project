import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Global variables
selected_watch_img = None
carousel_img = None
watch_images = []

def load_images_from_current_directory():
    images = []
    current_directory = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(current_directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                images.append(img)
    return images

def overlay_image(background, overlay, x, y, scale=1, angle=0):
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
    (h, w) = overlay.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    h, w, _ = overlay.shape

    if x >= background.shape[1] or y >= background.shape[0] or x + w <= 0 or y + h <= 0:
        return background

    x_end = min(x + w, background.shape[1])
    y_end = min(y + h, background.shape[0])

    overlay_x_start = max(0, x)
    overlay_y_start = max(0, y)
    overlay_x_end = x_end - x
    overlay_y_end = y_end - y

    if overlay_x_end != overlay_x_start and overlay_y_end != overlay_y_start:
        alpha_overlay = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(3):
            background[y:y_end, x:x_end, c] = (alpha_overlay * overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end, c] +
                                               alpha_background * background[y:y_end, x:x_end, c])
    return background

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def main():
    global selected_watch_img, carousel_img, watch_images

    watch_images = load_images_from_current_directory()
    if len(watch_images) > 0:
        selected_watch_img = watch_images[0]

    carousel_img = np.zeros((100, 640, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1])
                wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])

                index_finger_base_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * img.shape[1])
                index_finger_base_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * img.shape[0])
                pinky_base_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * img.shape[1])
                pinky_base_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * img.shape[0])

                angle = calculate_angle((pinky_base_x, pinky_base_y), (index_finger_base_x, index_finger_base_y))
                distance = np.sqrt((index_finger_base_x - wrist_x)**2 + (index_finger_base_y - wrist_y)**2)
                scale_factor = distance / 150.0

                if selected_watch_img is not None:
                    img = overlay_image(img, selected_watch_img,
                                        wrist_x - int(selected_watch_img.shape[1] * scale_factor) // 2,
                                        wrist_y - int(selected_watch_img.shape[0] * scale_factor) // 2,
                                        scale=scale_factor, angle=angle)

        height, width, _ = img.shape
        carousel_img_resized = cv2.resize(carousel_img, (width, carousel_img.shape[0]))

        combined_img = np.vstack((img, carousel_img_resized))

        cv2.imshow("Virtual Watch Try-On", combined_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
