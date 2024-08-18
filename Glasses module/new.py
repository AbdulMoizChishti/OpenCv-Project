import cv2
import cvzone
import os

# Initialize the camera and classifiers
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Load all glass images into a list
glass_folder = 'Glasses'
glass_files = [f for f in os.listdir(glass_folder) if f.endswith('.png')]
glasses = [cv2.imread(os.path.join(glass_folder, f), cv2.IMREAD_UNCHANGED) for f in glass_files]

selected_glass = 0

def draw_catalog(frame, glasses):
    catalog_height = 100
    num_glasses = len(glasses)
    thumbnail_size = 50
    spacing = 20

    for i, glass in enumerate(glasses):
        thumbnail = cv2.resize(glass, (thumbnail_size, thumbnail_size))
        x = spacing + i * (thumbnail_size + spacing)
        y = frame.shape[0] - catalog_height + 10

        # Ensure the thumbnails fit within the frame
        if x + thumbnail_size > frame.shape[1]:
            break

        frame[y:y+thumbnail_size, x:x+thumbnail_size] = thumbnail[:, :, :3]
        if i == selected_glass:
            cv2.rectangle(frame, (x-5, y-5), (x+thumbnail_size+5, y+thumbnail_size+5), (0, 255, 0), 2)

def check_catalog_click(x, y, num_glasses, frame_height, frame_width):
    catalog_height = 100
    thumbnail_size = 50
    spacing = 20

    if y < frame_height - catalog_height:
        return None
    for i in range(num_glasses):
        bx = spacing + i * (thumbnail_size + spacing)
        by = frame_height - catalog_height + 10
        if bx + thumbnail_size > frame_width:
            break
        if bx <= x <= bx + thumbnail_size and by <= y <= by + thumbnail_size:
            return i
    return None

def mouse_click(event, x, y, flags, param):
    global selected_glass
    if event == cv2.EVENT_LBUTTONDOWN:
        glass_num = check_catalog_click(x, y, len(glasses), frame.shape[0], frame.shape[1])
        if glass_num is not None:
            selected_glass = glass_num

cv2.namedWindow('SnapLens')
cv2.setMouseCallback('SnapLens', mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_scale)

    for (x, y, w, h) in faces:
        roi_gray = gray_scale[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        overlay = cv2.resize(glasses[selected_glass], (w, int(h*0.8)))
        frame = cvzone.overlayPNG(frame, overlay, [x, y])

    draw_catalog(frame, glasses)
    
    cv2.imshow('SnapLens', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

