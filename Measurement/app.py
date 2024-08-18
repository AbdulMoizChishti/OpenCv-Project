from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import imutils
from imutils import perspective
from scipy.spatial import distance

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def giveSizeAccordingToMeasurement(measurement):
    size = ''
    min_diff = 1000
    sizechart = {'48':'S', '53':'M', '58': 'L', '63':'XL', '68':'XXL', '73':'XXXL'}
    for key in sizechart.keys():
        diff = abs(measurement - int(key))
        if diff < min_diff:
            size = sizechart[key]
            min_diff = diff
    return size

def get_size(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 'Error: Image not found!'
        
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) == 0:
            return 'Error: No contours found!'
        
        c = max(cnts, key=cv2.contourArea)
        ((x, y), (w, h), angle) = cv2.minAreaRect(c)
        
        box = cv2.boxPoints(((x, y), (w, h), angle))
        box = np.array(box, dtype="int")
        
        if len(box) == 0:
            return 'Error: No valid points found!'
        
        box = perspective.order_points(box)
        
        (tl, tr, br, bl) = box
        
        tltrX, tltrY = midpoint(tl, tr)
        blbrX, blbrY = midpoint(bl, br)
        
        tlblX, tlblY = midpoint(tl, bl)
        trbrX, trbrY = midpoint(tr, br)
        
        dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        if dA < dB:
            dimA = dA
        else:
            dimA = dB
        
        pixelsPerMetric = dimA / 24.0
        
        dimA = dimA / pixelsPerMetric
        
        return giveSizeAccordingToMeasurement(dimA)
    except Exception as e:
        return 'Error: {}'.format(str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        size_result = get_size(filepath)
        return jsonify({'result': size_result})
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
