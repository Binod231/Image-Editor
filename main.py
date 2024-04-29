from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processImage(filename, operation):
    img = cv2.imread(f"uploads/{filename}")
    if operation == "cgray":
        imgProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newFilename = f"static/{filename}"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cwebp": 
        newFilename = f"static/{filename.split('.')[0]}.webp"
        cv2.imwrite(newFilename, img)
        return newFilename
    elif operation == "cjpg": 
        newFilename = f"static/{filename.split('.')[0]}.jpg"
        cv2.imwrite(newFilename, img)
        return newFilename
    elif operation == "cpng": 
        newFilename = f"static/{filename.split('.')[0]}.png"
        cv2.imwrite(newFilename, img)
        return newFilename
    # Add 6 more operations here
    elif operation == "crotate":
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)  # Rotate the image by 90 degrees
        imgProcessed = cv2.warpAffine(img, M, (cols, rows))
        newFilename = f"static/{filename.split('.')[0]}_rotate.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "chflip":
        imgProcessed = cv2.flip(img, 1)  # Horizontal flip
        newFilename = f"static/{filename.split('.')[0]}_hflip.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cvflip":
        imgProcessed = cv2.flip(img, 0)  # Vertical flip
        newFilename = f"static/{filename.split('.')[0]}_vflip.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cedges":
        imgProcessed = cv2.Canny(img, 100, 200)  # Detect edges
        newFilename = f"static/{filename.split('.')[0]}_edges.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cinvert":
        imgProcessed = cv2.bitwise_not(img)  # Invert colors
        newFilename = f"static/{filename.split('.')[0]}_invert.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cblur":
        imgProcessed = cv2.GaussianBlur(img, (5, 5), 0)  # Blur
        newFilename = f"static/{filename.split('.')[0]}_blur.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    # Additional operations
    elif operation == "csharpen":
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        imgProcessed = cv2.filter2D(img, -1, kernel)
        newFilename = f"static/{filename.split('.')[0]}_sharpen.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cthreshold":
        _, imgProcessed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        newFilename = f"static/{filename.split('.')[0]}_threshold.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cgaussianblur":
        imgProcessed = cv2.GaussianBlur(img, (9, 9), 0)
        newFilename = f"static/{filename.split('.')[0]}_gaussianblur.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cmedianblur":
        imgProcessed = cv2.medianBlur(img, 5)
        newFilename = f"static/{filename.split('.')[0]}_medianblur.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cdilate":
        kernel = np.ones((5, 5), np.uint8)
        imgProcessed = cv2.dilate(img, kernel, iterations=1)
        newFilename = f"static/{filename.split('.')[0]}_dilate.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename
    elif operation == "cerode":
        kernel = np.ones((5, 5), np.uint8)
        imgProcessed = cv2.erode(img, kernel, iterations=1)
        newFilename = f"static/{filename.split('.')[0]}_erode.png"
        cv2.imwrite(newFilename, imgProcessed)
        return newFilename

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == "POST": 
        operation = request.form.get("operation")
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return "error no selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new = processImage(filename, operation)
            flash(f"Your image has been processed and is available <a href='/{new}' target='_blank'>here</a>")
            return render_template("index.html")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
