from flask import Flask, render_template, request, flash, redirect, session
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'super secret key'

# Initialize the ResNet50 model for image classification
model = ResNet50(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filename, operation):
    img = cv2.imread(f"uploads/{filename}")
    img = cv2.resize(img, (224, 224))

    if operation == "ai_classification":
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        label = decode_predictions(preds)
        return label

    elif operation == "upscaling":
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "noise_reduction":
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "sharpening":
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "color_correction":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "hdr_effect":
        img = cv2.addWeighted(img, 0.5, img, 0.5, 0)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "skin_smoothing":
        img = cv2.bilateralFilter(img, 9, 75, 75)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "background_blur":
        img = cv2.GaussianBlur(img, (15, 15), 0)
        cv2.imwrite(f"static/{filename}", img)
        return f"static/{filename}"

    elif operation == "selective_editing":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_and(img, img, mask=mask)

    elif operation == "style_transfer":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Image format conversions
    elif operation == "cpng":
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "cgray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "cwebp":
        cv2.imwrite(f"static/{filename}.webp", img)
    elif operation == "cjpg":
        cv2.imwrite(f"static/{filename}.jpg", img)

    # Basic image adjustments
    elif operation == "cthresh":
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "cblur":
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "csharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "crotate":
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)  # Rotate 90 degrees
        img = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "chflip":
        img = cv2.flip(img, 1)  # Horizontal flip
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "cvflip":
        img = cv2.flip(img, 0)  # Vertical flip
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "cedges":
        img = cv2.Canny(img, 100, 200)  # Detect edges
        cv2.imwrite(f"static/{filename}.png", img)
    elif operation == "cinvert":
        img = cv2.bitwise_not(img)  # Invert colors
        cv2.imwrite(f"static/{filename}.png", img)

    # Save the processed image
    return f"static/{filename}"

@app.route("/")
def home():
    if session.get('logged_in'):
        return render_template("index.html", demo=True, demo_counter=session.get('demo_counter', 0))
    else:
        return render_template("index.html", demo=True, demo_counter=3)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == "POST": 
        operation = request.form.get("operation")
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/')
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect('/')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            new_image_path = process_image(filename, operation)
            if new_image_path:
                flash(f"Your image has been processed and is available <a href='/{new_image_path}' target='_blank'>here</a>")
            else:
                flash("Invalid operation")
            return redirect('/')

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True, port=5001)