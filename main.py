from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, jsonify, make_response
from werkzeug.utils import secure_filename
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import cv2
import os
import re
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()



# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)

jwt = JWTManager(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Temporary in-memory user store (use DB in production)
users = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_filename(original_name, operation):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(secure_filename(original_name))[0]
    ext_map = {
        'cpng': '.png', 'cgray': '.png', 'cwebp': '.webp', 'cjpg': '.jpg',
        'cthresh': '.png', 'cblur': '.png', 'csharpen': '.png', 'crotate': '.png',
        'chflip': '.png', 'cvflip': '.png', 'cedges': '.png', 'cinvert': '.png',
        'cbackgrounderase': '.png', 'ccartoon': '.png', 'csepia': '.png', 'canime': '.png'
    }
    return f"{base}_{operation}_{timestamp}{ext_map.get(operation, '.png')}"

def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def sepia_effect(img):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def anime_style_effect(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    img[edges == 255] = [0, 0, 0]
    return img

def remove_background(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = img * mask2[:, :, np.newaxis]
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(result)
    return cv2.merge([b, g, r, alpha], 4)

def process_image(filename, operation):
    try:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image file")

        operations = {
            'cgray': lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
            'cwebp': lambda x: x, 'cjpg': lambda x: x, 'cpng': lambda x: x,
            'crotate': lambda x: cv2.warpAffine(x, cv2.getRotationMatrix2D((x.shape[1]/2, x.shape[0]/2), 90, 1), (x.shape[1], x.shape[0])),
            'chflip': lambda x: cv2.flip(x, 1), 'cvflip': lambda x: cv2.flip(x, 0),
            'cedges': lambda x: cv2.Canny(x, 100, 200), 'cinvert': lambda x: cv2.bitwise_not(x),
            'cblur': lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            'csharpen': lambda x: cv2.filter2D(x, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
            'cthresh': lambda x: cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1],
            'cbackgrounderase': remove_background, 'ccartoon': cartoon_effect,
            'csepia': sepia_effect, 'canime': anime_style_effect
        }

        processed_img = operations[operation](img)
        new_filename = generate_filename(filename, operation)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], new_filename)

        if operation == 'cwebp':
            cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_WEBP_QUALITY, 90])
        elif operation == 'cjpg':
            cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            cv2.imwrite(output_path, processed_img)

        return f"processed/{new_filename}"

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/edit", methods=["POST"])
@jwt_required()
def edit():
    if 'file' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('home'))

    file = request.files['file']
    operation = request.form.get("operation")

    if not operation or operation == "Choose an Operation":
        flash("Please select an operation", "error")
        return redirect(url_for('home'))

    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('home'))

    if not allowed_file(file.filename):
        flash("File type not allowed", "error")
        return redirect(url_for('home'))

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        processed_path = process_image(filename, operation)
        flash(f"Image processed! <a href='/{processed_path}' target='_blank'>View</a>", "success")
        os.remove(upload_path)

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        flash(f"Error processing image: {str(e)}", "error")

    return redirect(url_for('home'))

@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email in users and users[email]['password'] == password:
            access_token = create_access_token(identity=email)
            response = make_response(redirect(url_for("home")))
            response.set_cookie("access_token_cookie", access_token)
            flash(f"Welcome back, {email}!", "success")
            return response
        else:
            flash("Invalid email or password", "error")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        users[email] = {'name': name, 'password': password}
        
        if not is_valid_password(password):
            flash("Password must be at least 7 characters long, contain 1 uppercase letter, 1 number, and 1 special character.", "danger")
            return render_template('signup.html')
        
        flash(f"Account created for {name}!", "success")
        return redirect(url_for("login"))
    
    return render_template("signup.html")

def is_valid_password(password):
    pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]).{7,}$'
    return re.match(pattern, password)

@app.route("/logout")
def logout():
    response = redirect(url_for("home"))
    response.delete_cookie("access_token_cookie")
    flash("You have been logged out!", "info")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
