from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os
from flask import send_from_directory
import numpy as np
from datetime import datetime

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)
# In app.py, update these lines:
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))  # Fallback for local dev
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['PROCESSED_FOLDER'] = os.environ.get('PROCESSED_FOLDER', 'static/processed')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_filename(original_name, operation):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(secure_filename(original_name))[0]
    
    # Update the extensions for new operations
    ext = {
        'cpng': '.png',
        'cgray': '.png',
        'cwebp': '.webp',
        'cjpg': '.jpg',
        'cthresh': '.png',
        'cblur': '.png',
        'csharpen': '.png',
        'crotate': '.png',
        'chflip': '.png',
        'cvflip': '.png',
        'cedges': '.png',
        'cinvert': '.png',
        'cbackgrounderase': '.png',
        'ccartoon': '.png',    # New operation: cartoon
        'csepia': '.png',      # New operation: sepia
        'canime': '.png'       # New operation: anime style
    }.get(operation, '.png')  # Default to .png if operation is unknown
    
    return f"{base}_{operation}_{timestamp}{ext}"


def process_image(filename, operation):
    """Process image with OpenCV based on selected operation"""
    try:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError("Could not read image file")
            
        operations = {
            'cgray': lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
            'cwebp': lambda x: x,
            'cjpg': lambda x: x,
            'cpng': lambda x: x,
            'crotate': lambda x: cv2.warpAffine(
                x, 
                cv2.getRotationMatrix2D((x.shape[1]/2, x.shape[0]/2), 90, 1), 
                (x.shape[1], x.shape[0])
            ),
            'chflip': lambda x: cv2.flip(x, 1),
            'cvflip': lambda x: cv2.flip(x, 0),
            'cedges': lambda x: cv2.Canny(x, 100, 200),
            'cinvert': lambda x: cv2.bitwise_not(x),
            'cblur': lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            'csharpen': lambda x: cv2.filter2D(
                x, -1, 
                np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            ),
            'cthresh': lambda x: cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1],
            'cbackgrounderase': lambda x: remove_background(x),
            'ccartoon': cartoon_effect,
            'csepia': sepia_effect,
            'canime': anime_style_effect
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
            
        processed_img = operations[operation](img)
        
        new_filename = generate_filename(filename, operation)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], new_filename)
        
        # Handle different output formats
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

def cartoon_effect(img):
    """Apply cartoon effect to the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter for smoothing
    color = cv2.bilateralFilter(img, 9, 300, 300)
    
    # Combine color and edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

def sepia_effect(img):
    """Apply sepia effect to the image"""
    # Create the sepia filter matrix
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    
    # Apply the sepia filter
    sepia_img = cv2.transform(img, kernel)
    sepia_img = np.clip(sepia_img, 0, 255)
    
    return sepia_img.astype(np.uint8)

def anime_style_effect(img):
    """Apply anime-style effect using bilateral filter and edge enhancement"""
    # Bilateral filter to smooth the image
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    
    # Combine the edges with the original image
    img[edges == 255] = [0, 0, 0]
    
    return img


def remove_background(img):
    """Advanced background removal using grabCut algorithm"""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define a rectangle around the object (you might want to make this smarter)
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = img * mask2[:, :, np.newaxis]
    
    # Create transparent background
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(result)
    rgba = [b, g, r, alpha]
    return cv2.merge(rgba, 4)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/edit", methods=["POST"])
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
        flash(f"Your image has been processed! <a href='/{processed_path}' target='_blank'>View result</a>", "success")
        
        # Clean up upload after processing
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
        # You can add validation or database check here
        flash(f"Welcome back, {email}!", "success")
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        # You can store user info in database here
        flash(f"Account created for {name}!", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  
    app.run(host='0.0.0.0', port=port)