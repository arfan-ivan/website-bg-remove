from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from PIL import Image
import cv2
import numpy as np
from rembg import remove
import io
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def remove_background(image_path):
    """Remove background while retaining original image quality and resolution."""
    with open(image_path, 'rb') as f:
        input_image = f.read()
    output_image = remove(input_image)
    original_img = Image.open(image_path)
    processed_img = Image.open(io.BytesIO(output_image)).convert("RGBA")
    if processed_img.size != original_img.size:
        processed_img = processed_img.resize(original_img.size, Image.ANTIALIAS)
    return processed_img

def remove_watermark(image_path):
    """Remove watermarks and maintain image quality."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    inpainted = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    result_img = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    return result_img

def enhance_image_quality(image_path):
    """Enhance image quality by increasing resolution and detail."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    model_path = "ESPCN_x4.pb"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", 4)
    upscaled_img = sr.upsample(img)
    lab = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    result_img = Image.fromarray(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    return result_img

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    """Process image based on the selected task."""
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    image = request.files['image']
    task = request.form.get('task', 'background')
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        if task == 'background':
            processed_image = remove_background(image_path)
        elif task == 'watermark':
            processed_image = remove_watermark(image_path)
        elif task == 'enhance':
            processed_image = enhance_image_quality(image_path)
        else:
            return jsonify({'error': 'Invalid task'}), 400

        output_filename = f'arfanBgRemove_{os.path.splitext(image.filename)[0]}.png'
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        processed_image.save(output_path, format="PNG")
        return redirect(url_for('download_image', filename=output_filename))
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

@app.route('/download/<filename>')
def download_image(filename):
    """Provide the processed image for download."""
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, mimetype='image/png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5022)
