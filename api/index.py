from flask import Flask, request, send_file, render_template_string
import segno
import os
import shutil
from werkzeug.utils import secure_filename
from io import BytesIO
import tempfile
import numpy as np
from PIL import Image, ImageStat, ImageOps, ImageSequence

import cv2
from pyzbar.pyzbar import decode

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <title>QR Code Generator</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --background: #f8f9fa;
            --card-bg: #ffffff;
            --text: #212529;
            --border: #dee2e6;
            --success: #38b000;
            --warning: #ff5400;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 600px;
            margin: 2rem auto;
            padding: 0 1rem;
        }

        .card {
            background-color: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            padding: 2rem;
            margin-bottom: 2rem;
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        h1 {
            color: var(--primary);
            text-align: center;
            margin-top: 0;
            margin-bottom: 1.5rem;
            font-weight: 700;
            font-size: 2rem;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--text);
        }

        input[type="text"],
        input[type="file"],
        button {
            width: 100%;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            background-color: white;
            font-size: 1rem;
            transition: all 0.2s ease;
        }

        input[type="text"]:focus,
        input[type="file"]:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15);
        }

        .file-upload {
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 1.5rem;
            background-color: rgba(67, 97, 238, 0.03);
        }

        .file-upload:hover {
            border-color: var(--accent);
            background-color: rgba(67, 97, 238, 0.06);
        }

        .file-upload p {
            margin: 0 0 0.5rem 0;
            font-weight: 500;
        }

        .file-upload input[type="file"] {
            display: none;
        }

        .file-name {
            font-size: 0.875rem;
            color: var(--accent);
            margin-top: 0.5rem;
            display: none;
        }

        .color-section {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .color-row {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .color-picker-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-grow: 1;
        }

        .color-preview {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            border: 1px solid var(--border);
            display: inline-block;
        }

        input[type="color"] {
            -webkit-appearance: none;
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 8px;
            background: none;
            cursor: pointer;
        }

        input[type="color"]::-webkit-color-swatch-wrapper {
            padding: 0;
        }

        input[type="color"]::-webkit-color-swatch {
            border: none;
            border-radius: 8px;
            box-shadow: 0 0 0 1px var(--border);
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--primary);
        }

        button {
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            border: none;
            cursor: pointer;
            padding: 1rem;
            transition: all 0.2s ease;
        }

        button:hover {
            background-color: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67, 97, 238, 0.2);
        }

        .coffee-section {
            text-align: center;
            margin-top: 1rem;
            padding: 1.5rem;
            background-color: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }

        .coffee-button {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.8rem 1.5rem;
            background: linear-gradient(45deg, #ffdd00, #ffca00);
            color: #333;
            font-weight: 700;
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(255, 202, 0, 0.2);
        }

        .coffee-button:hover {
            transform: translateY(-3px) scale(1.03);
            box-shadow: 0 6px 18px rgba(255, 202, 0, 0.25);
        }

        .coffee-icon {
            font-size: 1.2rem;
        }

        #color-warning {
            display: none;
            color: var(--warning);
            margin-top: 0.5rem;
            font-size: 0.875rem;
            font-weight: 500;
            padding: 0.5rem;
            background-color: rgba(255, 84, 0, 0.1);
            border-radius: 6px;
        }

        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            overflow: auto;
        }

        .modal-content {
            background-color: var(--card-bg);
            margin: 15% auto;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            max-width: 500px;
            width: 90%;
            text-align: center;
            animation: modalAppear 0.3s ease-out;
        }

        @keyframes modalAppear {
            from {opacity: 0; transform: translateY(-30px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .modal-title {
            color: var(--warning);
            margin-top: 0;
        }

        .modal-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .modal-buttons button {
            max-width: 150px;
        }

        .btn-secondary {
            background-color: #6c757d;
        }

        .btn-secondary:hover {
            background-color: #5a6268;
        }

        @media (max-width: 768px) {
            .container {
                padding: 0 1rem;
                margin: 1rem auto;
            }

            .card {
                padding: 1.5rem;
            }

            .color-row {
                flex-direction: column;
                align-items: flex-start;
            }

            .checkbox-group {
                margin-top: 0.5rem;
            }
        }

        .footer {
            margin-top: 2rem;
            text-align: center;
            font-size: 0.875rem;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>QR Code Generator</h1>
            <form id="qr-form" action="/generate" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="data-input">Text or URL</label>
                    <input type="text" id="data-input" name="data" placeholder="Enter text or URL for your QR code" required>
                </div>

                <div class="form-group">
                    <label>Background Image/GIF (Optional)</label>
                    <div class="file-upload" id="file-upload-area">
                        <p>Drag & drop your image here or click to browse</p>
                        <span class="file-name" id="file-name">No file chosen</span>
                        <input type="file" name="file" id="file-input" accept="image/*">
                    </div>
                </div>

                <div class="form-group color-section">
                    <label for="color-picker">QR Code Options</label>
                    <div class="color-row">
                        <div class="color-picker-container">
                            <input type="color" id="color-picker" name="color" value="#000000">
                        </div>
                        <div id="color-warning" style="display: none; color: red; font-weight: bold;">Please select a darker color for better QR code visibility!</div>
                        <div class="checkbox-group">
                            <input type="checkbox" name="contrast_qr" id="contrast_qr">
                            <label for="contrast_qr">Automatically Pick QR Color</label>
                        </div>
                    </div>
                    <div id="color-warning">
                        Please select a dark color for better QR code visibility!
                    </div>
                </div>

                <div class="form-group">
                    <label for="filename-input">Filename (Optional)</label>
                    <input type="text" id="filename-input" name="filename" placeholder="Enter filename for your QR code">
                </div>

                <button type="submit">Generate QR Code</button>
            </form>
        </div>

        <div class="coffee-section">
            <p>If this tool is helpful, please consider supporting me</p>
            <a href="https://buymeacoffee.com/talltreee" target="_blank" class="coffee-button">
                <span class="coffee-icon">☕</span>
                Buy Me a Coffee
            </a>
        </div>

        <div class="footer">
            <p>Made with ❤️ by talltreee</p>
        </div>
    </div>

    <!-- Add modal for unscannable QR code notification -->
    <div id="unscannable-modal" class="modal">
        <div class="modal-content">
            <h2 class="modal-title">⚠️ QR Code May Not Be Scannable</h2>
            <p>The QR code generated with this background may be difficult to scan. This can happen due to:</p>
            <ul style="text-align: left;">
                <li>Background image is too busy or has too much contrast</li>
                <li>QR code color doesn't stand out enough from the background</li>
                <li>Background image has patterns that interfere with QR code scanning</li>
            </ul>
            <p>Would you like to try with a different background image or adjust the QR color?</p>
            <div class="modal-buttons">
                <button id="try-again-btn" class="btn-secondary">Try Again</button>
                <button id="download-anyway-btn">Download Anyway</button>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const colorPicker = document.getElementById('color-picker');
            const colorPreview = document.getElementById('color-preview');
            const colorHex = document.getElementById('color-hex');
            const colorWarning = document.getElementById('color-warning');
            const qrForm = document.getElementById('qr-form');
            const fileInput = document.getElementById('file-input');
            const fileUploadArea = document.getElementById('file-upload-area');
            const fileName = document.getElementById('file-name');
            const unscannableModal = document.getElementById('unscannable-modal');
            const tryAgainBtn = document.getElementById('try-again-btn');
            const downloadAnywayBtn = document.getElementById('download-anyway-btn');
            
            // Flag to track if we're downloading anyway
            let downloadAnyway = false;
            let qrBlob = null;
            let qrFilename = null;

            function isLightColor(color) {
                // Convert hex to RGB
                const r = parseInt(color.substr(1, 2), 16);
                const g = parseInt(color.substr(3, 2), 16);
                const b = parseInt(color.substr(5, 2), 16);

                // Calculate luminance
                const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

                // Consider color light if luminance is above 0.5
                return luminance > 0.5;
            }

            colorPicker.addEventListener('input', function() {
                if (isLightColor(colorPicker.value)) {
                    colorWarning.style.display = 'block';
                    alert("Warning: The selected color is too light. Please choose a darker color for the QR code to be scannable.");
                } else {
                    colorWarning.style.display = 'none';
                }
            });

            qrForm.addEventListener('submit', function(event) {
                if (isLightColor(colorPicker.value)) {
                    event.preventDefault();
                    colorWarning.style.display = 'block';
                    window.scrollTo({
                        top: colorWarning.offsetTop - 100,
                        behavior: 'smooth'
                    });
                    return;
                }
                
                // If not downloading anyway, intercept the form submission
                if (!downloadAnyway) {
                    event.preventDefault();
                    
                    const formData = new FormData(qrForm);
                    
                    // Add a parameter to check scannability
                    formData.append('check_scannable', 'true');
                    
                    fetch('/generate', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        if (response.headers.get('Content-Type').includes('application/json')) {
                            return response.json().then(data => {
                                if (data.scannable === false) {
                                    // Show the modal if not scannable
                                    unscannableModal.style.display = 'block';
                                    // Save the blob data for potential download
                                    return response.blob().then(blob => {
                                        qrBlob = blob;
                                        qrFilename = data.filename;
                                        return data;
                                    });
                                } else {
                                    // If scannable, get the blob and trigger download
                                    return response.blob().then(blob => {
                                        downloadQrCode(blob, data.filename);
                                        return data;
                                    });
                                }
                            });
                        } else {
                            // If response is not JSON, it's the direct file
                            return response.blob().then(blob => {
                                downloadQrCode(blob, formData.get('filename') || 'qr_code');
                                return { scannable: true };
                            });
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred. Please try again.');
                    });
                }
                
                // Reset the flag after submission
                downloadAnyway = false;
            });

            // Download the QR code blob
            function downloadQrCode(blob, filename) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }

            // Try again button
            tryAgainBtn.addEventListener('click', function() {
                unscannableModal.style.display = 'none';
            });

            // Download anyway button
            downloadAnywayBtn.addEventListener('click', function() {
                unscannableModal.style.display = 'none';
                if (qrBlob) {
                    downloadQrCode(qrBlob, qrFilename);
                    qrBlob = null;
                } else {
                    // If we don't have the blob yet, resubmit the form with the flag
                    downloadAnyway = true;
                    qrForm.submit();
                }
            });

            // File upload handling
            fileUploadArea.addEventListener('click', function() {
                fileInput.click();
            });

            fileInput.addEventListener('change', function() {
                if (fileInput.files.length > 0) {
                    fileName.textContent = fileInput.files[0].name;
                    fileName.style.display = 'block';
                    fileUploadArea.style.borderColor = 'var(--success)';
                } else {
                    fileName.textContent = 'No file chosen';
                    fileName.style.display = 'none';
                    fileUploadArea.style.borderColor = 'var(--border)';
                }
            });

            // Drag and drop handling
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                fileUploadArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                fileUploadArea.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                fileUploadArea.addEventListener(eventName, unhighlight, false);
            });

            function highlight() {
                fileUploadArea.style.borderColor = 'var(--accent)';
                fileUploadArea.style.backgroundColor = 'rgba(67, 97, 238, 0.1)';
            }

            function unhighlight() {
                fileUploadArea.style.borderColor = 'var(--border)';
                fileUploadArea.style.backgroundColor = 'rgba(67, 97, 238, 0.03)';
            }

            fileUploadArea.addEventListener('drop', handleDrop, false);

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;

                if (files.length > 0) {
                    fileInput.files = files;
                    fileName.textContent = files[0].name;
                    fileName.style.display = 'block';
                    fileUploadArea.style.borderColor = 'var(--success)';
                }
            }
        });
    </script>
</body>
</html>
'''

def is_qr_code_scannable(image_path):
    """Check if the generated QR code is scannable."""
    try:
        image = cv2.imread(image_path)
        qr_codes = decode(image)
        return len(qr_codes) > 0
    except Exception as e:
        print(f"Error checking QR code scannability: {e}")
        return False

def get_opposite_dark_color(image_path):
    """Returns a high-contrast dark color based on the majority color of the image."""
    image = Image.open(image_path)
    if image.format == "GIF":
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
        all_pixels = np.concatenate([np.array(frame).reshape(-1, 3) for frame in frames], axis=0)
    else:
        image = image.convert("RGB")
        all_pixels = np.array(image).reshape(-1, 3)

    avg_color = np.mean(all_pixels, axis=0)
    opposite_color = 255 - avg_color
    dark_color = tuple(max(0, min(100, int(c))) for c in opposite_color)
    return f'#{dark_color[0]:02x}{dark_color[1]:02x}{dark_color[2]:02x}'

def is_image_dark(image_path):
    try:
        image = Image.open(image_path)

        if image.format == "GIF":
            frames = [frame.convert("L") for frame in ImageSequence.Iterator(image)]
            avg_luminance = sum(ImageStat.Stat(frame).mean[0] for frame in frames) / len(frames)
        else:
            image = image.convert("L")
            avg_luminance = ImageStat.Stat(image).mean[0]

        return avg_luminance < 128
    except Exception as e:
        print(f"Error analyzing image brightness: {e}")
        return False

def is_image_high_contrast(image_path):
    try:
        image = Image.open(image_path)
        if image.format == "GIF":
            frames = [frame.convert("L") for frame in ImageSequence.Iterator(image)]
            all_pixels = np.concatenate([np.array(frame).flatten() for frame in frames])
        else:
            image = image.convert("L")
            all_pixels = np.array(image).flatten()

        dark_threshold = np.percentile(all_pixels, 5)
        bright_threshold = np.percentile(all_pixels, 95)

        contrast = bright_threshold - dark_threshold
        return contrast > 100
    except Exception as e:
        print(f"Error analyzing image contrast: {e}")
        return False

def invert_image(image_path):
    try:
        image = Image.open(image_path)
        inverted_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        if image.format == "GIF":
            frames = [ImageOps.invert(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]
            frames[0].save(inverted_path, save_all=True, append_images=frames[1:])
        else:
            image = ImageOps.invert(image.convert("RGB"))
            image.save(inverted_path)

        return inverted_path
    except Exception as e:
        print(f"Error inverting image: {e}")
        return image_path

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/generate', methods=['POST'])
def generate_qr():
    data = request.form.get('data')
    file = request.files.get('file')
    filename = request.form.get('filename') or 'qr_code'
    color = request.form.get('color', '#000000')
    bw = request.form.get('bw')
    contrast_qr = request.form.get('contrast_qr')  # Checkbox for high contrast QR color
    check_scannable = request.form.get('check_scannable', 'false') == 'true'  # New parameter

    if not data:
        return "No data provided", 400

    filename = secure_filename(filename)
    file_extension = 'png'
    qr = segno.make(data, error='H', boost_error=True)
    temp_file_path = None

    if file:
        uploaded_filename = secure_filename(file.filename)
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif'}
        file_ext = os.path.splitext(uploaded_filename)[1].lower()
        if file_ext not in allowed_extensions:
            return "Invalid file format. Please upload a PNG, JPG, or GIF image.", 400

        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=file_ext)
        file.save(temp_file.name)
        temp_file_path = temp_file.name

        if is_image_dark(temp_file_path) and not is_image_high_contrast(temp_file_path):
            temp_file_path = invert_image(temp_file_path)

        file_extension = file_ext[1:].lower()

    if contrast_qr and temp_file_path:  # If the checkbox is checked, apply opposite color
        color = get_opposite_dark_color(temp_file_path)

    img_io = BytesIO()
    is_scannable = True

    if temp_file_path:
        try:
            output_qr_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}").name
            qr.to_artistic(background=temp_file_path, target=output_qr_path, scale=12, border=4, kind='jpeg' if file_extension == 'jpg' else file_extension, dark=color)

            # Check if the QR code is scannable
            if check_scannable:
                is_scannable = is_qr_code_scannable(output_qr_path)

            with open(output_qr_path, "rb") as qr_file:
                img_io.write(qr_file.read())

            img_io.seek(0)
        except Exception as e:
            return f"Error generating QR code with background: {e}", 400
        finally:
            if temp_file_path:
                os.unlink(temp_file_path)
            if 'output_qr_path' in locals() and os.path.exists(output_qr_path):
                os.unlink(output_qr_path)
    else:
        qr.save(img_io, kind='png', scale=12, border=4, dark=color)
        img_io.seek(0)
        # Plain QR codes should always be scannable
        is_scannable = True

    # If checking scannability and the QR is not scannable, return JSON response
    if check_scannable and not is_scannable:
        from flask import jsonify
        return jsonify({
            'scannable': False,
            'filename': f"{filename}.{file_extension}",
            'message': "The generated QR code may not be scannable. Please try a different background or color."
        })

    return send_file(img_io, mimetype=f'image/{file_extension}', as_attachment=True, download_name=f"{filename}.{file_extension}")

if __name__ == '__main__':
    app.run(debug=True)
