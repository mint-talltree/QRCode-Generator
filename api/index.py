from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import pandas as pd
import tempfile
from werkzeug.utils import secure_filename
from pathlib import Path
import cv2
import numpy as np
import re
import io
import unicodedata
from google.cloud import vision

app = Flask(__name__)
app.secret_key = "business_card_scanner_secret_key"

# Configure upload folder
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULTS_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload and results directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_text_with_google_vision(image_path):
    """Detect text in an image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # Get full text and also the individual text annotations with bounding boxes
    full_text = response.text_annotations[0].description if response.text_annotations else ""

    # Extract individual text blocks with their positions for spatial analysis
    text_blocks = []
    if len(response.text_annotations) > 1:  # Skip first one as it's the full text
        for text_annotation in response.text_annotations[1:]:
            vertices = [(vertex.x, vertex.y) for vertex in text_annotation.bounding_poly.vertices]
            # Calculate centroid of bounding box
            x_coords = [vertex[0] for vertex in vertices]
            y_coords = [vertex[1] for vertex in vertices]
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)

            text_blocks.append({
                'text': text_annotation.description,
                'vertices': vertices,
                'centroid': (centroid_x, centroid_y)
            })

    return full_text, text_blocks

def normalize_text(text):
    """Normalize text by removing extra spaces and converting to full-width to half-width."""
    # Convert full-width characters to half-width
    text = unicodedata.normalize('NFKC', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_email(text):
    """Extract email addresses from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ''

def extract_phone_fax(text):
    """Extract phone and fax numbers from text."""
    # Enhanced phone pattern matching for international and Japanese formats
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{4}',  # International format
        r'(?:\(?\d{2,4}\)?[-.\s]?){2,3}\d{4}',                 # Standard format
        r'\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{4}',                  # Japanese format
        r'0120[-.\s]?\d{2,3}[-.\s]?\d{3,4}'                    # Japanese toll-free
    ]

    phone = ''
    fax = ''

    # Process each line to distinguish between phone and fax
    for line in text.split('\n'):
        line_lower = line.lower()

        # Check if the line is explicitly labeled as fax
        is_fax_line = any(fax_label in line_lower for fax_label in ['fax', 'ファックス', 'f:', 'f.'])

        # Apply each pattern to find numbers
        for pattern in phone_patterns:
            numbers = re.findall(pattern, line)
            for num in numbers:
                # Clean up the found number
                clean_num = re.sub(r'[\s.]+', '-', num)

                # Store as either phone or fax
                if is_fax_line and not fax:
                    fax = clean_num
                elif not is_fax_line and not phone:
                    phone = clean_num

    return phone, fax

def extract_website(text):
    """Extract website URLs from text."""
    # Enhanced website pattern matching
    website_patterns = [
        r'(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:\/\S*)?',
        r'(?:[a-zA-Z0-9][a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:\/\S*)?'
    ]

    for line in text.split('\n'):
        line = line.lower().strip()
        # Skip lines that are likely not websites (e.g., email addresses)
        if '@' in line:
            continue

        for pattern in website_patterns:
            websites = re.findall(pattern, line)
            if websites:
                website = websites[0]
                # Remove trailing punctuation
                website = re.sub(r'[.,;:]$', '', website)
                # Add https:// if missing
                if not website.startswith(('http://', 'https://')):
                    website = 'https://' + website
                return website.strip('/')
    return ''

def extract_position(text):
    """Extract job position/title from text."""
    # Common job titles in English and Japanese
    position_patterns = [
        r'(?i)(CEO|CTO|CFO|COO|President|Director|Manager|Executive|Officer|Supervisor|Leader|Head\s+of.*?|.*?Engineer|.*?Developer|Consultant|Specialist|Associate|Analyst|Coordinator)',
        r'(代表取締役|取締役|部長|課長|主任|係長|担当|エンジニア|マネージャー|コンサルタント|スペシャリスト)'
    ]

    for line in text.split('\n'):
        line = line.strip()
        for pattern in position_patterns:
            match = re.search(pattern, line)
            if match:
                # Check if the position is part of a longer title or just a word in a sentence
                # Return the full line if it seems like a concise title
                if len(line) < 40 and not re.search(r'@|www|https?:|\.com', line):
                    # Check if line appears to be a full address
                    if not re.search(r'\d+[-\d]*', line) or not any(x in line for x in ['市', '区', '県', 'Street', 'Ave']):
                        return line
                # Otherwise just return the matched position
                return match.group(0)
    return ''

def extract_company_names(text, text_blocks):
    """Extract company names in Japanese and English."""
    company_jp = ''
    company_en = ''

    # Japanese company indicators
    jp_company_indicators = ['株式会社', '有限会社', '合同会社', '株式会社', '社団法人', '財団法人', '企業']

    # English company indicators
    en_company_indicators = ['Co.', 'Ltd.', 'Inc.', 'Corp.', 'Corporation', 'Company', 'Limited', 'LLC', 'LLP', 'Group']

    # First check in the full text by lines
    for line in text.split('\n'):
        line = line.strip()

        # Japanese company detection
        if not company_jp:
            for indicator in jp_company_indicators:
                if indicator in line:
                    # Extract the company name including the indicator
                    company_jp = line
                    break

        # English company detection
        if not company_en:
            for indicator in en_company_indicators:
                # Match the indicator as a whole word
                if re.search(r'\b' + re.escape(indicator) + r'\b', line):
                    company_en = line
                    break

    # If company names weren't found by line analysis, try spatial analysis with text blocks
    if text_blocks and (not company_jp or not company_en):
        # Sort text blocks by y-coordinate (top to bottom)
        sorted_blocks = sorted(text_blocks, key=lambda x: x['centroid'][1])

        # Look for company indicators in the blocks
        for block in sorted_blocks:
            block_text = block['text']

            # Japanese company
            if not company_jp:
                for indicator in jp_company_indicators:
                    if indicator in block_text:
                        company_jp = block_text
                        break

            # English company
            if not company_en:
                for indicator in en_company_indicators:
                    if re.search(r'\b' + re.escape(indicator) + r'\b', block_text):
                        company_en = block_text
                        break

            # Exit if both found
            if company_jp and company_en:
                break

    # Clean up company names
    if company_jp:
        company_jp = clean_company_name(company_jp)
    if company_en:
        company_en = clean_company_name(company_en)

    return company_jp, company_en

def clean_company_name(company):
    """Clean and standardize company names."""
    # Remove common noise patterns
    noise_patterns = [
        r'\s*[\(\（].*?[\)\）]',  # Text in parentheses
        r'\s*[,\.]$',            # Trailing punctuation
        r'^\s*[,\.]',            # Leading punctuation
    ]

    cleaned = company
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned)

    return cleaned.strip()

def is_japanese_text(text):
    """Check if text contains Japanese characters."""
    return any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text)

def extract_names(text, text_blocks):
    """Extract Japanese and English names with improved accuracy."""
    japanese_name = ''
    english_name = ''

    # First look at the top blocks which usually contain names
    if text_blocks:
        # Sort blocks by Y position (top to bottom)
        sorted_blocks = sorted(text_blocks, key=lambda x: x['centroid'][1])
        top_blocks = sorted_blocks[:5]  # Look at top 5 blocks

        for block in top_blocks:
            block_text = block['text'].strip()

            # Skip if it's likely an address, email, website, or phone
            if re.search(r'@|www|https?:|\.com|\d{3,}', block_text):
                continue

            # Japanese name detection
            if not japanese_name and is_japanese_text(block_text):
                if 1 < len(block_text) <= 10:  # Adjusted length for full names
                    japanese_name = block_text

            # English name detection
            if not english_name and re.match(r'^[A-Za-z\s.]+$', block_text):
                if 2 < len(block_text) < 40 and len(block_text.split()) <= 4:  # Reasonable name length
                    english_name = block_text

    # If names weren't found in blocks, try line-by-line analysis
    if not japanese_name or not english_name:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            if i > 4:  # Only check first few lines
                break

            # Skip if it's likely an address, email, website, or phone
            if re.search(r'@|www|https?:|\.com|\d{3,}', line):
                continue

            # Japanese name
            if not japanese_name and is_japanese_text(line):
                if 1 < len(line) <= 10:
                    japanese_name = line

            # English name
            if not english_name and re.match(r'^[A-Za-z\s.]+$', line):
                if 2 < len(line) < 40 and len(line.split()) <= 4:
                    english_name = line

    return japanese_name, english_name

def extract_address(text):
    """Extract postal address from text."""
    address = ''

    # Japanese address patterns (postal code + address)
    jp_address_patterns = [
        r'〒\d{3}-\d{4}\s*[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+',  # Postal code + Japanese text
        r'\d{3}-\d{4}\s*[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+',    # Postal code without symbol + Japanese text
        r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]*[都道府県市区町村][\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\d-]+' # Address with prefecture/city indicators
    ]

    # English/international address patterns
    en_address_patterns = [
        r'\d+\s+[A-Za-z\s,]+(?:Street|St\.|Road|Rd\.|Avenue|Ave\.|Boulevard|Blvd\.|Lane|Ln\.|Drive|Dr\.|Building|Bldg\.)',
        r'[A-Za-z]+\s+\d+,\s+[A-Za-z\s,]+\d{4,}',  # Format like "Some Street 123, City 12345"
        r'\d+\s+[A-Za-z\s,]+,\s+[A-Za-z\s]+,\s+[A-Z]{2}\s+\d{5}'  # US format with state code
    ]

    # Process each line to find addresses
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Skip lines that are clearly not addresses
        if re.search(r'@|www|https?:|\.com', line):
            continue

        # Check Japanese patterns
        for pattern in jp_address_patterns:
            if re.search(pattern, line):
                # If the line contains an address pattern and has location indicators, it's likely an address
                if any(marker in line for marker in ['丁目', '番地', '号', '階', '市', '区', '町', '村', '県']):
                    return line

        # Check English patterns
        for pattern in en_address_patterns:
            if re.search(pattern, line, re.I):
                return line

        # Fallback for lines that look like addresses but don't match patterns
        if len(line) > 10 and re.search(r'\d+', line):
            words = line.split()
            # If line has numbers and multiple words, it might be an address
            if len(words) > 3 and any(re.match(r'\d+', word) for word in words):
                # Check for common address words
                address_indicators = ['floor', 'suite', 'apt', 'apartment', 'room', 'building', 'block', 'street']
                if any(indicator in line.lower() for indicator in address_indicators):
                    return line

    return address

def extract_postal_code(text):
    """Extract postal code from text."""
    # Japanese postal code patterns
    jp_patterns = [
        r'〒\s*(\d{3}-\d{4})',
        r'郵便番号\s*(\d{3}-\d{4})',
        r'(\d{3}-\d{4})'  # Basic pattern, might need context checking
    ]

    # International postal code patterns
    intl_patterns = [
        r'[Pp]ostal\s+[Cc]ode:?\s*([A-Z0-9\s-]{3,10})',
        r'[Zz][Ii][Pp]\s*[Cc]ode:?\s*(\d{5}(?:-\d{4})?)'
    ]

    for line in text.split('\n'):
        # Check Japanese patterns first
        for pattern in jp_patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

        # Then check international patterns
        for pattern in intl_patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

    return ''

def extract_info_from_image(image_path):
    """Extract all information from a business card image."""
    # Get text from Google Cloud Vision
    full_text, text_blocks = detect_text_with_google_vision(image_path)

    # Normalize the text
    normalized_text = normalize_text(full_text)

    # Extract phone and fax
    phone, fax = extract_phone_fax(full_text)

    # Extract company names
    company_jp, company_en = extract_company_names(full_text, text_blocks)

    # Extract personal names
    japanese_name, english_name = extract_names(full_text, text_blocks)

    # Gather all information
    info = {
        'japanese_name': japanese_name,
        'english_name': english_name,
        'email': extract_email(full_text),
        'phone': phone,
        'fax': fax,
        'website': extract_website(full_text),
        'company_jp': company_jp,
        'company_en': company_en,
        'position': extract_position(full_text),
        'address': extract_address(full_text),
        'postal_code': extract_postal_code(full_text),
        'raw_text': full_text.strip(),
        'filename': Path(image_path).name
    }

    return info

def process_multiple_cards(file_paths):
    """Process multiple business card images and return results."""
    data = []
    
    for file_path in file_paths:
        try:
            info = extract_info_from_image(file_path)
            if info:
                data.append(info)
        except Exception as e:
            flash(f"Error processing {Path(file_path).name}: {str(e)}", "error")
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Create temporary directories for this session
    session_upload_folder = tempfile.mkdtemp()
    session_results_folder = tempfile.mkdtemp()
    
    if 'files[]' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    if not files or all(file.filename == '' for file in files):
        flash('No selected file', 'error')
        return redirect(request.url)
    
    # Save files to temp upload folder
    file_paths = []
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_upload_folder, filename)
                file.save(file_path)
                file_paths.append(file_path)
            else:
                flash(f'Invalid file type: {file.filename}. Only PNG and JPG are allowed.', 'error')
        
        if not file_paths:
            return redirect(request.url)
        
        # Process the uploaded business cards
        results = process_multiple_cards(file_paths)
        
        if results:
            # Generate results files
            df = pd.DataFrame(results)
            result_id = str(int(time.time()))
            
            csv_path = os.path.join(session_results_folder, f"results_{result_id}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # Store path in session
            session['last_result'] = {
                'csv_path': csv_path,
                'result_id': result_id
            }
            
            # Clean up upload directory
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            try:
                os.rmdir(session_upload_folder)
            except:
                pass
            
            # Trigger immediate download
            return send_file(
                csv_path,
                as_attachment=True,
                download_name=f"business_cards_{result_id}.csv",
                mimetype='text/csv'
            )
        else:
            flash('No business card information could be extracted', 'error')
            return redirect(request.url)
            
    except Exception as e:
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(request.url)
    finally:
        # Clean up temp directories in case of errors
        try:
            import shutil
            shutil.rmtree(session_upload_folder, ignore_errors=True)
            shutil.rmtree(session_results_folder, ignore_errors=True)
        except:
            pass

@app.route('/download/<filetype>/<result_id>')
def download_file(filetype, result_id):
    from flask import session
    
    if 'last_result' not in session or session['last_result']['result_id'] != result_id:
        flash('Result not found or expired', 'error')
        return redirect(url_for('index'))
    
    if filetype == 'csv':
        file_path = session['last_result']['csv_path']
        return send_file(file_path, as_attachment=True, download_name=f"business_cards_{result_id}.csv")
    elif filetype == 'excel':
        file_path = session['last_result']['excel_path']
        return send_file(file_path, as_attachment=True, download_name=f"business_cards_{result_id}.xlsx")
    else:
        flash('Invalid file type requested', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
