import os
import threading
import time
import uuid

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from license_plate import detect

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# CORS
CORS(app)

# Configure app from environment variables with defaults
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'public')
app.config['FILE_LIFETIME'] = int(os.getenv('FILE_LIFETIME', '60'))  # seconds
app.config['DEBUG'] = os.getenv('DEBUG', 'True').lower() == 'true'
app.config['PORT'] = int(os.getenv('PORT', '5000'))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def delete_file_after_delay(filepath, delay):
    """Delete a file after a specified delay in seconds"""
    def delete():
        time.sleep(delay)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file {filepath}: {str(e)}")

    thread = threading.Thread(target=delete)
    thread.daemon = True  # Thread will be terminated when main program exits
    thread.start()


@app.route('/process-plate', methods=['POST'])
def process_plate():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'ok': False, 'status': 400, 'error': 'No file provided'}), 400

        # Get the effect parameter (default to blur)
        effect = request.form.get('effect', 'blur')
        if effect not in ['blur', 'white']:
            return jsonify({'ok': False, 'status': 400, 'error': 'Invalid effect. Must be either "blur" or "white"'}), 400

        start_at = time.time()

        file = request.files['file']

        # Read image file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Create a unique temporary file name using timestamp and UUID
        temp_uuid = str(uuid.uuid4())
        temp_timestamp = str(int(time.time() * 1000))
        temp_path = f'temp_{temp_timestamp}_{temp_uuid}.jpg'

        try:
            # Save temporary file for processing
            cv2.imwrite(temp_path, img)

            # Detect license plate
            result = detect(temp_path)

            if result[0] is None or result[1] is None:
                return jsonify({'ok': False, 'status': 400, 'error': 'Could not detect license plate'}), 400

            license_plate_number = result[0]
            plate_img = result[1]

            # Apply effect and get modified image
            original = cv2.imread(temp_path)

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(
                        f"Error removing temporary file {temp_path}: {str(e)}")

        # Convert plate image to grayscale if it's not already
        if len(plate_img.shape) == 3:
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            plate_gray = plate_img

        # Use template matching to find the plate in the original image
        result = cv2.matchTemplate(cv2.cvtColor(
            original, cv2.COLOR_BGR2GRAY), plate_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Get the dimensions of the plate image
        h, w = plate_img.shape[:2]

        # Create white mask for the plate area
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Apply the selected effect
        if effect == 'white':
            original[top_left[1]:bottom_right[1], top_left[0]
                :bottom_right[0]] = [255, 255, 255]
        else:  # blur
            roi = original[top_left[1]:bottom_right[1],
                           top_left[0]:bottom_right[0]]
            blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
            original[top_left[1]:bottom_right[1],
                     top_left[0]:bottom_right[0]] = blurred_roi

        # Generate unique filename and save the processed image
        file_uuid = str(uuid.uuid4())
        _, buffer = cv2.imencode('.png', original)
        file_extension = 'png'
        filename = f"{file_uuid}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, original)

        # Schedule file deletion after FILE_LIFETIME seconds
        delete_file_after_delay(filepath, app.config['FILE_LIFETIME'])

        # Generate download URL
        download_url = f"/download/{filename}"

        # took time in seconds
        end_at = time.time()
        took = end_at - start_at

        return jsonify({
            "ok": True,
            "status": 200,
            "meta": {
                "took": took,
            },
            'number': license_plate_number,
            'url': download_url
        })

    except Exception as e:
        return jsonify({'ok': False, 'status': 500, 'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        return jsonify({'ok': False, 'status': 404, 'error': str(e)}), 404


if __name__ == '__main__':
    app.run(
        debug=app.config['DEBUG'],
        port=app.config['PORT']
    )
