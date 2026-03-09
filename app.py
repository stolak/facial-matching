from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import base64
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        # Check if files are present
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both image1 and image2 are required'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Invalid file format. Allowed: jpg, jpeg, png, gif, bmp'}), 400
        
        # Save uploaded files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'temp1_' + filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'temp2_' + filename2)
        
        file1.save(path1)
        file2.save(path2)
        
        # Verify faces
        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name="ArcFace",
            detector_backend="retinaface"
        )
        
        # Clean up temp files
        os.remove(path1)
        os.remove(path2)
        
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'message': '✅ MATCH (Same person)' if result['verified'] else '❌ NOT A MATCH (Different persons)'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/compare/binary', methods=['POST'])
def compare_faces_binary():
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Both image1 and image2 are required as base64 strings'}), 400

        # Decode base64 images
        try:
            image1_data = base64.b64decode(data['image1'])
            image2_data = base64.b64decode(data['image2'])
        except Exception as e:
            return jsonify({'error': 'Invalid base64 encoding: ' + str(e)}), 400

        # Save to temporary files
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'temp1.jpg')
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'temp2.jpg')
        with open(path1, 'wb') as f:
            f.write(image1_data)
        with open(path2, 'wb') as f:
            f.write(image2_data)
        # Begin processing
        print("Behing processing")
        
        # Verify faces
        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name="ArcFace",
            detector_backend="retinaface"
        )

        # Clean up temp files
        os.remove(path1)
        os.remove(path2)

        # Return result
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'message': '✅ MATCH (Same person)' if result['verified'] else '❌ NOT A MATCH (Different persons)'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
