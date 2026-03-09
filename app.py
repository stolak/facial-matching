from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Max upload size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

# ===============================
# PRELOAD MODEL (IMPORTANT)
# ===============================
print("Loading DeepFace model... This may take a few seconds.")

MODEL_NAME = "Facenet"   # lighter than ArcFace
DETECTOR = "opencv"      # lighter detector

model = DeepFace.build_model(MODEL_NAME)

print("DeepFace model loaded successfully")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===================================
# FILE UPLOAD FACE COMPARISON
# ===================================
@app.route('/compare', methods=['POST'])
def compare_faces():

    try:

        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both image1 and image2 are required'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']

        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'No selected files'}), 400

        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Invalid file format'}), 400

        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], "temp1_" + filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], "temp2_" + filename2)

        file1.save(path1)
        file2.save(path2)

        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            model=model,
            enforce_detection=False
        )

        os.remove(path1)
        os.remove(path2)

        return jsonify({
            "verified": result["verified"],
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "message": "✅ MATCH (Same person)" if result["verified"] else "❌ NOT A MATCH"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================
# BASE64 IMAGE COMPARISON
# ===================================
@app.route('/compare/binary', methods=['POST'])
def compare_faces_binary():

    try:

        data = request.get_json()

        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Both image1 and image2 base64 strings required'}), 400

        image1 = base64.b64decode(data['image1'])
        image2 = base64.b64decode(data['image2'])

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], "temp1.jpg")
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], "temp2.jpg")

        with open(path1, "wb") as f:
            f.write(image1)

        with open(path2, "wb") as f:
            f.write(image2)

        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            model=model,
            enforce_detection=False
        )

        os.remove(path1)
        os.remove(path2)

        return jsonify({
            "verified": result["verified"],
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "message": "✅ MATCH (Same person)" if result["verified"] else "❌ NOT A MATCH"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================
# HEALTH CHECK
# ===================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": MODEL_NAME
    })


# ===================================
# RUN LOCAL SERVER
# ===================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)