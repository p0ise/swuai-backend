import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from business.face import FaceCompareClient, FaceFeatureClient


app = Flask(__name__)
CORS(app)

# 配置上传文件夹和允许的扩展
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/api/face-compare', methods=['POST'])
def face_compare():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)
        # 假设FaceCompareClient有一个compare_faces方法
        client = FaceCompareClient()
        result = client.compare_faces(filepath1, filepath2)
        return jsonify(result)
    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/api/face-features', methods=['POST'])
def face_features():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # 假设FaceFeatureClient有一个analyze_all_features方法
        client = FaceFeatureClient()
        result = client.analyze_all(filepath)
        return jsonify(result)
    return jsonify({'error': 'Invalid file format'}), 400


if __name__ == '__main__':
    app.run(debug=True)
