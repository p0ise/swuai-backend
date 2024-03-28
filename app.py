import os
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile

from business.face import FaceCompareClient, FaceFeatureClient

app = Flask(__name__)
CORS(app)

# 配置上传文件夹和允许的扩展
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_filename(filename):
    # 使用UUID或其他方法生成唯一的文件名
    unique_filename = str(uuid.uuid4())
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    safe_filename = f"{unique_filename}.{ext}" if ext else unique_filename
    return safe_filename


@app.route('/api/face-compare', methods=['POST'])
def face_compare():
    # 检查是否上传了两个文件
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': '缺少文件参数'}), 400
    file1 = request.files['image1']
    file2 = request.files['image2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    elif not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return jsonify({'error': '文件类型不正确'}), 400

    # 使用临时文件
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath1 = os.path.join(tmpdirname, secure_filename(file1.filename))
        filepath2 = os.path.join(tmpdirname, secure_filename(file2.filename))

        file1.save(filepath1)
        file2.save(filepath2)

        # 调用人脸比对服务
        client = FaceCompareClient()
        code, data = client.compare_faces(filepath1, filepath2)

    if code == 0:
        return jsonify({'message': '比对成功', 'data': data}), 200
    else:
        return jsonify({'error': data}), 400


@app.route('/api/face-features', methods=['POST'])
def face_features():
    if 'image' not in request.files:
        return jsonify({'error': '缺少文件参数'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择文件或文件类型不正确'}), 400
    elif not allowed_file(file.filename):
        return jsonify({'error': '文件类型不正确'}), 400

    # 使用临时文件
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        file.save(tmpfile.name)
        # 分析人脸特征
        client = FaceFeatureClient()
        result = client.analyze_all(tmpfile.name)


    # 清理临时文件
    os.remove(tmpfile.name)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
