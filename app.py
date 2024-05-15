#!/usr/bin/env python
# encoding: utf-8
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from keras._tf_keras.keras.preprocessing import image
from PIL import Image  
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


app = Flask(__name__)

BASE_URL = "127.0.0.1:5000"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
DATA_FOLDER = os.path.join(APP_ROOT, 'data')
MODEL_PATH = os.path.join(APP_ROOT, 'models/train_features.pkl')

# Tạo danh sách các tệp hợp lệ từ tất cả các tệp
list_images = []
for root, dirs, files in os.walk(DATA_FOLDER):
    dirs.sort()
    files.sort()
    for file in files:
        file_path = os.path.join(root, file)
        list_images.append(file_path)

# Kích thước ảnh đầu vào 
img_width, img_height = 128, 128

# Khởi tạo mô hình ResNet50 và bỏ đi lớp fully connected (top layer)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Đọc các đặc trưng từ tệp
with open(MODEL_PATH, 'rb') as file:
    train_features = pickle.load(file)

# Chuyển đổi thành mảng numpy
X_train = np.array(train_features)

# Trích xuất đặc trưng từ mô hình ResNet50
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

app.config['UPLOAD'] = UPLOAD_FOLDER
app.config['DATA'] = DATA_FOLDER

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD'], filename)

@app.route('/data/<folder>/<filename>')
def data_image(folder,filename):
    return send_from_directory(app.config['DATA'], folder + '/' +filename)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def recognize():
    file = request.files.get('image')
    number_result = request.form.get('number_result')
    if file:
        filename = secure_filename(file.filename)
        file_name_random = get_random_string(12) + filename
        filepath = os.path.join(app.config['UPLOAD'], file_name_random)
        file.save(filepath)

        # Khi có một ảnh mới đầu vào
        new_image_path = filepath

        # Trích xuất đặc trưng của ảnh mới
        new_image_feature = extract_features(new_image_path)

        x_test = [new_image_feature]

        # Tính toán cosine similarity giữa vector đặc trưng của hình ảnh mới và tất cả các hình ảnh trong X_train
        cos_similarities = cosine_similarity(x_test, X_train)

        # Sắp xếp các món ăn dựa trên cosine similarity
        sorted_indices = np.argsort(cos_similarities[0])[::-1]

        # Lấy danh sách các món ăn tương tự 
        num_similar_items = int(number_result)
        similar_items = sorted_indices[:num_similar_items]

        list_image_urls = []
        list_food_names = []
        # Duyệt qua từng đường dẫn ảnh trong danh sách similar_items
        for i in similar_items:
            food_name = list_images[similar_items[0]].split('data\\')[1].split('\\')[0].replace("_"," ")
            list_food_names.append(food_name)

            imageUrl = 'http://'+ BASE_URL + '/data' +list_images[i].split('data')[1].replace("\\", "/")
            list_image_urls.append(imageUrl)
        
        counter = Counter(list_food_names)
        food_name = counter.most_common(1)[0][0]

        return jsonify({
            'food_name': food_name,
            'list_image_urls': list_image_urls
        })
    else:
        return jsonify({'message': 'No file uploaded!'})
    

if __name__ == "__main__":
    app.run(debug=True)

