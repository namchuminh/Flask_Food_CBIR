# Flask_Food_Detection
Xây dựng hệ thống gợi ý các món ăn liên quan dựa trên hình ảnh

# Yêu cầu
- Python 3.11.4
- tensorflow==2.12.0
- keras==3.3.3

# Cài đặt
- B1: Clone repo này về máy
- B2: Tạo lệnh cmd: pip install -r requirements.txt
- B3: Chạy server: python app.py
- B4: Truy cập địa chỉ: http://127.0.0.1:5000/
- B5: Upload ảnh món ăn và tìm kiếm kết quả

# Hướng dẫn tải lại Model và dữ liệu (data)

## Hướng dẫn tải dữ liệu 

Tạo 1 thư mục tên: data

Tải dữ liệu từ link này: https://drive.google.com/file/d/11kzCboVpLUJto95o_PTwZthjrI_EpWme/view

Giải nén data.rar và đưa toàn bộ dữ liệu vừa tải về vào thư mục data

Cấu trúc thư mục sẽ là:
data
    apple_pie
    baklava
    ...
    xoi

## Tải Model đã huấn luyện (128x128)

Tạo một thư mục tên: models

Tải model từ link này: https://drive.google.com/file/d/1r35daQzz0DcSioEamB5cAhkm2-MGYghN/view

Đưa file vừa tài vào trong thư mục models

Cấu trúc thư mục sẽ là:
models
    train_features.pkl

# Hướng dẫn train lại Model mới dựa trên dữ liệu

Muốn train lại model mở tập tin train-features.ipynb để xem code train

Khi train lại model với ảnh kích thước khác 128x128 thì 

Sửa code trong file app.py

``img_width, img_height = 128, 128``


