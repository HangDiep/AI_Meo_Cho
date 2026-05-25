tạo môt trường ảo python -m venv venv
Kích hoạt môi trường
Windows:
venv\Scripts\activate 
hoặc source venv/Scripts/activate

Nếu thành công sẽ hiện:

(venv)
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install MediaPipe 
# pip install torch torchvision


# Kích hoạt venv
venv\Scripts\activate

# 1. Phân tích dataset
python src/data_preprocessing.py

# 2. Huấn luyện (2 pha: freeze → fine-tune)
python src/train.py

# 3. Đánh giá chi tiết  
python src/evaluate.py

# 4. Webcam realtime
python src/realtime_detect.py

python app.py # để chạy ra hẳn giao diện 
python -m pip install flask