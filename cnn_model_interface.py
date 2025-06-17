import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load mô hình đã huấn luyện (.h5)
def load_my_cnn_model(model_path="pneumonia_model.h5"):
    try:
        model = load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# Dự đoán nhãn ảnh (image: kiểu PIL.Image)
def predict_image_with_model(model, image):
    try:
        # Resize về kích thước phù hợp với mô hình
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = img_array / 255.0  # Chuẩn hóa
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

        prediction = model.predict(img_array)[0][0]
        label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return {
            "label": label,
            "confidence": round(float(confidence), 4)
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            "label": "ERROR",
            "confidence": 0.0
        }