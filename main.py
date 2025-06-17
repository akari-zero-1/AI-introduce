# main.py

import streamlit as st
from PIL import Image
import io
import time

from cnn_model_interface import load_my_cnn_model, predict_image_with_model

# Cấu hình giao diện ứng dụng
st.set_page_config(
    page_title="Ứng dụng kiểm thử Model CNN",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="auto"
)

# Tiêu đề và mô tả
st.title("📸 Ứng dụng kiểm thử Model CNN")
st.markdown("""
Chào mừng bạn đến với ứng dụng kiểm thử model CNN!
Tải lên một file ảnh để model của bạn dự đoán nội dung của nó.
""")

# Tải mô hình CNN khi ứng dụng khởi động
model = load_my_cnn_model()

if model is None:
    st.error("❌ Model CNN không thể được tải. Vui lòng kiểm tra file `cnn_model_interface.py` và đường dẫn model (`model.h5`).")
    st.stop()  # Dừng ứng dụng nếu model không tải được

st.markdown("---")

# Bộ tải file ảnh
uploaded_file = st.file_uploader(
    "**Bước 1:** Chọn một ảnh để kiểm tra...",
    type=["jpg", "jpeg", "png"],
    help="Chỉ chấp nhận các định dạng ảnh JPG, JPEG, PNG."
)

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    try:
        # Đọc file ảnh dưới dạng bytes và mở bằng Pillow
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        st.subheader("Ảnh đã tải lên:")
        st.image(image, caption='Ảnh của bạn.', use_column_width=True)
        st.success("Ảnh đã được tải lên thành công!")

        st.markdown("---")

        # Nút để bắt đầu dự đoán
        st.subheader("**Bước 2:** Nhấn nút để dự đoán")
        if st.button("🔍 Dự đoán ảnh này", key="predict_button", help="Click để chạy model CNN trên ảnh đã tải."):
            st.info("⏳ Đang xử lý và dự đoán... Vui lòng chờ.")
            with st.spinner("Đang chạy model CNN..."):
                time.sleep(1)  # giả lập xử lý chờ
                prediction_result = predict_image_with_model(model, image)

            st.subheader("🎯 Kết quả dự đoán:")
            st.success(f"**{prediction_result}**")
            st.balloons()  # Hiệu ứng khi có kết quả
            st.markdown("---")
            st.info("Bạn có thể tải lên ảnh khác để tiếp tục kiểm tra.")
        else:
            st.info("Nhấn 'Dự đoán ảnh này' để xem kết quả từ model của bạn.")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý ảnh: {e}")
        st.warning("Vui lòng đảm bảo file bạn tải lên là một ảnh hợp lệ.")
else:
    st.info("Hãy tải lên một file ảnh (.jpg, .jpeg, .png) để bắt đầu kiểm thử.")

st.markdown("---")
st.write("Được xây dựng với ❤️ bằng Streamlit.")
