# Mini-Project: AI + Python Sinh Báo Cáo Khoa Học Tự Động từ Dữ Liệu Y Tế

## 1. Giới thiệu
Dự án xây dựng quy trình tự động hóa phân tích dữ liệu khoa học (bộ dữ liệu y tế Diabetes Dataset), trực quan hóa hồi quy tuyến tính, xuất báo cáo định dạng PDF chuẩn hóa và cung cấp giao diện Dashboard tương tác bằng Streamlit.

## 2. Cấu trúc thư mục
- `app.py`: Mã nguồn chính xử lý dữ liệu, huấn luyện mô hình hồi quy, sinh báo cáo PDF và chạy Dashboard Streamlit.
- `medical_dataset.csv`: Dữ liệu mẫu y tế đã làm sạch (442 bản ghi, 11 thuộc tính).
- `BaoCao_MiniProject.pdf`: Báo cáo khoa học được hệ thống tự động sinh ra.
- `requirements.txt`: Danh sách các thư viện Python cần thiết.
- `LICENSE`: Giấy phép sử dụng mã nguồn mở MIT.

## 3. Hướng dẫn cài đặt và chạy
```bash
pip install -r requirements.txt
streamlit run app.py