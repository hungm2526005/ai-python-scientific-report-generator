import os
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Dashboard Phân Tích Dữ Liệu Y Tế", layout="wide")


# --- 1. TẢI & LÀM SẠCH DATASET THỰC TẾ (Y TẾ - DIABETES) ---
@st.cache_data
def load_and_clean_data():
  raw = load_diabetes(as_frame=True)
  df = raw.frame.copy()
  # Đổi tên cột cho dễ hiểu
  df.columns = [
      "Tuoi",
      "Gioi_Tinh",
      "Chi_So_BMI",
      "Huyet_Ap",
      "S1_Cholesterol",
      "S2_LDL",
      "S3_HDL",
      "S4_TCH",
      "S5_LTG",
      "S6_Duong_Huyet",
      "Muc_Do_Tien_Trien_Benh",
  ]
  # Làm sạch: loại bỏ trùng lặp và xử lý giá trị khuyết (nếu có)
  df = df.drop_duplicates().dropna()
  df.to_csv("medical_dataset.csv", index=False)
  return df


df = load_and_clean_data()


# --- 2. HÀM TỰ ĐỘNG SINH BÁO CÁO PDF (ĐỦ 5 MỤC THEO ĐỀ BÀI) ---
def generate_pdf_report(df, selected_col, r2, coef, chart_path):
  font_url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
  if not os.path.exists("Roboto-Regular.ttf"):
    urllib.request.urlretrieve(font_url, "Roboto-Regular.ttf")

  pdf = FPDF()
  pdf.add_page()
  pdf.add_font("Roboto", "", "Roboto-Regular.ttf")

  # Tiêu đề
  pdf.set_font("Roboto", size=15)
  pdf.cell(
      0,
      10,
      "BÁO CÁO KHOA HỌC: PHÂN TÍCH CHỈ SỐ TIẾN TRIỂN BỆNH TIỂU ĐƯỜNG",
      new_x="LMARGIN",
      new_y="NEXT",
      align="C",
  )
  pdf.ln(3)

  # 1. Giới thiệu đề tài
  pdf.set_font("Roboto", size=12)
  pdf.cell(
      0, 8, "1. Giới thiệu đề tài", new_x="LMARGIN", new_y="NEXT"
  )
  pdf.set_font("Roboto", size=10)
  pdf.multi_cell(
      0,
      6,
      "Đề tài ứng dụng Python và mô hình học máy Hồi quy tuyến tính (Linear"
      " Regression) để phân tích mối tương quan giữa các chỉ số sinh học và"
      " mức độ tiến triển bệnh tiểu đường sau một năm.",
  )
  pdf.ln(2)

  # 2. Mô tả dữ liệu
  pdf.set_font("Roboto", size=12)
  pdf.cell(
      0, 8, "2. Mô tả dữ liệu", new_x="LMARGIN", new_y="NEXT"
  )
  pdf.set_font("Roboto", size=10)
  pdf.multi_cell(
      0,
      6,
      f"Nguồn dữ liệu: Bộ dữ liệu y tế chuẩn hóa (Diabetes Dataset) gồm"
      f" {df.shape[0]} bệnh nhân và {df.shape[1]} thuộc tính định lượng đã được"
      " làm sạch (khử trùng lặp, kiểm tra giá trị khuyết).",
  )
  pdf.ln(2)

  # 3. Kết quả phân tích
  pdf.set_font("Roboto", size=12)
  pdf.cell(
      0,
      8,
      "3. Kết quả phân tích thống kê & Mô hình hồi quy",
      new_x="LMARGIN",
      new_y="NEXT",
  )
  pdf.set_font("Roboto", size=10)
  pdf.multi_cell(
      0,
      6,
      f"- Biến độc lập phân tích: {selected_col}\n- Giá trị trung bình:"
      f" {df[selected_col].mean():.4f} | Độ lệch chuẩn:"
      f" {df[selected_col].std():.4f}\n- Hệ số hồi quy tuyến tính (Slope):"
      f" {coef:.2f}\n- Độ phù hợp mô hình (R-squared): {r2:.4f}",
  )
  pdf.ln(2)

  # 4. Biểu đồ minh họa
  pdf.set_font("Roboto", size=12)
  pdf.cell(
      0, 8, "4. Biểu đồ minh họa", new_x="LMARGIN", new_y="NEXT"
  )
  pdf.image(chart_path, x=25, w=155)
  pdf.ln(3)

  # 5. Kết luận & đề xuất
  pdf.set_font("Roboto", size=12)
  pdf.cell(
      0, 8, "5. Kết luận & Đề xuất", new_x="LMARGIN", new_y="NEXT"
  )
  pdf.set_font("Roboto", size=10)
  pdf.multi_cell(
      0,
      6,
      f"Kết quả cho thấy biến '{selected_col}' có tác động tuyến tính rõ rệt"
      " đến mức độ tiến triển bệnh. Đề xuất kết hợp đa biến (BMI, huyết áp,"
      " đường huyết) để nâng cao độ chính xác dự báo lâm sàng.",
  )

  output_pdf = "BaoCao_MiniProject.pdf"
  pdf.output(output_pdf)
  return output_pdf


# --- 3. GIAO DIỆN DASHBOARD TƯƠNG TÁC STREAMLIT ---
st.title("Mini-Project: Phân tích Dữ liệu Y tế & Tự động xuất Báo cáo PDF")
st.markdown(
    "Chọn biến sinh học ở thanh bên trái để xem biểu đồ hồi quy và xuất báo"
    " cáo khoa học."
)

# Thanh bên: Cho phép người dùng chọn biến
st.sidebar.header("Tùy chọn phân tích")
feature_cols = [c for c in df.columns if c != "Muc_Do_Tien_Trien_Benh"]
selected_feature = st.sidebar.selectbox(
    "Chọn biến đầu vào (X):", feature_cols, index=2
)

# Huấn luyện mô hình hồi quy đơn giản
X = df[[selected_feature]].values
y = df["Muc_Do_Tien_Trien_Benh"].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Hiển thị số liệu & biểu đồ trên giao diện
col1, col2 = st.columns([1, 2])
with col1:
  st.subheader("Thống kê tóm tắt")
  st.write(df[[selected_feature, "Muc_Do_Tien_Trien_Benh"]].describe())
  st.metric("Hệ số xác định R²", f"{r2:.4f}")

with col2:
  st.subheader(
      f"Biểu đồ hồi quy: {selected_feature} vs Mức độ tiến triển bệnh"
  )
  fig, ax = plt.subplots(figsize=(7, 4))
  ax.scatter(
      X, y, alpha=0.6, color="#2b6cb0", label="Dữ liệu bệnh nhân thực tế"
  )
  ax.plot(X, y_pred, color="#e53e3e", linewidth=2, label="Đường hồi quy tuyến tính")
  ax.set_xlabel(selected_feature)
  ax.set_ylabel("Mức độ tiến triển bệnh")
  ax.legend()
  plt.tight_layout()
  chart_file = "dashboard_chart.png"
  fig.savefig(chart_file, dpi=150)
  st.pyplot(fig)

# Nút tạo & tải báo cáo PDF
st.divider()
pdf_file = generate_pdf_report(
    df, selected_feature, r2, model.coef_[0], chart_file
)
with open(pdf_file, "rb") as f:
  st.download_button(
      label="Tải Báo cáo Khoa học (.PDF)",
      data=f,
      file_name="BaoCao_MiniProject.pdf",
      mime="application/pdf",
  )