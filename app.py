import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
import requests
from bs4 import BeautifulSoup

# Streamlit configuration
st.set_page_config(page_title="Labubu Trend Analysis", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Crawl Dữ Liệu", "Dự Đoán Xu Hướng"],
        icons=["cloud-download", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# Function for crawling data from Shopee
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# Hàm crawl dữ liệu từ Shopee
def crawl_shopee(keyword="labubu", max_pages=1):
    options = Options()
    options.add_argument("--headless")  # Chạy trình duyệt ở chế độ không hiển thị
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service("path/to/chromedriver")  # Thay đường dẫn tới chromedriver của bạn

    driver = webdriver.Chrome(service=service, options=options)
    product_data = []

    try:
        for page in range(max_pages):
            url = f"https://shopee.vn/search?keyword={keyword}&page={page}"
            driver.get(url)
            time.sleep(5)  # Chờ trang load xong

            # Tìm tất cả các sản phẩm
            items = driver.find_elements(By.CLASS_NAME, "shopee-search-item-result__item")
            for item in items:
                try:
                    title = item.find_element(By.CLASS_NAME, "_1NoI8_._16BAGk").text
                    sales = item.find_element(By.CLASS_NAME, "_18SLBt").text if item.find_elements(By.CLASS_NAME, "_18SLBt") else "0"
                    product_data.append({"Product": title, "Sales": sales})
                except Exception as e:
                    continue
    finally:
        driver.quit()

    return pd.DataFrame(product_data)

# Thử nghiệm hàm crawl
data = crawl_shopee(keyword="labubu", max_pages=1)
print(data)

# Mock function for social media data
def crawl_social_media(keyword="labubu", max_posts=50):
    # Return mock data for now
    return pd.DataFrame({
        "Post": ["Labubu is amazing!", "Best gift for kids!", "Limited Labubu stocks!"],
        "Likes": [150, 230, 300],
        "Comments": [20, 35, 50]
    })

# Page 1: Crawl Data
if selected == "Crawl Dữ Liệu":
    st.title("🛒 Crawl Dữ Liệu Từ Shopee & MXH")

    # Shopee Crawl
    st.subheader("🔗 Crawl từ Shopee")
    keyword = st.text_input("Nhập từ khóa tìm kiếm (ví dụ: labubu):", value="labubu")
    max_pages = st.slider("Số trang cần crawl:", 1, 5, 1)

    if st.button("Crawl Shopee"):
        shopee_data = crawl_shopee(keyword, max_pages)
        st.write(f"Kết quả crawl từ Shopee ({len(shopee_data)} sản phẩm):")
        st.dataframe(shopee_data)

    # Social Media Crawl
    st.subheader("💬 Crawl từ Mạng Xã Hội")
    if st.button("Crawl Mạng Xã Hội"):
        social_data = crawl_social_media(keyword=keyword, max_posts=50)
        st.write(f"Kết quả crawl từ mạng xã hội ({len(social_data)} bài đăng):")
        st.dataframe(social_data)

# Page 2: Dự Đoán Xu Hướng
if selected == "Dự Đoán Xu Hướng":
    st.title("📈 Dự Đoán Xu Hướng Sản Phẩm: Labubu Doll")

    # Sample dataset
    data = {
        "Date": ["2024-12-01", "2024-12-02", "2024-12-03", "2024-12-04", "2024-12-05"],
        "Sales": [150, 180, 170, 200, 210],  # Lượt bán
        "Posts": [10, 15, 12, 18, 20],       # Số bài đăng
        "Interactions": [2000, 2500, 2200, 3000, 3500]  # Lượt tương tác
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = (df["Date"] - df["Date"].min()).dt.days

    # Display dataset
    st.subheader("🔍 Dữ Liệu Mẫu")
    st.dataframe(df)

    # Plot trends
    st.subheader("📊 Biểu Đồ Xu Hướng Hiện Tại")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Sales"], marker='o', label="Sales (TMĐT)")
    plt.plot(df["Date"], df["Posts"], marker='o', label="Posts (MXH)")
    plt.plot(df["Date"], df["Interactions"], marker='o', label="Interactions (MXH)")
    plt.title("Xu Hướng Sản Phẩm: Labubu Doll")
    plt.xlabel("Date")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Prediction with Linear Regression
    X = df[["Day"]].values
    y_sales = df["Sales"].values
    model_sales = LinearRegression()
    model_sales.fit(X, y_sales)
    future_days = np.array([[i] for i in range(6, 11)])
    sales_predictions = model_sales.predict(future_days)

    # Display predictions
    future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=5)
    predicted_data = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": sales_predictions.astype(int)
    })

    st.subheader("📅 Dự Đoán Doanh Số Trong 5 Ngày Tiếp Theo")
    st.dataframe(predicted_data)

    # Plot predictions
    st.subheader("🔮 Biểu Đồ Dự Đoán Doanh Số")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Sales"], marker='o', label="Actual Sales")
    plt.plot(future_dates, sales_predictions, marker='o', linestyle="--", label="Predicted Sales")
    plt.title("Dự Đoán Doanh Số: Labubu Doll")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
