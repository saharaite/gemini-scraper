# تحلیلگر هوشمند اخبار با Gemini

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)

یک ابزار هوشمند تحت وب برای ساخت خودکار اسکرپر (Scraper) وب‌سایت‌های خبری و تحلیل محتوای آن‌ها با استفاده از Google Gemini. این برنامه قادر است با تحلیل ساختار یک سایت، کد پایتون لازم برای استخراج اخبار را تولید کرده و به کاربران اجازه پرسش و پاسخ از محتوای استخراج شده را بدهد.

(./assets/demo.png)


## ✨ ویژگی‌های کلیدی

- **ساخت اسکرپر با هوش مصنوعی**: با استفاده از مدل `gemini-2.5-pro`، برنامه به صورت خودکار ساختار HTML یک وب‌سایت را تحلیل کرده و کد پایتون لازم برای استخراج اطلاعات را تولید می‌کند.
- **ذخیره‌سازی و پایداری اسکرپرها**: کدهای تولید شده برای هر دامنه در یک فایل `scrapers.json` ذخیره می‌شوند. این ویژگی باعث می‌شود در مراجعات بعدی، فرآیند استخراج داده سریع‌تر و ارزان‌تر انجام شود.
- **دو روش دریافت محتوا**:
    - **ساده (Requests)**: برای وب‌سایت‌های استاتیک و سریع.
    - **پیشرفته (Selenium)**: برای وب‌سایت‌های داینامیک که محتوای آن‌ها با جاوااسکریپت بارگذاری می‌شود.
- **پرسش و پاسخ تعاملی (RAG)**: پس از استخراج اخبار، محتوای متنی به کمک مدل `text-embedding-004` به یک پایگاه دانش برداری تبدیل می‌شود. کاربران می‌توانند سوالات خود را در یک رابط چت بپرسند و پاسخ‌های دقیق مبتنی بر محتوای اخبار دریافت کنند.
- **ویرایشگر کد و تایید دستی**: کاربر می‌تواند کد تولید شده توسط هوش مصنوعی را قبل از اجرا مشاهده، ویرایش و تایید کند.
- **مدیریت امن کلید API**: برنامه ابتدا کلید Gemini API را از فایل `secrets.toml` (روش استاندارد Streamlit) می‌خواند و در غیر این صورت به کاربر اجازه ورود موقت آن را می‌دهد.

## 🛠️ نصب و راه‌اندازی

برای اجرای این پروژه به صورت محلی، مراحل زیر را دنبال کنید.

### ۱. پیش‌نیازها

- پایتون نسخه 3.9 یا بالاتر
- یک کلید Google Gemini API

### ۲. کلون کردن ریپازیتوری

```bash
git clone [https://github.com/saharaite/gemini-scraper.git](https://github.com/saharaite/gemini-scraper.git)
cd gemini-scraper
```

### ۳. ایجاد و فعال‌سازی محیط مجازی (توصیه می‌شود)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### ۴. نصب کتابخانه‌های مورد نیاز

یک فایل با نام `requirements.txt` در ریشه پروژه ایجاد کرده و محتوای زیر را در آن قرار دهید:

```txt
streamlit
requests
beautifulsoup4
google-generativeai
selenium
webdriver-manager-chrome
numpy
lxml
```

سپس دستور زیر را برای نصب آن‌ها اجرا کنید:

```bash
pip install -r requirements.txt
```

### ۵. پیکربندی کلید API

بهترین روش برای مدیریت کلید API، استفاده از قابلیت `secrets` در Streamlit است.

1.  یک پوشه با نام `.streamlit` در ریشه پروژه خود ایجاد کنید.
2.  داخل این پوشه، یک فایل با نام `secrets.toml` بسازید.
3.  کلید API خود را به شکل زیر در این فایل قرار دهید:

```toml
# .streamlit/secrets.toml

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

## 🚀 اجرا

پس از انجام مراحل نصب، با اجرای دستور زیر در ترمینال، برنامه را اجرا کنید:

```bash
streamlit run gemini_scraper_app_0.14.py
```

سپس مرورگر خود را باز کرده و به آدرس `http://localhost:8501` مراجعه کنید.

## 📄 لایسنس

این پروژه تحت لایسنس **MIT** منتشر شده است. برای اطلاعات بیشتر فایل `LICENSE` را مشاهده کنید.