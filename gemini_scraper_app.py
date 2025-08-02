# gemini_scraper_app_final.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
from urllib.parse import urlparse
import numpy as np
import time
import os

# تلاش برای وارد کردن Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# --- بخش تنظیمات و پیکربندی ---
st.set_page_config(layout="wide", page_title="تحلیلگر هوشمند اخبار با Gemini")
SCRAPER_FILE = 'scrapers.json'

# --- استایل CSS با فونت شبنم از CDN معتبر ---
st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/shabnam-font/5.0.1/font-face.css');
    
    html, body, [class*="css"]  {
        direction: rtl;
        font-family: 'Shabnam', sans-serif !important;
        font-weight: normal;
        text-align: right;
    }
    h1, h2, h3, h4, h5, h6, strong {
        font-weight: bold !important;
    }
    textarea[key="editor"], code, .stDataFrame {
        direction: ltr !important;
        text-align: left !important;
        font-family: monospace !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] p,
    .stChatMessage div[data-testid="stMarkdownContainer"] ul,
    .stChatMessage div[data-testid="stMarkdownContainer"] ol {
        text-align: right !important;
        direction: rtl !important;
    }
    .stRadio > div {
        flex-direction: row-reverse;
        justify-content: flex-end;
    }
    .stRadio label {
        margin-left: 0.5rem;
        margin-right: 0;
    }
    [data-testid="stSidebar"] {
        text-align: right !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- مدیریت وضعیت برنامه ---
if 'gemini_model' not in st.session_state: st.session_state.gemini_model = None
if 'messages' not in st.session_state: st.session_state.messages = []
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'scraped_data' not in st.session_state: st.session_state.scraped_data = None
if 'html_content' not in st.session_state: st.session_state.html_content = None
if 'current_domain' not in st.session_state: st.session_state.current_domain = None
if 'scraper_code_to_approve' not in st.session_state: st.session_state.scraper_code_to_approve = None


# --- بخش جدید: مدیریت هوشمند کلید API ---
st.sidebar.title("تنظیمات")
st.sidebar.header("پیکربندی کلید Gemini API")

def configure_api(api_key: str) -> bool:
    """پیکربندی API با کلید ارائه شده و برگرداندن وضعیت موفقیت."""
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        st.session_state.gemini_model = model
        return True
    except Exception as e:
        st.sidebar.error(f"کلید API نامعتبر است: {e}")
        st.session_state.gemini_model = None
        return False

# اولویت اول: تلاش برای خواندن کلید از st.secrets
if st.session_state.gemini_model is None:
    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY")
        if secrets_key and configure_api(secrets_key):
            st.sidebar.success("کلید API از secrets.toml با موفقیت بارگذاری شد. ✅")
    except (FileNotFoundError, KeyError):
        # این خطا طبیعی است اگر فایل secrets وجود نداشته باشد
        pass

# اگر کلید از secrets بارگذاری نشد، به کاربر اجازه ورود می‌دهیم
if st.session_state.gemini_model is None:
    st.sidebar.info("کلیدی در فایل secrets.toml یافت نشد. لطفاً کلید خود را در زیر وارد کنید.")
    user_api_key = st.sidebar.text_input(
        "کلید API Gemini خود را اینجا وارد کنید:",
        type="password",
        help="کلید شما فقط در این جلسه استفاده می‌شود و جایی ذخیره نخواهد شد."
    )
    if st.sidebar.button("تایید و ذخیره کلید"):
        if configure_api(user_api_key):
            st.sidebar.success("کلید API با موفقیت تایید و ذخیره شد.")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.warning("لطفاً یک کلید معتبر وارد کنید.")

# --- توابع مدیریت پایگاه داده اسکرپرها ---
def load_scrapers() -> dict:
    if not os.path.exists(SCRAPER_FILE): return {}
    try:
        with open(SCRAPER_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_scraper(domain: str, code: str):
    scrapers = load_scrapers()
    scrapers[domain] = code
    with open(SCRAPER_FILE, 'w', encoding='utf-8') as f:
        json.dump(scrapers, f, indent=4, ensure_ascii=False)

# --- توابع اصلی برنامه ---
def fetch_html_simple(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        st.session_state.html_content = response.text
    except requests.exceptions.RequestException as e:
        st.error(f"[Requests] خطا در دریافت اطلاعات از سایت: {e}")
        st.session_state.html_content = None

def fetch_html_advanced(url: str):
    if not SELENIUM_AVAILABLE:
        st.error("کتابخانه Selenium نصب نشده است.")
        st.session_state.html_content = None
        return
    driver = None
    status = st.status("در حال اجرای مرورگر خودکار (Selenium)...", expanded=True)
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        status.write("نصب یا به‌روزرسانی درایور مرورگر...")
        service = ChromeService(ChromeDriverManager().install())
        status.write(f"باز کردن آدرس: {url}")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.get(url)
        status.write("منتظر ماندن برای بارگذاری محتوای دینامیک...")
        time.sleep(7)
        html_content = driver.page_source
        status.update(label="مرورگر خودکار با موفقیت کار خود را تمام کرد.", state="complete")
        st.session_state.html_content = html_content
    except Exception as e:
        st.error(f"[Selenium] خطا در اجرای مرورگر خودکار: {e}")
        status.update(label="خطا در اجرای Selenium.", state="error")
        st.session_state.html_content = None
    finally:
        if driver:
            driver.quit()

def generate_scraper_with_gemini(html_content: str, site_url: str) -> str | None:
    generation_model = st.session_state.gemini_model
    soup = BeautifulSoup(html_content, 'lxml')
    body_content_sample = str(soup.body.prettify())[:100000] if soup.body else ""
    status = st.status("در حال اجرای تحلیل متخصصانه توسط Gemini...", expanded=True)
    try:
        status.update(label="مرحله ۱: نقشه‌برداری از ساختار کلی سایت...")
        prompt_step_1 = f"""Analyze the following HTML from {site_url}. Identify CSS selectors for the main containers that each hold a single news article. Provide your answer as a JSON object with a key "selectors" which is a list of strings. Example: {{"selectors": ["div.story-wrapper", "a.news-card"]}} \nHTML:\n```html\n{body_content_sample}\n```"""
        response_step_1 = generation_model.generate_content(prompt_step_1)
        selectors_json_str = response_step_1.text.strip().replace("```json", "").replace("```", "").strip()
        candidate_selectors = json.loads(selectors_json_str).get("selectors", [])
        if not candidate_selectors: return None
        status.write(f"مرحله ۱: سلکتورهای کاندید پیدا شدند: `{'`, `'.join(candidate_selectors)}`")
        
        status.update(label="مرحله ۲: تحلیل عمیق و مقایسه‌ای نمونه‌ها...")
        samples_html = ""
        found_articles = soup.select(", ".join(candidate_selectors), limit=5)
        if not found_articles: return None
        for article in found_articles: samples_html += f"<!-- Sample Article Container -->\n{article.prettify()}\n\n"
        prompt_step_2 = f"""Based on the HTML samples, determine reliable CSS selectors to extract title, link, and description. Provide analysis as a clean JSON object with keys: "best_article_selector", "title_selector", "link_selector", "description_selector". If the link is on the main container itself, use "self".\nHTML SAMPLES:\n```html\n{samples_html}\n```"""
        response_step_2 = generation_model.generate_content(prompt_step_2)
        analysis = json.loads(response_step_2.text.strip().replace("```json", "").replace("```", "").strip())
        status.write("مرحله ۲: تحلیل ساختار داخلی مقالات با موفقیت انجام شد.")

        status.update(label="مرحله ۳: تولید کد نهایی...")
        prompt_step_3 = f"""Write a Python function `scrape_news(html_content)` using this analysis: `{json.dumps(analysis)}`. Base URL: `{site_url}`. CRITICAL: Must include `from bs4 import BeautifulSoup` and `from urllib.parse import urljoin`. Use `try-except Exception: continue`. Return ONLY Python code."""
        response_step_3 = generation_model.generate_content(prompt_step_3)
        final_code = response_step_3.text.strip().replace("```python", "").replace("```", "").strip()
        status.update(label="تحلیل متخصصانه و تولید کد با موفقیت به پایان رسید!", state="complete")
        return final_code
    except Exception as e:
        status.update(label=f"خطا در فرآیند تحلیل: {e}", state="error")
        st.error(f"خطا در فرآیند تحلیل چند مرحله‌ای: {e}")
        return None

def execute_scraper(html_content: str, scraper_code: str) -> list | None:
    try:
        execution_scope = {}
        exec(scraper_code, execution_scope)
        scraper_function = execution_scope.get('scrape_news')
        if callable(scraper_function):
            return scraper_function(html_content)
        st.error("تابع `scrape_news` در کد تولید شده یافت نشد.")
        return None
    except Exception as e:
        st.error(f"خطا در اجرای کد استخراج‌کننده: {e}")
        return None

def build_vector_store(scraped_data: list):
    st.session_state.vector_store = []
    if not scraped_data: return
    items_to_embed = [item for item in scraped_data if item.get('title') and item.get('link')]
    if not items_to_embed: return

    # --- اصلاح کلیدی برای رفع باگ لینک ---
    # لینک خبر را به متنی که embed می‌شود اضافه می‌کنیم تا در پایگاه دانش برای جستجو و پاسخگویی موجود باشد.
    contents = [f"عنوان: {item.get('title', '')}\nتوضیحات: {item.get('description', '')}\nمنبع (لینک): {item.get('link', '')}" for item in items_to_embed]
    
    with st.spinner(f"در حال آماده‌سازی دانش برای چت‌بات (پردازش {len(contents)} خبر)..."):
        try:
            BATCH_SIZE = 100
            all_embeddings = []
            for i in range(0, len(contents), BATCH_SIZE):
                batch_content = contents[i:i+BATCH_SIZE]
                embedding_response = genai.embed_content(model="models/text-embedding-004", content=batch_content, task_type="RETRIEVAL_DOCUMENT")
                all_embeddings.extend(embedding_response['embedding'])
                time.sleep(1) # برای جلوگیری از خطاهای rate limit
            
            for content, vector in zip(contents, all_embeddings):
                st.session_state.vector_store.append({"content": content, "vector": np.array(vector)})
            st.success(f"پایگاه دانش چت‌بات با **{len(st.session_state.vector_store)}** سند آماده شد.")
        except Exception as e:
            st.error(f"خطا در ساخت embedding: {e}")

def find_relevant_context(query: str, top_k: int = 7) -> str:
    if not st.session_state.vector_store: return ""
    try:
        query_embedding = genai.embed_content(model="models/text-embedding-004", content=query, task_type="RETRIEVAL_QUERY")['embedding']
        query_vector = np.array(query_embedding)
        vectors = np.array([item['vector'] for item in st.session_state.vector_store])
        
        # محاسبه شباهت کسینوسی
        dot_products = np.dot(vectors, query_vector)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        similarities = dot_products / norms
        
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        context = "### متون مرتبط از اخبار استخراج شده:\n\n"
        for index in top_k_indices:
            context += st.session_state.vector_store[index]['content'] + "\n---\n"
        return context
    except Exception as e:
        st.error(f"خطا در پیدا کردن متن مرتبط: {e}")
        return ""

# --- رابط کاربری اصلی Streamlit ---
st.title("تحلیلگر هوشمند و پایدار اخبار 🤖📝")

# بررسی وضعیت پیکربندی API قبل از نمایش رابط کاربری اصلی
if st.session_state.gemini_model is None:
    st.info("💡 برای شروع، لطفاً از منوی کناری (Sidebar) کلید Gemini API خود را وارد و تایید کنید.")
    st.warning("اگر فایل `.streamlit/secrets.toml` را با کلید خود ایجاد کنید، برنامه به صورت خودکار از آن استفاده خواهد کرد.")
else:
    st.markdown("این برنامه برای هر سایت، یک اسکرپر مخصوص با هوش مصنوعی می‌سازد و آن را ذخیره می‌کند تا در استفاده‌های بعدی سریع‌تر و ارزان‌تر عمل کند.")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("۱. مدیریت استخراج")
        url = st.text_input("آدرس وب‌سایت:", placeholder="https://www.varzesh3.com/")
        fetch_method = st.radio("روش دریافت:", ('ساده (Requests)', 'پیشرفته (Selenium)'), index=1, horizontal=True)
        force_regenerate = st.checkbox("ساخت مجدد اسکرپر (نادیده گرفتن کد ذخیره شده)")

        if st.button("🚀 شروع پردازش", type="primary", use_container_width=True):
            if url:
                # ریست کردن وضعیت برای پردازش جدید
                st.session_state.messages, st.session_state.vector_store = [], None
                st.session_state.scraped_data, st.session_state.scraper_code_to_approve = None, None
                
                domain = urlparse(url).netloc
                st.session_state.current_domain = domain
                scrapers = load_scrapers()
                fetch_html_advanced(url) if fetch_method == 'پیشرفته (Selenium)' else fetch_html_simple(url)

                if st.session_state.html_content:
                    st.success("✔️ محتوای HTML دریافت شد.")
                    if domain in scrapers and not force_regenerate:
                        st.info(f"✅ اسکرپر ذخیره شده برای `{domain}` اجرا می‌شود...")
                        scraper_code = scrapers[domain]
                        st.session_state.scraped_data = execute_scraper(st.session_state.html_content, scraper_code)
                    else:
                        st.info(f"در حال ساخت اسکرپر جدید برای `{domain}`...")
                        generated_code = generate_scraper_with_gemini(st.session_state.html_content, url)
                        if generated_code:
                            st.session_state.scraper_code_to_approve = generated_code
                        else:
                            st.error("تولید کد ناموفق بود. ممکن است ساختار سایت پیچیده باشد یا خطایی در API رخ داده باشد.")
                else:
                    st.error("دریافت HTML ناموفق بود.")
            else:
                st.warning("لطفاً آدرس را وارد کنید.")

        if st.session_state.get('scraper_code_to_approve'):
            st.subheader("۲. تایید و اصلاح اسکرپر")
            edited_code = st.text_area("کد اسکرپر:", value=st.session_state.scraper_code_to_approve, height=300, key="editor")
            c1, c2 = st.columns(2)
            if c1.button("✅ تست و اجرا", use_container_width=True):
                with st.spinner("در حال تست کد..."):
                    st.session_state.scraped_data = execute_scraper(st.session_state.html_content, edited_code)
            if c2.button("💾 ذخیره و آماده‌سازی", type="primary", use_container_width=True, disabled=not bool(st.session_state.scraped_data)):
                save_scraper(st.session_state.current_domain, edited_code)
                st.success(f"کد برای `{st.session_state.current_domain}` ذخیره شد.")
                build_vector_store(st.session_state.scraped_data)
                st.session_state.scraper_code_to_approve = None # پاک کردن کد از حالت تایید
                st.rerun()

        if st.session_state.get('scraped_data') is not None:
            st.subheader("نتایج استخراج شده")
            total_items = len(st.session_state.scraped_data)
            st.success(f"تعداد کل اخبار یافت شده: **{total_items}**")
            with st.expander(f"برای مشاهده {total_items} خبر کلیک کنید"):
                st.dataframe(st.session_state.scraped_data)
            if not st.session_state.get('vector_store'):
                if st.button("🧠 آماده‌سازی برای پرسش و پاسخ", use_container_width=True):
                    build_vector_store(st.session_state.scraped_data)
                    st.rerun()

    with col2:
        st.subheader("پرسش و پاسخ از محتوا")
        if st.session_state.get('vector_store'):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
            
            if prompt := st.chat_input("سؤال خود را در مورد اخبار بپرسید..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt, unsafe_allow_html=True)
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("در حال تحلیل سوال و یافتن پاسخ..."):
                        generation_model = st.session_state.gemini_model
                        
                        intent_prompt = f"""Classify the user's query into 'retrieval' or 'aggregation'.
                        - 'retrieval': Asks about specific content (e.g., "What did the coach say?").
                        - 'aggregation': Asks for a list, summary, or count (e.g., "List all headlines", "How many news items?").
                        Query: "{prompt}" -> Category:"""
                        intent_response = generation_model.generate_content(intent_prompt)
                        intent = intent_response.text.strip().lower()

                        full_response = ""
                        if "aggregation" in intent:
                            message_placeholder.info("درخواست تجمیعی شناسایی شد... در حال پردازش تمام داده‌ها.")
                            all_data = [f"عنوان: {item.get('title', 'بدون عنوان')}\nلینک: {item.get('link', 'بدون لینک')}" for item in st.session_state.scraped_data]
                            context = "\n".join(all_data)
                            total_count = len(all_data)
                            final_prompt = f"""You are an assistant. Answer the user's question in Persian based ONLY on the provided list of data. Create a clear, structured, and categorized summary of the news.
                            Data: Total articles: {total_count}\n{context}
                            User's Question: {prompt}"""
                        else: # retrieval
                            message_placeholder.info("درخواست جستجوی اطلاعات خاص شناسایی شد...")
                            context = find_relevant_context(prompt)
                            if not context.strip() or "###" not in context:
                                full_response = "متاسفانه نتوانستم اطلاعات مرتبطی در اخبار استخراج شده پیدا کنم."
                            else:
                                final_prompt = f"""You are a helpful AI assistant. Answer the user's question in Persian based ONLY on the provided context.
                                **CRITICAL: When you mention a news item, you MUST provide its source link, which is included in the context. Format the link as a clickable Markdown link, like this: [متن لینک](URL).**
                                If the answer isn't in the context, say so clearly.

                                Context:
                                {context}
                                
                                Question: {prompt}
                                """
                        
                        if not full_response: # اگر پاسخ از قبل مشخص نشده بود
                            response = generation_model.generate_content(final_prompt)
                            full_response = response.text

                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.info("ابتدا اطلاعات یک سایت را استخراج و سپس برای پرسش و پاسخ آماده کنید تا این بخش فعال شود.")