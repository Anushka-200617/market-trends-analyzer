from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import random
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_healthcare_sources():
    return {
        "The Startup Journal": "https://www.thestartupjournal.com/",
        "FierceHealthcare": "https://www.fiercehealthcare.com/",
        "HealthcareITNews": "https://www.healthcareitnews.com/",
        "MedCityNews": "https://medcitynews.com/",
        "MobiHealthNews": "https://www.mobihealthnews.com/",
        "BeckersHospitalReview": "https://www.beckershospitalreview.com/",
        "HITConsultant": "https://hitconsultant.net/",
        "MedTechDive": "https://www.medtechdive.com/",
        "HealthcareFinanceNews": "https://www.healthcarefinancenews.com/",
        "ModernHealthcare": "https://www.modernhealthcare.com/",
        "DigitalHealthBiz": "https://digitalhealthbiz.com/",
        "BiopharmaDive": "https://www.biopharmadive.com/",
        "Digital Health Global": "https://www.digitalhealthglobal.com/",
        "Healthcare Innovation": "https://www.hcinnovationgroup.com/"
    }

def get_finance_sources():
    return {
        "The Startup Journal": "https://www.thestartupjournal.com/",
        "Finextra": "https://www.finextra.com/",
        "FintechFutures": "https://www.fintechfutures.com/",
        "TechCrunch Fintech": "https://techcrunch.com/category/fintech/",
        "American Banker": "https://www.americanbanker.com/",
        "PYMNTS": "https://www.pymnts.com/",
        "FintechNews": "https://fintechnews.org/",
        "The Financial Brand": "https://thefinancialbrand.com/",
        "BankingDive": "https://www.bankingdive.com/",
        "Crowdfund Insider": "https://www.crowdfundinsider.com/",
        "PaymentsDive": "https://www.paymentsdive.com/",
        "BankInnovation": "https://bankinnovation.net/",
        "Fintech Insight": "https://fintechinsight.com/",
        "FinTech Magazine": "https://fintechmagazine.com/"
    }

def scrape_website(website):
    logger.info(f"Connecting to local Chrome WebDriver for {website}...")
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
    except Exception as e:
        logger.error(f"Failed to initialize Chrome WebDriver: {str(e)}")
        return f"ERROR: Failed to initialize Chrome WebDriver: {str(e)}"
    
    try:
        driver.set_window_size(1920, 1080)
        driver.get(website)
        logger.info(f"Navigated to {website}! Waiting for dynamic content to load...")
        time.sleep(3)
        
        total_height = driver.execute_script("return document.body.scrollHeight")
        for i in range(3):
            driver.execute_script(f"window.scrollTo(0, {total_height * (i+1) / 4});")
            time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        
        try:
            cookie_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree') or contains(text(), 'Continue')]")
            if cookie_buttons:
                cookie_buttons[0].click()
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Error handling cookie banner: {e}")
        
        logger.info("Scraping page content...")
        html = driver.page_source
        return html
    except Exception as e:
        logger.error(f"Error scraping {website}: {str(e)}")
        return f"ERROR: Error scraping {website}: {str(e)}"
    finally:
        driver.quit()

def extract_body_content(html_content):
    if isinstance(html_content, str) and html_content.startswith("ERROR:"):
        return html_content
    
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        content_elements = [
            soup.find_all('article'),
            soup.find_all('div', {'class': ['post', 'article', 'entry', 'content', 'news-item']}),
            soup.find_all('div', {'id': ['content', 'main-content', 'article-content', 'post-content']}),
            soup.find_all('main'),
            soup.find_all('section', {'class': ['content', 'main', 'articles', 'news']}),
            soup.find_all('h2'),
            soup.find_all('h3')
        ]
        
        for elements in content_elements:
            if elements:
                content = "\n\n".join([element.get_text(separator="\n") for element in elements])
                if content and len(content) > 500:
                    return content
        
        body_content = soup.body
        if body_content:
            return body_content.get_text(separator="\n")
        
        return "No content found on the page."
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return f"ERROR: Error extracting content: {str(e)}"

def clean_body_content(body_content):
    if isinstance(body_content, str) and body_content.startswith("ERROR:"):
        return body_content
    
    try:
        if isinstance(body_content, str):
            soup = BeautifulSoup(f"<div>{body_content}</div>", "html.parser")
        else:
            soup = body_content
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'meta', 'form', 'iframe', 'noscript']):
            element.extract()
        
        content = soup.get_text(separator="\n") if isinstance(body_content, str) else soup
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        cleaned_content = "\n".join(lines)
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        cleaned_content = re.sub(r'(subscribe to our newsletter|sign up for our newsletter|subscribe for updates|email address).*', '', cleaned_content, flags=re.IGNORECASE)
        
        return cleaned_content
    except Exception as e:
        logger.error(f"Error cleaning content: {str(e)}")
        return f"ERROR: Error cleaning content: {str(e)}"

def split_dom_content(dom_content, max_length=8000, chunk_size=None):
    if chunk_size is not None:
        max_length = chunk_size
    
    if isinstance(dom_content, str) and dom_content.startswith("ERROR:"):
        return [dom_content]
    
    try:
        if len(dom_content) <= max_length:
            return [dom_content]
        
        chunks = []
        paragraphs = dom_content.split("\n\n")
        current_chunk = ""
        
        for paragraph in paragraphs:
            if not isinstance(paragraph, str):
                logger.warning(f"Skipping non-string paragraph: {type(paragraph)}")
                continue
            
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(paragraph) > max_length:
                    sentences = paragraph.split(". ")
                    current_chunk = ""
                    for sentence in sentences:
                        if not isinstance(sentence, str):
                            logger.warning(f"Skipping non-string sentence: {type(sentence)}")
                            continue
                        if len(current_chunk) + len(sentence) + 2 <= max_length:
                            if current_chunk:
                                current_chunk += ". "
                            current_chunk += sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk + ".")
                            if len(sentence) > max_length:
                                for i in range(0, len(sentence), max_length):
                                    chunks.append(sentence[i:i+max_length])
                                current_chunk = ""
                            else:
                                current_chunk = sentence
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Error splitting content: {str(e)}")
        return [f"ERROR: Error splitting content: {str(e)}"]