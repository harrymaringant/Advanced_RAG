import yaml
import time
from typing import List
from llama_index.core import Document 
import logging
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
import re

def fetch_html(url):
    """
    Fetch the HTML content of the given URL.
    """
    try:
        logging.info(f"Fetching HTML content from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response #.text
    except requests.RequestException as e:
        logging.error(f"Error fetching the URL {url}: {e}")
        return None

def get_context(url):
    """
    Extracts the text content from a given HTML section.

    Args:
        html: A BeautifulSoup element representing the HTML section.

    Returns:
        The extracted text content as a string.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html.text, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    context = soup.get_text(separator=" ")

    return context

def preprocess(text):
    # Remove Unicode private use area characters
    cleaned_text = text.replace('|', ' ').replace(':', ' ').replace('?',' ').replace('-',' ')
    cleaned_text = re.sub(r'[\ue000-\uf8ff]', ' ', cleaned_text)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    # Normalize whitespace but keep newlines intact
    # cleaned_text = re.sub(r' +', ' ', cleaned_text)  # Replace multiple spaces with a single space
    # cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)  # Normalize spaces after newlines
    cleaned_text = cleaned_text.strip()  # Remove leading and trailing spaces
    return cleaned_text.lower()

def clean_context(html_content):
    """
    Extracts the text content from a given HTML section.

    Args:
        html: A BeautifulSoup element representing the HTML section.

    Returns:
        The extracted text content as a string.
    """
    soup = BeautifulSoup(html_content, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text(separator=" ")

    # Handle plain text processing
    # formatted_text = re.sub(r"\s+", " ", text).strip()
    # cleaned_lines = [line.strip() for line in formatted_text.splitlines() if line.strip()]
    # cleaned_text = "\n".join(cleaned_lines)

    # Split the text into lines
    lines = text.split("\n")

    # Remove duplicate newlines
    cleaned_lines = []
    for line in lines:
        if line and line not in cleaned_lines:

            # # Remove extra spaces (including \xa0)
            line = re.sub(r"\s+", " ", line).strip()
            cleaned_lines.append(line)

    # Join the cleaned lines with a single newline
    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text

def process_webpages(config_file: str) -> List[Document]:
    """
    Processes webpages specified in a YAML configuration file by extracting text content and storing them as documents.

    Args:
        config_file: Path to the YAML configuration file.

    Returns:
        A list of Document objects containing the extracted text content.
    """
    documents = []

    config = yaml.safe_load(open(config_file, 'r'))

    for category, details in config.items():
        for detail_page in details['detail_pages']:
            webpage = detail_page['url']
            title = detail_page['title']
            
            if isinstance(webpage, str):  # If it's a URL
                html_content = get_context(webpage)  # Replace with your fetch function
            else:  # If it's already HTML content
                html_content = webpage

            if html_content:
                context = get_context(webpage)
                time.sleep(1)
                cleaned_context = clean_context(context)
                cleaned_text = preprocess(cleaned_context)
                
                if category == "Promo":
                    if webpage == 'https://www.banksinarmas.com/id/promosi':
                        final_text = extract_all_promo_text(cleaned_text)
                    elif webpage == 'https://www.banksinarmas.com/id/promosi/brastagi-supermarket':
                        final_text = "Promo brastagi supermarket saat ini ada di medan." + extract_detail_promo_text(cleaned_text)
                    else:
                        final_text = extract_detail_promo_text(cleaned_text)
                elif category == "Produk":
                    if webpage == 'https://www.banksinarmas.com/id/personal/produk/kartukredit':
                        final_text = "Terdapat dua jenis kartu kredit yaitu personal dan korporat. 1. personal : 1. silver : alfamart dan indigo, 2. platinum : platinum, dan 2. korporat: platinum." + extract_produk_text(cleaned_text)
                    else:
                        final_text = extract_produk_text(cleaned_text)
                # elif category == "Kurs Mata Uang":
                #     final_text = extract_kurs_text(cleaned_text)
                elif category == "Profil Bank Sinarmas":
                    final_text = extract_manajemen_text(cleaned_text)
                elif category == "Manajemen Bank Sinarmas":
                    final_text = "Manajemen Bank Sinarmas terdiri atas Komisaris dan Direksi." + extract_manajemen_text(cleaned_text)
                else:
                    continue

                final_text = "[page start]" + final_text + "[end of page]"
                
                
                documents.append(Document(text=final_text, 
                                          metadata={'title': title, 'category': category},
                                          metadata_seperator="::",
                                          metadata_template="{key}=>{value}",
                                          text_template="Metadata: {metadata_str}\n-----\nContent: {content}"))
    return documents

def extract_detail_promo_text(text):
    """
    Extracts the text content between specified keywords from a text.

    Args:
        text (str): The text content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    start_index = text.index("promosi") + len("promosi") + 1 # +1 for extra space

    # Find the ending index of the substring
    end_index = text.index("bagikan")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index -1] #-1 to delete extra space

    return extracted_text

def extract_all_promo_text(text):
    """
    Extracts the text content between specified keywords from a text.

    Args:
        text (str): The text content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    start_index = text.index("semua promo")

    # Find the ending index of the substring
    end_index = text.index("kantor pusat")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index]

    return extracted_text

def extract_kurs_text(text):
    """
    Extracts the text content between specified keywords from a text.

    Args:
        text (str): The text content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    start_index = text.rindex("personal perbankan bisnis") + len("personal perbankan bisnis") + 1

    # Find the ending index of the substring
    end_index = text.index("bagikan")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index]

    return extracted_text

def extract_produk_text(text):
    """
    Extracts the text content between specified keywords from a text.

    Args:
        text (str): The text content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    try:
        start_index = text.rindex("personal perbankan bisnis") + len("personal perbankan bisnis") + 1
    except:
        start_index = text.rindex("personal banking") + len("personal banking") + 1
        
    # Find the ending index of the substring
    end_index = text.index("kantor pusat")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index]

    return extracted_text

def extract_manajemen_text(text):
    """
    Extracts the text content between specified keywords from a text.

    Args:
        text (str): The text content as a string.

    Returns:
        str: The extracted text content.
    """
    # Find the starting index of the substring
    start_index = text.rindex("personal perbankan bisnis") + len("personal perbankan bisnis") + 1

    # Find the ending index of the substring
    end_index = text.index("kantor pusat")

    # Extract the substring using string slicing
    extracted_text = text[start_index:end_index]

    return extracted_text

def preprocess_input(input_text):
    replacements = {
        'kk': 'kartu kredit',
        'cc': 'kartu kredit',
        'bsim': 'bank sinarmas',
        'kpr': 'kredit pemilikan rumah',
        'kta': 'kredit tanpa agunan',
        'qr' : 'qris',
        'magal': 'magal korean',
        'prosedur': 'cara',
        'ketentuan': 'syarat',
        'ibu': 'bu',
        'mcd': 'mcdonald',
        'apply': 'daftar',
        'bikin': 'daftar',
        'buat': 'daftar'
        # 'manajemen': 'komisaris dan direktur',
        # 'stakeholder': 'komisaris dan direktur',
        # 'di medan': 'di kota medan'
    }

    # Lists of words to remove
    harmful_words = ['membunuh', 'menyakiti', 'anjing', 'bangsat', 'tai', 'sampah', 'kurang ajar', 'sialan']  # add more harmful words as needed
    
    # Replace words according to replacements dictionary
    for key, value in replacements.items():
        input_text = input_text.replace(key, value)
    
    # Remove animal and harmful words
    words = input_text.split()
    filtered_words = [word for word in words if word.lower()  not in harmful_words]
    input_text = ' '.join(filtered_words)
    
    return input_text
