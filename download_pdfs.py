# Improved PDF Downloader Script for GMS Tutorials
# Handles 403 errors and includes more robust retry logic

import os
import requests
from bs4 import BeautifulSoup
import re
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import urllib.parse

# Thread-safe counter for progress tracking
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

# User agent rotation to avoid being blocked
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
]

def get_headers():
    """Get random headers to avoid being blocked"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'https://aquaveo.com/software/gms/learning-tutorials',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',  # Do Not Track
    }

def try_alternative_urls(name, counter, total):
    """Try various URL patterns that might work"""
    base_urls = [
        "https://s3.amazonaws.com/gmstutorials-10.8.aquaveo.com/",
        "https://s3.amazonaws.com/gmstutorials-10.7.aquaveo.com/",
        "https://s3.amazonaws.com/gmstutorials-10.6.aquaveo.com/",
        "https://s3.amazonaws.com/gmstutorials-10.5.aquaveo.com/",
        "https://s3.amazonaws.com/gmstutorials-10.4.aquaveo.com/",
    ]
    
    name_variants = [
        name,
        name.lower(),
        name.upper(),
        # Try with spaces instead of removing them
        name.replace("", " ").strip(),
        # Try with hyphens instead of removing spaces
        name.replace("", "-").strip(),
        # Attempt URL encoding
        urllib.parse.quote(name)
    ]
    
    for base_url in base_urls:
        for variant in name_variants:
            pdf_url = f"{base_url}{variant}.pdf"
            pdf_path = os.path.join("pdfs", f"{name}.pdf")
            
            try:
                time.sleep(0.5)  # Be polite to the server
                headers = get_headers()
                response = requests.get(pdf_url, timeout=30, headers=headers)
                
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                    count = counter.increment()
                    print(f"[{count}/{total}] Successfully downloaded {name} (alternative URL: {pdf_url})")
                    return True
            except Exception as e:
                pass
    
    count = counter.increment()
    print(f"[{count}/{total}] Failed to download {name} after trying all alternative URLs")
    return False

def download_pdf(name, s3_base_url, counter, total, retry_count=3):
    """Download a single PDF file with retries"""
    pdf_url = f"{s3_base_url}{name}.pdf"
    pdf_path = os.path.join("pdfs", f"{name}.pdf")
    
    # Skip if already downloaded
    if os.path.exists(pdf_path):
        count = counter.increment()
        print(f"[{count}/{total}] Skipping {name} - already downloaded")
        return True
    
    for attempt in range(retry_count):
        try:
            # Add a small delay between attempts
            if attempt > 0:
                time.sleep(2 + random.random() * 3)  # Random delay between 2-5 seconds
                
            headers = get_headers()
            response = requests.get(pdf_url, timeout=30, headers=headers)
            
            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                count = counter.increment()
                print(f"[{count}/{total}] Successfully downloaded {name}")
                return True
            elif response.status_code == 403:
                if attempt == retry_count - 1:
                    # This was our last retry, try alternative URLs
                    return try_alternative_urls(name, counter, total)
            else:
                # Try alternative URL format
                alt_pdf_url = f"{s3_base_url.rstrip('/')}/{name}.pdf"
                headers = get_headers()
                response = requests.get(alt_pdf_url, timeout=30, headers=headers)
                
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                    count = counter.increment()
                    print(f"[{count}/{total}] Successfully downloaded {name} (alt URL)")
                    return True
        except Exception as e:
            if attempt == retry_count - 1:
                count = counter.increment()
                print(f"[{count}/{total}] Error downloading {name} after {retry_count} attempts: {e}")
                # Try alternative URLs as a last resort
                return try_alternative_urls(name, counter, total)
            else:
                print(f"Attempt {attempt+1}/{retry_count} failed for {name}: {e}. Retrying...")
    
    count = counter.increment()
    print(f"[{count}/{total}] Failed to download {name}")
    return False

def clean_tutorial_name(name):
    """Clean up tutorial name to make it more likely to match the PDF name"""
    # Remove special characters but keep spaces for now
    name = re.sub(r'[^a-zA-Z0-9\s\-]', '', name)
    # Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name).strip()
    # Replace spaces with nothing
    name = name.replace(' ', '')
    return name

def find_pdfs_on_webpage(url):
    """Find PDF links and tutorial names from the webpage"""
    print(f"Fetching tutorial page: {url}")
    tutorial_names = []
    direct_links = []
    
    try:
        response = requests.get(url, headers=get_headers())
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching tutorial page: {e}")
        return tutorial_names, direct_links
    
    # Look for direct PDF links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.pdf') and 'gmstutorials' in href:
            direct_links.append(href)
            filename = href.split('/')[-1]
            tutorial_name = filename.replace('.pdf', '')
            tutorial_names.append(tutorial_name)
    
    # Extract text-based tutorial names from table cells
    tutorial_elements = soup.find_all(['td', 'th'])
    for element in tutorial_elements:
        text = element.get_text().strip()
        if text and not text.startswith('=') and len(text) > 1:
            clean_name = clean_tutorial_name(text)
            # Skip very short names or navigation elements
            if len(clean_name) > 2 and clean_name not in ['', '|']:
                tutorial_names.append(clean_name)
    
    # Extract potential headings that might be tutorial names
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        text = heading.get_text().strip()
        if text and len(text) > 3:
            clean_name = clean_tutorial_name(text)
            if len(clean_name) > 2:
                tutorial_names.append(clean_name)
    
    # Remove duplicates
    tutorial_names = list(set(tutorial_names))
    
    return tutorial_names, direct_links

def main():
    # Create pdfs directory if it doesn't exist
    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")
        print("Created 'pdfs' directory")

    # URL of the GMS tutorials page
    url = "https://aquaveo.com/software/gms/learning-tutorials"

    # Find PDF links and tutorial names from the webpage
    tutorial_names, direct_links = find_pdfs_on_webpage(url)
    
    print(f"Found {len(tutorial_names)} potential tutorials")
    print(f"Found {len(direct_links)} direct PDF links")

    # S3 base URL for PDFs
    s3_base_url = "https://s3.amazonaws.com/gmstutorials-10.8.aquaveo.com/"

    # Initialize counter
    counter = Counter()
    
    # For direct links, try to download them first
    direct_download_count = 0
    for link in direct_links:
        filename = link.split('/')[-1]
        tutorial_name = filename.replace('.pdf', '')
        pdf_path = os.path.join("pdfs", filename)
        
        if os.path.exists(pdf_path):
            print(f"Skipping {filename} - already downloaded")
            direct_download_count += 1
            continue
            
        try:
            headers = get_headers()
            response = requests.get(link, timeout=30, headers=headers)
            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {filename} from direct link")
                direct_download_count += 1
            else:
                print(f"Failed to download from direct link: {link} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading direct link: {e}")
        
        # Be polite to the server
        time.sleep(0.5)
    
    print(f"Successfully downloaded {direct_download_count} PDFs from direct links")
    
    # Download PDFs in parallel with a smaller thread pool to be gentler
    print("Starting downloads for remaining tutorials...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(
            lambda name: download_pdf(name, s3_base_url, counter, len(tutorial_names)), 
            tutorial_names
        ))
    
    # Count successful downloads
    successful_downloads = results.count(True) + direct_download_count
    print(f"Downloaded {successful_downloads} out of {len(tutorial_names)} tutorials")
    
    # List of failed downloads
    failed_downloads = [tutorial_names[i] for i, result in enumerate(results) if not result]
    if failed_downloads:
        print(f"Failed to download {len(failed_downloads)} tutorials:")
        for name in failed_downloads[:10]:  # Show first 10 failures
            print(f"  - {name}")
        if len(failed_downloads) > 10:
            print(f"  ... and {len(failed_downloads) - 10} more")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")