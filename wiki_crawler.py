import requests
from bs4 import BeautifulSoup
import re
import os
import json
import time
import pickle
from urllib.parse import urljoin

# Constants
WIKI_BASE_URL = "https://www.xmswiki.com"
WIKI_STARTING_URL = "https://www.xmswiki.com/wiki/GMS:GMS_User_Manual_10.8"
WIKI_DATA_DIR = "wiki_data"  # This was missing from your original script
MAX_PAGES = 1000  # Limit to prevent excessive crawling

def setup_directories():
    """Create necessary directories"""
    if not os.path.exists(WIKI_DATA_DIR):
        os.makedirs(WIKI_DATA_DIR)
        print(f"Created directory: {WIKI_DATA_DIR}")

def get_wiki_page(url):
    """Fetch a wiki page with error handling and retries"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise exception for error status codes
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            if attempt < 2:  # Don't sleep after the last attempt
                time.sleep(2 + attempt * 2)  # Incrementally longer delays
    
    return None  # Return None if all attempts failed

def extract_wiki_content(html, url):
    """Extract main content from a wiki page"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract title
    title_elem = soup.select_one('h1.firstHeading')
    title = title_elem.text.strip() if title_elem else "Unknown Title"
    
    # Extract main content
    content_div = soup.select_one('div#mw-content-text')
    if not content_div:
        return None
    
    # Remove navigation elements, tables of contents, etc.
    for unwanted in content_div.select('.toc, .navbox, .vertical-navbox, .noprint, .mw-jump-link, .mw-editsection'):
        if unwanted:
            unwanted.decompose()
    
    # Get clean text
    content = content_div.get_text(separator=' ', strip=True)
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Extract sections (headers and content)
    sections = []
    current_section = {'title': 'Introduction', 'content': ''}
    
    for heading in content_div.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
        # Save the previous section
        if current_section['content'].strip():
            sections.append(current_section.copy())
        
        # Start a new section
        section_title = heading.get_text(strip=True)
        # Clean section title by removing [edit] links
        section_title = re.sub(r'\[edit\]', '', section_title).strip()
        current_section = {'title': section_title, 'content': ''}
        
        # Collect content for this section
        elem = heading.next_sibling
        section_content = []
        while elem and not (elem.name in ['h2', 'h3', 'h4', 'h5', 'h6']):
            if hasattr(elem, 'get_text') and elem.get_text(strip=True):
                section_content.append(elem.get_text(strip=True))
            elem = elem.next_sibling
        
        current_section['content'] = ' '.join(section_content)
    
    # Add the last section
    if current_section['content'].strip():
        sections.append(current_section)
    
    # Extract links to other wiki pages
    wiki_links = []
    for link in content_div.select('a[href^="/wiki/GMS:"]'):
        href = link.get('href')
        if href and not href.endswith('.jpg') and not href.endswith('.png'):
            full_url = urljoin(WIKI_BASE_URL, href)
            wiki_links.append(full_url)
    
    # Create the page data
    page_data = {
        'url': url,
        'title': title,
        'content': content,
        'sections': sections,
        'links': wiki_links
    }
    
    return page_data

def crawl_wiki():
    """Crawl the GMS wiki starting from the user manual page"""
    setup_directories()
    
    # Initialize the crawler
    pages_to_visit = [WIKI_STARTING_URL]
    visited_pages = set()
    wiki_data = []
    page_count = 0
    
    while pages_to_visit and page_count < MAX_PAGES:
        # Get the next URL to visit
        current_url = pages_to_visit.pop(0)
        
        # Skip if already visited
        if current_url in visited_pages:
            continue
        
        # Mark as visited
        visited_pages.add(current_url)
        
        print(f"Crawling page {page_count + 1}: {current_url}")
        
        # Fetch and parse the page
        html = get_wiki_page(current_url)
        if not html:
            continue
        
        # Extract content
        page_data = extract_wiki_content(html, current_url)
        if page_data:
            wiki_data.append(page_data)
            page_count += 1
            
            # Add new links to visit
            for link in page_data['links']:
                if link not in visited_pages and link not in pages_to_visit:
                    pages_to_visit.append(link)
        
        # Be polite to the server
        time.sleep(1)
        
        # Periodically save the data
        if page_count % 20 == 0:
            save_wiki_data(wiki_data)
    
    # Final save
    save_wiki_data(wiki_data)
    print(f"Crawling complete. Processed {page_count} pages.")
    
    # Process the data for search
    process_wiki_data()

def save_wiki_data(wiki_data):
    """Save the crawled wiki data to a JSON file"""
    with open(os.path.join(WIKI_DATA_DIR, 'wiki_data.json'), 'w', encoding='utf-8') as f:
        json.dump(wiki_data, f, indent=2)
    print(f"Saved {len(wiki_data)} pages to wiki_data.json")

def process_wiki_data():
    """Process wiki data to create searchable sections"""
    # Load the wiki data
    try:
        with open(os.path.join(WIKI_DATA_DIR, 'wiki_data.json'), 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading wiki data: {e}")
        return
    
    # Create flattened sections for searchability
    wiki_sections = []
    
    for page in wiki_data:
        # Add page title and full content as a section
        wiki_sections.append({
            'id': f"wiki-{len(wiki_sections)}",
            'url': page['url'],
            'title': page['title'],
            'content': page['content'],
            'type': 'page'
        })
        
        # Add each section separately
        for section in page['sections']:
            if len(section['content'].strip().split()) > 10:  # Only add substantial sections
                wiki_sections.append({
                    'id': f"wiki-{len(wiki_sections)}",
                    'url': page['url'] + "#" + section['title'].replace(' ', '_'),
                    'title': section['title'],
                    'content': section['content'],
                    'parent_title': page['title'],
                    'type': 'section'
                })
    
    # Save the processed sections
    with open(os.path.join(WIKI_DATA_DIR, 'wiki_sections.json'), 'w', encoding='utf-8') as f:
        json.dump(wiki_sections, f, indent=2)
    
    # Create search index
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Extract section texts
    section_texts = [section['content'] for section in wiki_sections]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(section_texts)
    
    # Save the vectorizer and matrix
    with open(os.path.join(WIKI_DATA_DIR, 'wiki_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(WIKI_DATA_DIR, 'wiki_tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    print(f"Processed {len(wiki_sections)} wiki sections for search.")

def main():
    print("Starting GMS Wiki Crawler")
    crawl_wiki()
    print("Finished crawling and processing GMS Wiki")

if __name__ == "__main__":
    main()