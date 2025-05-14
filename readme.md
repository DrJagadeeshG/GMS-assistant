# GMS Tutorial Assistant

A powerful search tool that helps you find information across GMS (Groundwater Modeling System) tutorials and wiki documentation.

## Overview

The GMS Tutorial Assistant allows you to search through multiple PDF tutorials and the official GMS Wiki simultaneously. It uses natural language processing to find relevant content and provides direct links to the source materials.

![GMS Tutorial Assistant Screenshot](screenshot.png)

## Features

- ✓ Search across multiple GMS tutorials simultaneously
- ✓ Search the GMS Wiki documentation
- ✓ Get direct links to relevant PDF tutorials and wiki pages
- ✓ Find specific sections that answer your questions
- ✓ Filter results by source (PDFs or Wiki)
- ✓ Control how many results to show from each source

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/gms-tutorial-helper.git
cd gms-tutorial-helper
```

2. **Create a virtual environment**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download PDF tutorials**

Place GMS tutorial PDFs in the `pdfs` directory. Alternatively, run the downloader script:

```bash
python download_pdfs.py
```

5. **Crawl the GMS Wiki** (optional but recommended)

```bash
python fixed_wiki_crawler.py
```

## Usage

1. **Start the application**

```bash
streamlit run app.py
```

2. **Access the web interface**

Open your browser and go to http://localhost:8501

3. **Using the search**

- Enter your question in the search box
- Select how many results you want to see per category
- Choose which sources to search (PDFs, Wiki, or both)
- Click "Search" to see the results
- Click on the links to view the original PDFs or Wiki pages

## Directory Structure

```
gms-tutorial-helper/
├── app.py                  # Main Streamlit application
├── download_pdfs.py        # Script to download GMS tutorial PDFs
├── fixed_wiki_crawler.py   # Script to crawl and process the GMS Wiki
├── pdfs/                   # Directory for PDF tutorials
├── logos/                  # Logo images (Aquaveo and Smart Bhujal)
├── processed_data/         # Processed PDF data and search indices
├── wiki_data/              # Processed Wiki data and search indices
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Technical Details

The GMS Tutorial Assistant uses:

- **Streamlit** for the web interface
- **TF-IDF** (Term Frequency-Inverse Document Frequency) for search functionality
- **BeautifulSoup** for Wiki crawling
- **scikit-learn** for text processing and similarity calculations
- **PyPDF2** for PDF parsing

## Requirements

See `requirements.txt` for a complete list of dependencies:

```
streamlit>=1.25.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
beautifulsoup4>=4.12.0
requests>=2.30.0
PyPDF2>=3.0.0
```

## About

Developed by [Smart Bhujal](https://www.smartbhujal.com), an official reseller of GMS software in India.

For more information, contact info@smartbhujal.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.