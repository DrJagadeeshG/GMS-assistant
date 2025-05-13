# GMS Tutorial Assistant - Final Version without Process Button
# Improved version with automatic loading and no process button display

import streamlit as st
import os
import json
import pickle
import base64
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Page configuration
st.set_page_config(
    page_title="GMS Tutorial Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Function to convert image to base64
def get_base64_image(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Custom CSS for minimal clean design with white background and fixed title color
st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* General page styling */
.stApp {
    background-color: white;
}

/* Clean container styling */
div.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Success message hiding */
div.stSuccessMessage {
    display: none !important;
}

/* Title styling - with explicit color setting */
.app-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1A202C !important; /* Dark color for the title */
    text-align: center;
    margin: 1rem 0;
}

/* Form styling */
.stButton > button {
    background-color: #E53E3E;
    color: white;
    border: none;
    padding: 0.5rem 2rem;
    font-weight: 500;
    border-radius: 5px;
}

.stButton > button:hover {
    background-color: #C53030;
}

/* Input field styling */
input[type="text"] {
    background-color: #1A202C !important;
    color: white !important;
    border-radius: 5px !important;
    border: none !important;
    padding: 10px !important;
}

input[type="text"]::placeholder {
    color: #A0AEC0 !important;
}

/* Selectbox styling */
.stSelectbox > div > div > div {
    background-color: white !important;
    color: #1A202C !important;
}

.stSelectbox label {
    color: #1A202C !important;
}

/* PDF link styling */
.pdf-link {
    background-color: #f7fafc;
    border-left: 4px solid #4299E1;
    padding: 1rem;
    margin-top: 0.75rem;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.pdf-link-title {
    font-weight: 600;
    color: #2c5282;
    font-size: 1.1rem;
}

.pdf-link a, .pdf-recommendation a {
    color: white;
    text-decoration: none;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    background-color: #4299E1;
    border: none;
    font-size: 0.85rem;
    margin-left: 0.8rem;
    white-space: nowrap;
    display: inline-block;
}

.pdf-link a:hover, .pdf-recommendation a:hover {
    background-color: #3182CE;
}

.pdf-context {
    margin-top: 0.8rem;
    font-style: italic;
    color: #4a5568;
    border-left: 2px solid #cbd5e0;
    padding-left: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.5;
}

.pdf-recommendation {
    background-color: #f0fff4;
    border-left: 4px solid #48BB78;
    padding: 1rem;
    margin-top: 0.75rem;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Message styling */
.user-message {
    background-color: #ebf8ff;
    color: #2c5282;
    padding: 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    border-left: 4px solid #4299E1;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.assistant-message {
    background-color: #f7fafc;
    color: #1a202c;
    padding: 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    border-left: 4px solid #48BB78;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Footer styling */
.footer {
    text-align: center;
    padding: 1rem;
    color: #4a5568;
    font-size: 0.9rem;
    margin-top: 2rem;
    border-top: 1px solid #e2e8f0;
}

/* Section headers - with explicit color setting */
h1, h2, h3, h4, h5, h6 {
    color: #1A202C !important;
}

/* Question prompt styling */
.question-prompt {
    color: #1A202C !important;
    font-size: 1.25rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* Results count label */
.results-count-label {
    color: #1A202C !important;
    font-weight: 500;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Link styling */
a {
    color: inherit;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

.footer a {
    color: #4a5568;
    text-decoration: none;
}

.footer a:hover {
    color: #2c5282;
    text-decoration: underline;
}

/* Info message styling */
.st-bd {
    display: none !important;
}

.stAlert {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.tutorial_data = {}
    st.session_state.section_data = []
    st.session_state.tfidf_vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.loading_timestamp = None

# Add a state for tracking submitted queries
if 'submitted_query' not in st.session_state:
    st.session_state.submitted_query = False

# Store the number of results preference
if 'num_results' not in st.session_state:
    st.session_state.num_results = 3

# Suppress "data loaded" message after first load
if 'suppress_message' not in st.session_state:
    st.session_state.suppress_message = False

# Data directories
DATA_DIR = "processed_data"
PDFS_DIR = "pdfs"
LOGOS_DIR = "logos"

# Function to preprocess PDFs and save the data
def preprocess_pdfs():
    """Convert PDFs to JSON data and compute TF-IDF matrix"""
    import PyPDF2
    import re
    
    # Create directories if they don't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(PDFS_DIR):
        os.makedirs(PDFS_DIR)
        return False
    
    pdf_files = [f for f in os.listdir(PDFS_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        return False
    
    # Process each PDF silently
    tutorial_data = {}
    all_sections = []
    
    for idx, pdf_file in enumerate(pdf_files):
        file_path = os.path.join(PDFS_DIR, pdf_file)
        tutorial_name = pdf_file.replace('.pdf', '')
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
                
                # Save the full text
                tutorial_data[tutorial_name] = {
                    "text": text,
                    "filename": pdf_file,
                    "pages": len(reader.pages)
                }
                
                # Split into sections (paragraphs)
                sections = []
                paragraphs = re.split(r'\n\s*\n', text)
                
                for i, para in enumerate(paragraphs):
                    # Skip very short paragraphs
                    if len(para.strip().split()) > 5:
                        section = {
                            "id": f"{tutorial_name}-{i}",
                            "tutorial": tutorial_name,
                            "content": para.strip(),
                            "index": i
                        }
                        sections.append(section)
                        all_sections.append(section)
                
        except Exception as e:
            pass
    
    # Create and save TF-IDF vectorizer
    section_texts = [section["content"] for section in all_sections]
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(section_texts)
    
    # Save the processed data
    with open(os.path.join(DATA_DIR, 'tutorial_data.json'), 'w') as f:
        json.dump(tutorial_data, f)
    
    with open(os.path.join(DATA_DIR, 'section_data.json'), 'w') as f:
        json.dump(all_sections, f)
    
    # Save the vectorizer and matrix
    with open(os.path.join(DATA_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(DATA_DIR, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    # Save the timestamp
    with open(os.path.join(DATA_DIR, 'processed_timestamp.txt'), 'w') as f:
        f.write(str(time.time()))
    
    return True

# Function to check if data needs to be updated
def check_data_freshness():
    """Check if processed data is up-to-date with PDF files"""
    # Check if processed data exists
    if not os.path.exists(os.path.join(DATA_DIR, 'processed_timestamp.txt')):
        return False
    
    # Get the timestamp of the last processing
    with open(os.path.join(DATA_DIR, 'processed_timestamp.txt'), 'r') as f:
        try:
            last_processed = float(f.read().strip())
        except:
            return False
    
    # Check if any PDF is newer than the processed data
    for pdf_file in os.listdir(PDFS_DIR):
        if pdf_file.endswith('.pdf'):
            file_path = os.path.join(PDFS_DIR, pdf_file)
            if os.path.getmtime(file_path) > last_processed:
                return False
    
    return True

# Function to load preprocessed data
def load_preprocessed_data():
    """Load the preprocessed data from disk"""
    try:
        # Load tutorial data
        with open(os.path.join(DATA_DIR, 'tutorial_data.json'), 'r') as f:
            st.session_state.tutorial_data = json.load(f)
        
        # Load section data
        with open(os.path.join(DATA_DIR, 'section_data.json'), 'r') as f:
            st.session_state.section_data = json.load(f)
        
        # Load vectorizer
        with open(os.path.join(DATA_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            st.session_state.tfidf_vectorizer = pickle.load(f)
        
        # Load TF-IDF matrix
        with open(os.path.join(DATA_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
            st.session_state.tfidf_matrix = pickle.load(f)
        
        # Load timestamp
        with open(os.path.join(DATA_DIR, 'processed_timestamp.txt'), 'r') as f:
            st.session_state.loading_timestamp = float(f.read().strip())
        
        return True
    except Exception as e:
        return False

# Function to search for relevant content
def search_content(query, top_n=5):
    """Search for relevant content using the TF-IDF matrix"""
    # Proper check for vectorizer and matrix existence
    if (st.session_state.tfidf_vectorizer is None) or (st.session_state.tfidf_matrix is None):
        return []
    
    # Transform the query using the vectorizer
    query_vector = st.session_state.tfidf_vectorizer.transform([query])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(query_vector, st.session_state.tfidf_matrix).flatten()
    
    # Get the top N most relevant sections
    top_indices = similarity_scores.argsort()[:-top_n-1:-1]
    
    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0.0:  # Only include relevant results
            results.append({
                "section": st.session_state.section_data[idx],
                "score": float(similarity_scores[idx])
            })
    
    return results

# Function to extract keywords from query
def extract_keywords(query):
    """Extract important keywords from the query"""
    # Remove common words and stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                'from', 'of', 'how', 'what', 'when', 'where', 'why', 'who', 'which'}
    
    words = query.lower().split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords

# Function to suggest relevant tutorials based on keywords
def suggest_tutorials(keywords, num_results=3):
    """Suggest tutorials that might be relevant to the keywords"""
    tutorial_scores = {}
    
    for keyword in keywords:
        for tutorial_name, data in st.session_state.tutorial_data.items():
            # Count keyword occurrences in the tutorial
            count = data["text"].lower().count(keyword.lower())
            
            if count > 0:
                if tutorial_name in tutorial_scores:
                    tutorial_scores[tutorial_name] += count
                else:
                    tutorial_scores[tutorial_name] = count
    
    # Sort tutorials by relevance score
    sorted_tutorials = sorted(tutorial_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [name for name, score in sorted_tutorials[:num_results]]

# Function to generate a response
def get_response(query, num_results=3):
    """Generate a response based on the user query with PDF links"""
    # First, search for relevant content
    search_results = search_content(query, top_n=max(5, num_results))
    
    # Base URL for online PDFs
    s3_base_url = "https://s3.amazonaws.com/gmstutorials-10.8.aquaveo.com/"
    
    if not search_results:
        # No direct matches, suggest tutorials based on keywords
        keywords = extract_keywords(query)
        suggested_tutorials = suggest_tutorials(keywords, num_results)
        
        if suggested_tutorials:
            response = "I couldn't find specific information about that, but these tutorials might be helpful:\n\n"
            for tutorial in suggested_tutorials:
                # Create a PDF link
                pdf_url = f"{s3_base_url}{tutorial}.pdf"
                response += f"""<div class='pdf-recommendation'>
                <span class='pdf-link-title'>{tutorial}</span>
                <a href="{pdf_url}" target="_blank">View PDF Online</a>
                </div>\n\n"""
        else:
            response = "I'm sorry, I couldn't find any relevant information in the available tutorials. Could you try rephrasing your question?"
    else:
        # Format the search results
        response = f"Here's what I found in the GMS tutorials (showing {min(len(search_results), num_results)} results):\n\n"
        
        for i, result in enumerate(search_results[:num_results]):  # Show top N results
            section = result["section"]
            content = section["content"]
            tutorial_name = section["tutorial"]
            
            # Create a PDF link
            pdf_url = f"{s3_base_url}{tutorial_name}.pdf"
            
            # Truncate content if too long
            if len(content) > 300:
                content = content[:300] + "..."
            
            response += f"""<div class='pdf-link'>
            <span class='pdf-link-title'>{tutorial_name}</span>
            <a href="{pdf_url}" target="_blank">View PDF Online</a>
            <div class='pdf-context'>{content}</div>
            </div>\n\n"""
    
    return response

# Callback function for form submission
def handle_input():
    if st.session_state.input_field:  # Only set submitted flag if there's text
        st.session_state.submitted_query = True

# Main function
def main():
    # Load data silently on startup
    if not st.session_state.data_loaded:
        # First check if data exists and is fresh
        if check_data_freshness():
            load_preprocessed_data()
            st.session_state.data_loaded = True
        # If not, try to load it anyway (might be partial)
        elif os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
            if load_preprocessed_data():
                st.session_state.data_loaded = True
    
    # Simple title with built-in images
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if os.path.exists(os.path.join(LOGOS_DIR, "aquaveo.png")):
            with st.container():
                st.markdown(
                    f"""<a href="https://www.aquaveo.com" target="_blank">
                    <img src="data:image/png;base64,{get_base64_image(os.path.join(LOGOS_DIR, 'aquaveo.png'))}" width="150">
                    </a>""",
                    unsafe_allow_html=True
                )
    
    with col2:
        # Use explicit dark color for the title
        st.markdown("<h1 class='app-title'>GMS Tutorial Assistant</h1>", unsafe_allow_html=True)
    
    with col3:
        if os.path.exists(os.path.join(LOGOS_DIR, "SmartBhujalLogo.png")):
            with st.container():
                st.markdown(
                    f"""<a href="https://www.smartbhujal.com" target="_blank">
                    <img src="data:image/png;base64,{get_base64_image(os.path.join(LOGOS_DIR, 'SmartBhujalLogo.png'))}" width="150">
                    </a>""",
                    unsafe_allow_html=True
                )
    
    # Main interface - always shown now, without process button section
    # Use explicit class for the prompt text color
    st.markdown("<div class='question-prompt'>Ask about GMS tutorials:</div>", unsafe_allow_html=True)
    
    # Create the input form
    with st.form(key="query_form", clear_on_submit=True):
        user_input = st.text_input("", key="input_field", placeholder="Type your question here...")
        
        # Add number of results selection - now directly below the input
        st.markdown("<div class='results-count-label'>Number of results to show:</div>", unsafe_allow_html=True)
        num_results = st.selectbox("", options=[3, 5, 10], key="results_count")
        # Update session state with the selected number
        st.session_state.num_results = num_results
        
        # Center the Ask button
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            submit_button = st.form_submit_button("Ask", on_click=handle_input)
    
    # Process the submission
    if st.session_state.submitted_query:
        # Get the user input
        user_input = st.session_state.input_field
        
        # Clear previous messages - only keep current query and response
        st.session_state.messages = []
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response using the specified number of results
        with st.spinner("Searching tutorials..."):
            response = get_response(user_input, num_results=st.session_state.num_results)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Reset the submitted flag
        st.session_state.submitted_query = False
        
        # Use st.rerun() to update the UI
        st.rerun()
    
    # Results section - only show if there are messages
    if st.session_state.messages:
        # Use explicit color for the header
        st.markdown("<h3 style='color: #1A202C !important;'>Results</h3>", unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Footer with clickable link
    st.markdown("<div class='footer'>Developed by <a href='https://www.smartbhujal.com' target='_blank'><b>Smart Bhujal</b></a>, an official reseller of GMS software in India.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()