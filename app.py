import streamlit as st
import os
import json
import pickle
import base64
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Minimal page configuration with white background
st.set_page_config(
    page_title="GMS Tutorial Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for white background, black general text, but white input text
st.markdown("""
<style>
    .stApp {
        background-color: white;
        color: black;
    }
    
    .stButton button {
        background-color: #E53E3E;
        color: white;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #C53030;
    }
    
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: black !important;
    }
    
    .stMarkdown, .stText {
        color: black !important;
    }
    
    label, .stSelectbox, .stNumberInput {
        color: black !important;
    }
    
    /* Make input text white since background is dark */
    input[type="text"], textarea, .stTextInput input, .stNumberInput input {
        color: white !important;
        background-color: #1E293B !important;
        border: 1px solid #4B5563 !important;
    }
    
    /* Style the placeholder text */
    ::placeholder {
        color: #94A3B8 !important;
        opacity: 1 !important;
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

# Data directories
DATA_DIR = "processed_data"
PDFS_DIR = "pdfs"
LOGOS_DIR = "logos"

# Function to convert image to base64
def get_base64_image(image_path):
    """Convert an image to base64 encoding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

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
    
    try:
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
    except Exception as e:
        # Handle any errors during search
        return []

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
                response += f"{tutorial} - [View PDF]({pdf_url})\n\n"
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
            
            response += f"**{tutorial_name}** - [View PDF]({pdf_url})\n\n{content}\n\n---\n\n"
    
    return response

# Main function - ultra simplified with no custom styling
def main():
    # Add logos at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if os.path.exists(os.path.join(LOGOS_DIR, "aquaveo.png")):
            aquaveo_image = get_base64_image(os.path.join(LOGOS_DIR, "aquaveo.png"))
            if aquaveo_image:
                st.markdown(
                    f'<a href="https://www.aquaveo.com" target="_blank"><img src="data:image/png;base64,{aquaveo_image}" width="150"></a>',
                    unsafe_allow_html=True
                )
    
    with col2:
        # Title - centered
        st.title("GMS Tutorial Assistant")
    
    with col3:
        if os.path.exists(os.path.join(LOGOS_DIR, "SmartBhujalLogo.png")):
            smartbhujal_image = get_base64_image(os.path.join(LOGOS_DIR, "SmartBhujalLogo.png"))
            if smartbhujal_image:
                st.markdown(
                    f'<a href="https://www.smartbhujal.com" target="_blank"><img src="data:image/png;base64,{smartbhujal_image}" width="150"></a>',
                    unsafe_allow_html=True
                )
    
    # Simple introduction
    st.write("""
    This tool helps you search through GMS (Groundwater Modeling System) tutorials 
    and find the information you need quickly.
    """)
    
    # Basic feature list
    st.markdown("✓ Search across multiple GMS tutorials")
    st.markdown("✓ Get direct links to relevant PDF tutorials")
    st.markdown("✓ Find specific sections that answer your questions")
    st.markdown("✓ Control how many results you want to see")
    
    # Add a separator
    st.markdown("---")
    
    # ULTRA SIMPLE INPUT FIELD - no custom styling at all
    st.subheader("Ask about GMS tutorials:")
    user_input = st.text_input("Type your question here:", key="query")
    
    # Simple number selector
    num_results = st.number_input(
        "Number of results to show:",
        min_value=3,
        max_value=50,
        value=5,
        step=1
    )
    
    # Basic search button
    if st.button("Search"):
        if user_input:
            # Search function
            with st.spinner("Searching..."):
                response = get_response(user_input, num_results=num_results)
            
            # Display results
            st.subheader("Results:")
            st.markdown(response)
    
    # Footer
    st.markdown("---")
    st.markdown("Developed by [Smart Bhujal](https://www.smartbhujal.com), an official reseller of GMS software in India.")
    st.markdown("For more information, write to info@smartbhujal.com")

if __name__ == "__main__":
    # Load data silently on startup if available
    if not st.session_state.data_loaded:
        if check_data_freshness():
            load_preprocessed_data()
            st.session_state.data_loaded = True
        elif os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
            if load_preprocessed_data():
                st.session_state.data_loaded = True
    
    main()