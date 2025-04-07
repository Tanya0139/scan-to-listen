import streamlit as st
from PIL import Image
import pytesseract
import spacy
import numpy as np
from difflib import get_close_matches
import time

# Sample data
AUDIO_CONTENT = {
    "The Power of Habit": "https://kukufm.com/power-of-habit-audiobook",
    "Atomic Habits": "https://kukufm.com/atomic-habits-podcast",
    "Steve Jobs": "https://kukufm.com/steve-jobs-biography-audio",
    "Ikigai": "https://kukufm.com/ikigai-japanese-secret-audio",
    "Deep Work": "https://kukufm.com/deep-work-productivity-podcast"
}

# Load spaCy model
@st.cache_resource
def load_model():
    try:
        return spacy.load('en_core_web_md')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

nlp = load_model()

if nlp is not None:
    # Precompute embeddings for all titles in the database
    title_embeddings = {title: nlp(title).vector for title in AUDIO_CONTENT.keys()}

def cosine_similarity(vec1, vec2):
    """Manual cosine similarity implementation"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def preprocess_image(image):
    """Basic image preprocessing for better OCR results"""
    gray = image.convert('L')
    return gray.point(lambda x: 0 if x < 128 else 255, '1')

def extract_text(image):
    """Extract text from image using OCR"""
    try:
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image)
        return text.strip()
    except Exception as e:
        st.error(f"Error in OCR processing: {e}")
        return ""

def fuzzy_match(text, possibilities, cutoff=0.6):
    """Fuzzy matching to handle OCR errors"""
    matches = get_close_matches(text, possibilities, n=3, cutoff=cutoff)
    return matches[0] if matches else None

def semantic_match(query, top_k=3):
    """Find semantically similar content using spaCy embeddings"""
    if nlp is None:
        return []
    
    query_embedding = nlp(query).vector
    similarities = {}
    
    for title, embedding in title_embeddings.items():
        sim = cosine_similarity(query_embedding, embedding)
        similarities[title] = sim
    
    sorted_titles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [title for title, score in sorted_titles[:top_k] if score > 0.5]

def semantic_match(query, top_k=3):
    """Enhanced semantic matching with title extraction"""
    if nlp is None or not query.strip():
        return []
    
    # Extract potential title candidates from OCR text
    doc = nlp(query)
    title_candidates = []
    
    # Look for proper nouns and noun phrases that might be titles
    for chunk in doc.noun_chunks:
        if chunk.text.upper() == chunk.text or len(chunk.text.split()) > 1:
            title_candidates.append(chunk.text)
    
    # Also consider the first line if it looks like a title
    first_line = query.split('\n')[0].strip()
    if len(first_line.split()) <= 5 and any(c.isupper() for c in first_line):
        title_candidates.append(first_line)
    
    # If we found specific title candidates, use those for matching
    if title_candidates:
        query_embedding = nlp(" ".join(title_candidates)).vector
    else:
        query_embedding = nlp(query).vector
    
    similarities = {}
    for title, embedding in title_embeddings.items():
        sim = cosine_similarity(query_embedding, embedding)
        similarities[title] = sim
    
    # Only return matches with reasonable confidence
    filtered = {k: v for k, v in similarities.items() if v > 0.6}
    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]

def find_matching_content(ocr_text):
    """Improved matching with title prioritization"""
    if not ocr_text.strip():
        return None
    
    # Clean the OCR text
    cleaned_text = " ".join([t for t in ocr_text.split() if t.isalpha()])
    
    # First try exact match with cleaned text
    for title in AUDIO_CONTENT.keys():
        if title.lower() in cleaned_text.lower():
            return title
    
    # Then try matching just the first few words (likely title)
    first_words = " ".join(cleaned_text.split()[:4])  # First 4 words
    for title in AUDIO_CONTENT.keys():
        if first_words.lower() in title.lower():
            return title
    
    # Then try fuzzy matching with higher threshold
    fuzzy_match_result = fuzzy_match(cleaned_text, list(AUDIO_CONTENT.keys()), cutoff=0.8)
    if fuzzy_match_result:
        return fuzzy_match_result
    
    # Finally try semantic matching
    semantic_matches = semantic_match(cleaned_text)
    return semantic_matches[0][0] if semantic_matches else None
def save_to_diary(title, url):
    """Simulate saving to user's Kuku Diary"""
    if 'diary' not in st.session_state:
        st.session_state.diary = {}
    
    st.session_state.diary[title] = {
        'url': url,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.success(f"Saved '{title}' to your Kuku Diary!")

def main():
    st.title("Scan-to-Listen Prototype")
    st.subheader("Bridging Physical to Digital Audio with Generative AI")
    
    st.sidebar.header("Kuku Diary")
    if 'diary' in st.session_state and st.session_state.diary:
        for title, data in st.session_state.diary.items():
            st.sidebar.write(f"- [{title}]({data['url']}) ({data['timestamp']})")
    else:
        st.sidebar.write("Your diary is empty. Scan some books to save audio content!")
    
    uploaded_file = st.file_uploader("Upload a book cover or poster image", type=["jpg", "jpeg", "png"])
    captured_image = st.camera_input("Or take a picture")

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    elif captured_image is not None:
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        
        with st.spinner("Processing image..."):
            # Step 1: OCR processing
            ocr_text = extract_text(image)
            
            if ocr_text:
                st.write("Extracted Text:", ocr_text)
                
                # Step 2: Find matching content
                matching_title = find_matching_content(ocr_text)
                
                if matching_title:
                    st.success(f"Match found: {matching_title}")
                    url = AUDIO_CONTENT[matching_title]
                    
                    # Display the match
                    st.write(f"**Audio Content:** [{matching_title}]({url})")
                    
                    # Play button (simulated)
                    if st.button("▶ Play Now"):
                        st.write(f"*Now playing: {matching_title}*")
                    
                    # Save to diary button
                    if st.button("💾 Save to Kuku Diary"):
                        save_to_diary(matching_title, url)
                else:
                    st.warning("No matching audio content found. Try a different image.")
            else:
                st.warning("No text could be extracted from the image. Please try a clearer image.")

if __name__ == "__main__":
    main()
