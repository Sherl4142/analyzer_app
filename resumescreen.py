import fitz  # PyMuPDF
import pandas as pd
import spacy

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")

# List of required skills
REQUIRED_SKILLS = [
    'python',
    'sql',
    'machine learning',
    'data visualization',
    'nlp',
    'cloud platforms',
    'resume parsing',
    'dashboarding',
    'streamlit',
    'communication'
]

# ---------------------------
# 1️⃣ Extract text from PDF
# ---------------------------
def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# ---------------------------
# 2️⃣ Clean text
# ---------------------------
def clean_text_spacy(text):
    """Clean text: lowercase, remove stopwords and non-alpha tokens."""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# ---------------------------
# 3️⃣ Score resume
# ---------------------------
def score_resume(text, required_skills=REQUIRED_SKILLS):
    """Score resume against required skills and return fit level, matched, missing."""
    found = []
    for skill in required_skills:
        if skill.lower() in text.lower():
            found.append(skill)
    missing = list(set(required_skills) - set(found))
    match_ratio = len(found) / len(required_skills)
    
    if match_ratio >= 0.75:
        fit = 'High Fit'
    elif match_ratio >= 0.4:
        fit = 'Medium Fit'
    else:
        fit = 'Low Fit'
    
    return {
        'fit_level': fit,
        'matched_skills': found,
        'missing_skills': missing
    }

# ---------------------------
# 4️⃣ Main evaluation function
# ---------------------------
def evaluate_resume(file_path):
    """
    Input: path to PDF resume
    Output: dictionary with fit_level, matched_skills, missing_skills
    """
    text = extract_text_from_pdf(file_path)
    cleaned = clean_text_spacy(text)
    result = score_resume(cleaned)
    return result
