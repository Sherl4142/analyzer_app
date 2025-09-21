import streamlit as st
import os
import fitz
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Define required skills for evaluation
required_skills = [
    'python', 'sql', 'machine learning', 'data visualization', 'nlp',
    'cloud platforms', 'resume parsing', 'dashboarding', 'streamlit', 'communication'
]

# -------------------------
# Helper functions
# -------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def clean_text_spacy(text):
    """Clean text using SpaCy: lowercase, remove stopwords/punctuation."""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def score_resume(text, required_skills):
    """Score a resume based on presence of required skills."""
    found = [skill for skill in required_skills if skill.lower() in text.lower()]
    missing = list(set(required_skills) - set(found))
    match_ratio = len(found) / len(required_skills)
    score = int(match_ratio * 100)  # Relevance score 0–100
   
    if match_ratio >= 0.75:
        fit = 'High Fit'
    elif match_ratio >= 0.4:
        fit = 'Medium Fit'
    else:
        fit = 'Low Fit'
   
    return pd.Series([score, fit, found, missing])

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Resume Evaluator", page_icon="📄", layout="wide")
st.title("AI Resume Evaluator")
st.markdown("Automate resume evaluation against job requirements at scale. Upload resumes and get instant insights!")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    data = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        cleaned_text = clean_text_spacy(text)
        score, fit, matched, missing = score_resume(cleaned_text, required_skills)
        data.append({
            'file_name': file.name,
            'relevance_score': score,
            'fit_level': fit,
            'matched_skills': ", ".join(matched),
            'missing_skills': ", ".join(missing)
        })
   
    df_resumes = pd.DataFrame(data)
   
    st.subheader("Resume Evaluation Results")
    st.dataframe(df_resumes)

    # Fit distribution pie chart
    st.subheader("Fit Level Distribution")
    fig, ax = plt.subplots()
    df_resumes["fit_level"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    plt.ylabel("")
    st.pyplot(fig)

    # Bar chart for relevance score
    st.subheader("Relevance Score per Resume")
    fig2, ax2 = plt.subplots()
    sns.barplot(x='relevance_score', y='file_name', data=df_resumes, ax=ax2, palette="viridis")
    ax2.set_xlabel("Relevance Score (0-100)")
    ax2.set_ylabel("Resume")
    st.pyplot(fig2)

    # Highlight gaps for improvement
    st.subheader("Suggested Improvements")
    for idx, row in df_resumes.iterrows():
        if row['missing_skills']:
            st.markdown(f"**{row['file_name']}**: Missing skills -> {row['missing_skills']}")
        else:
            st.markdown(f"**{row['file_name']}**: All required skills present ✅")

