# main.py
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Required skills for evaluation
required_skills = [
    'python', 'sql', 'machine learning', 'data visualization', 'nlp',
    'cloud platforms', 'resume parsing', 'dashboarding', 'streamlit', 'communication'
]

# -------------------------
# Helper functions
# -------------------------
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file uploaded via Streamlit."""
    text = ""
    try:
        # PyMuPDF can read file-like objects directly
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        pdf_file.seek(0)  # Reset file pointer after reading
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
    return text

def clean_text_spacy(text: str) -> str:
    """Clean text using SpaCy: lowercase, remove stopwords/punctuation."""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def score_resume(text: str, required_skills: list) -> pd.Series:
    """Score a resume based on presence of required skills."""
    found = [skill for skill in required_skills if skill.lower() in text.lower()]
    missing = list(set(required_skills) - set(found))
    match_ratio = len(found) / len(required_skills)
    score = int(match_ratio * 100)

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
st.markdown(
    "Automate resume evaluation against job requirements at scale. "
    "Upload resumes and get instant insights!"
)

# Upload multiple resumes
uploaded_files = st.file_uploader(
    "Upload PDF resumes", type=["pdf"], accept_multiple_files=True
)

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

    # -------------------------
    # Sidebar options
    # -------------------------
    st.sidebar.header("Analysis Options")
    show_table = st.sidebar.checkbox("Show Evaluation Table", True)
    show_fit_chart = st.sidebar.checkbox("Fit Level Distribution", True)
    show_score_chart = st.sidebar.checkbox("Relevance Score Chart", True)
    show_missing_skills = st.sidebar.checkbox("Missing Skills", True)
    show_matching_skills = st.sidebar.checkbox("Matching Skills", True)

    # Display based on sidebar selection
    if show_table:
        st.subheader("Resume Evaluation Results")
        st.dataframe(df_resumes)

    if show_fit_chart:
        st.subheader("Fit Level Distribution")
        fig, ax = plt.subplots()
        df_resumes["fit_level"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    if show_score_chart:
        st.subheader("Relevance Score per Resume")
        fig2, ax2 = plt.subplots()
        sns.barplot(
            x='relevance_score', y='file_name', data=df_resumes,
            ax=ax2, palette="viridis"
        )
        ax2.set_xlabel("Relevance Score (0-100)")
        ax2.set_ylabel("Resume")
        st.pyplot(fig2)

    if show_missing_skills:
        st.subheader("Missing Skills per Resume")
        for idx, row in df_resumes.iterrows():
            if row['missing_skills']:
                st.markdown(f"**{row['file_name']}**: Missing skills -> {row['missing_skills']}")
            else:
                st.markdown(f"**{row['file_name']}**: No missing skills ✅")

    if show_matching_skills:
        st.subheader("Matching Skills per Resume")
        for idx, row in df_resumes.iterrows():
            st.markdown(f"**{row['file_name']}**: Matching skills -> {row['matched_skills']}")

           

