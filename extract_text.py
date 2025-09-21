# %% Imports
import os
import glob
import fitz  # PyMuPDF
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# %% NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# %% Load SpaCy
nlp = spacy.load("en_core_web_sm")

# %% Function to extract text from a PDF (works with file path or file-like object)
def extract_text_from_pdf(pdf_input):
    """
    pdf_input: str (file path) OR file-like object (uploaded file)
    """
    if isinstance(pdf_input, str):  # file path
        if not os.path.exists(pdf_input):
            print(f"Warning: File {pdf_input} not found.")
            return ""
        doc = fitz.open(pdf_input)
    else:  # file-like object
        pdf_bytes = pdf_input.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_input.seek(0)
    
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# %% Folder containing PDF resumes
pdf_folder = r"D:\data1"  # Update to your folder
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

# %% Extract text from all PDFs
resumes = {}
for file in pdf_files:
    resumes[file] = extract_text_from_pdf(file)

# %% Preview first resume
for path, text in resumes.items():
    print(f"\n--- {path} ---\n")
    print(text[:500])  # first 500 characters
    break

# %% Create DataFrame
data = []
for file in pdf_files:
    text = extract_text_from_pdf(file)
    data.append({
        "file_path": file,
        "file_name": os.path.basename(file),
        "resume_text": text
    })

df_resumes = pd.DataFrame(data)
df_resumes.drop(columns="file_path", inplace=True)

# %% Text cleaning functions
def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

def clean_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

df_resumes['cleaned_text'] = df_resumes['resume_text'].apply(clean_text_spacy)

# %% Skills and scoring
required_skills = [
    'python', 'sql', 'machine learning', 'data visualization', 
    'nlp', 'cloud platforms', 'resume parsing', 'dashboarding', 
    'streamlit', 'communication'
]

def score_resume(text, required_skills):
    found = [skill for skill in required_skills if skill.lower() in text.lower()]
    missing = list(set(required_skills) - set(found))
    match_ratio = len(found) / len(required_skills)
    
    if match_ratio >= 0.75:
        fit = "High Fit"
    elif match_ratio >= 0.4:
        fit = "Medium Fit"
    else:
        fit = "Low Fit"
    
    return pd.Series([fit, found, missing])

df_resumes[['fit_level', 'matched_skills', 'missing_skills']] = df_resumes['cleaned_text'].apply(lambda x: score_resume(x, required_skills))

# %% Ranking and sorting
fit_order = {'High Fit': 0, 'Medium Fit': 1, 'Low Fit': 2}
df_resumes['fit_rank'] = df_resumes['fit_level'].map(fit_order)
df_sorted = df_resumes.sort_values(by='fit_rank')
df_sorted.drop(columns='cleaned_text', inplace=True)

# %% Visualization
df_sorted["fit_level"].value_counts(normalize=True).plot.pie(autopct='%1.2f%%')
plt.show()

# %% Final DataFrame
print(df_sorted[['file_name', 'fit_level', 'matched_skills', 'missing_skills']])
