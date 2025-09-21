def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# %%
import os
import glob
import fitz
import pandas as pd

pdf_folder = r"D:\data1"  # raw string to handle backslashes
pdf_files = glob.glob(os.path.join(pdf_folder, '*.pdf'))

resumes = {}
for file in pdf_files:
    resumes[file] = extract_text_from_pdf(file)


# %%
for path, text in resumes.items():
    print(f"\n--- {path} ---\n")
    print(text[:500])  # show first 500 characters
    break

# %%
data = []

for file in pdf_files:
    text = extract_text_from_pdf(file)
    data.append({
        'file_path': file,
        'file_name': os.path.basename(file),
        'resume_text': text
    })

df_resumes = pd.DataFrame(data)


# %%
data

# %%
df_resumes.head()


# %%
df_resumes.drop(columns="file_path",inplace=True)

# %%
df_resumes

# %%
df_resumes.info()

# %%
import nltk
nltk.download('stopwords')

# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# %%
stop_words = set(stopwords.words('english'))

# %%
stop_words

# %%

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# %%
import spacy
nlp = spacy.load("en_core_web_sm")


# %%
def clean_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)


# %%
df_resumes['cleaned_text'] = df_resumes['resume_text'].apply(clean_text_spacy)


# %%
df_resumes

# %%
required_skills = [
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


# %%
import numpy as np

# %%
def score_resume(text, required_skills):
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
    
    return pd.Series([fit, found, missing])


# %%
df_resumes[['fit_level', 'matched_skills', 'missing_skills']] = df_resumes['cleaned_text'].apply(lambda x: score_resume(x, required_skills))


# %%
df_resumes[['file_name', 'fit_level', 'matched_skills', 'missing_skills']].head(25)


# %%
fit_order = {'High Fit': 0, 'Medium Fit': 1, 'Low Fit': 2}


# %%
df_resumes['fit_rank'] = df_resumes['fit_level'].map(fit_order)


# %%
df_sorted = df_resumes.sort_values(by='fit_rank')


# %%
df_sorted.drop(columns="cleaned_text",inplace=True)

# %%
df_sorted

# %%
import matplotlib.pyplot as plt
df_sorted["fit_level"].value_counts(normalize=True).plot.pie(autopct = '%1.2f%%')
plt.show()

# %%
import seaborn as sns


# %%



