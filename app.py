import streamlit as st
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Resume-JD Match Analyzer", page_icon="🎯", layout="wide")

# Common English stopwords + resume/JD filler words we don't want treated as "skills"
CUSTOM_STOPWORDS = {
    "experience", "work", "working", "years", "year", "strong", "good",
    "knowledge", "ability", "skills", "skill", "team", "role", "job",
    "candidate", "looking", "responsibilities", "requirements", "required",
    "preferred", "plus", "etc", "including", "please", "join", "company"
}


# ---------------------------
# Text extraction helpers
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_input_text(label, key_prefix):
    """Lets user either paste text or upload a PDF for a given input."""
    mode = st.radio(f"{label} input method", ["Paste text", "Upload PDF"],
                     key=f"{key_prefix}_mode", horizontal=True)
    if mode == "Paste text":
        return st.text_area(f"Paste {label} here", height=250, key=f"{key_prefix}_text")
    else:
        file = st.file_uploader(f"Upload {label} PDF", type=["pdf"], key=f"{key_prefix}_file")
        if file:
            return extract_text_from_pdf(file)
        return ""


# ---------------------------
# Core matching logic
# ---------------------------
def compute_match(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percent = round(similarity * 100, 2)

    return match_percent, vectorizer, tfidf_matrix


def extract_top_keywords(vectorizer, tfidf_matrix, doc_index, top_n=25):
    """Extract top N TF-IDF keywords for a specific document (0=resume, 1=jd)."""
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix[doc_index].toarray().flatten()
    top_indices = scores.argsort()[::-1]

    keywords = []
    for idx in top_indices:
        word = feature_names[idx]
        if scores[idx] > 0 and word not in CUSTOM_STOPWORDS and len(word) > 2 and not word.isdigit():
            keywords.append(word)
        if len(keywords) >= top_n:
            break
    return keywords


def find_keyword_gaps(resume_text, jd_keywords):
    resume_clean = clean_text(resume_text)
    resume_words = set(resume_clean.split())
    missing = [kw for kw in jd_keywords if kw not in resume_words]
    present = [kw for kw in jd_keywords if kw in resume_words]
    return present, missing


# ---------------------------
# UI
# ---------------------------
st.title("🎯 Resume-JD Match Analyzer")
st.caption("Paste your resume and a job description to see your match score and missing keywords.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Your Resume")
    resume_text = get_input_text("Resume", "resume")

with col2:
    st.subheader("📋 Job Description")
    jd_text = get_input_text("Job Description", "jd")

st.divider()

if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please provide both resume and job description text.")
    else:
        match_percent, vectorizer, tfidf_matrix = compute_match(resume_text, jd_text)
        jd_keywords = extract_top_keywords(vectorizer, tfidf_matrix, doc_index=1, top_n=25)
        present, missing = find_keyword_gaps(resume_text, jd_keywords)

        # --- Score display ---
        st.subheader("Match Score")
        score_color = "green" if match_percent >= 60 else "orange" if match_percent >= 35 else "red"
        st.markdown(f"### :{score_color}[{match_percent}%]")
        st.progress(min(int(match_percent), 100) / 100)

        if match_percent >= 60:
            st.success("Strong match — your resume aligns well with this JD.")
        elif match_percent >= 35:
            st.info("Moderate match — consider adding some missing keywords below.")
        else:
            st.warning("Low match — your resume may need significant tailoring for this role.")

        st.divider()

        # --- Keyword breakdown ---
        kcol1, kcol2 = st.columns(2)
        with kcol1:
            st.subheader("✅ Keywords You Already Have")
            if present:
                st.write(", ".join(f"`{kw}`" for kw in present))
            else:
                st.write("None of the top JD keywords were found in your resume.")

        with kcol2:
            st.subheader("⚠️ Missing Keywords")
            if missing:
                st.write(", ".join(f"`{kw}`" for kw in missing))
                st.caption("Consider adding relevant ones (only if genuinely applicable to your experience).")
            else:
                st.write("Great — no major keyword gaps found!")

st.divider()
st.caption("Built with TF-IDF + Cosine Similarity | Python + Scikit-learn + Streamlit")
