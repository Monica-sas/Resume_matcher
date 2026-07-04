# Resume-JD Match Analyzer

A tool that scores how well your resume matches a job description using TF-IDF and
cosine similarity, and highlights keywords you're missing.

---

## Step-by-Step Build Guide (3-4 hours)

### Hour 1 — Setup & Understand the Pipeline
1. Install dependencies:
   ```bash
   pip install streamlit scikit-learn PyPDF2
   ```
2. Understand the flow before touching code:
   - Resume text + JD text → clean both → convert to TF-IDF vectors → cosine similarity → % match score
   - Separately: pull top TF-IDF keywords from the JD → check which ones appear in the resume → "present" vs "missing"
3. Test with two paragraphs of dummy text in a Python shell before building the UI — confirms the ML logic works before you touch Streamlit.

### Hour 2 — Core Matching Logic (`app.py`)
1. Write `clean_text()` — lowercase, strip punctuation, collapse whitespace.
2. Write `compute_match()` — fit a `TfidfVectorizer` on `[resume, jd]`, compute cosine similarity between the two vectors.
3. Write `extract_top_keywords()` — pull the highest-scoring TF-IDF terms from the JD vector specifically (not the resume).
4. Write `find_keyword_gaps()` — simple set comparison between JD keywords and resume words.
5. Test again with real resume/JD text pasted in.

### Hour 3 — Streamlit UI
1. Two-column layout: resume input | JD input.
2. Each side supports **paste text OR upload PDF** (PyPDF2 extracts PDF text).
3. "Analyze Match" button triggers scoring.
4. Display: percentage score with color coding (green/orange/red), present keywords, missing keywords.

### Hour 4 — Polish & Test
1. Test edge cases: empty inputs, PDF with weird formatting, very short JD.
2. Test with YOUR actual resume against 2-3 real job descriptions (from LinkedIn/Naukri postings for E-Con Systems, Zoho, TCS roles) — this also doubles as genuinely useful prep for your placements.
3. Deploy to Streamlit Community Cloud (free) so you have a live link for your resume.

---

## How to Run Locally

```bash
cd resume-jd-matcher
pip install -r requirements.txt
streamlit run app.py
```

It will open in your browser at `http://localhost:8501`.

---

## How to Deploy (Streamlit Community Cloud — free)

1. Push this folder to a new GitHub repo (e.g., `Monica-sas/resume-jd-matcher`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app", select your repo, branch `main`, main file `app.py`.
4. Deploy — you'll get a live URL like `resume-jd-matcher.streamlit.app`.
5. Put that live link directly on your resume next to the project title.

---

## For Your Resume

**Suggested bullet points:**

```
Resume-JD Match Analyzer | Python, Scikit-learn, Streamlit, NLP
• Built a tool using TF-IDF and cosine similarity to score resume-JD alignment
  and identify missing keywords, addressing a real gap in my own placement prep.
• Implemented PDF text extraction (PyPDF2) supporting both pasted text and
  uploaded resume/JD files.
• Deployed an interactive Streamlit dashboard with live match scoring and
  keyword gap highlighting.
```

## Interview Talking Points

- **Why you built it:** Practical problem from your own placement search — genuinely defensible, not a tutorial clone.
- **How it works technically:** Be ready to explain TF-IDF (term frequency-inverse document frequency — weighs words by how unique/important they are to a document, not just how often they appear) and cosine similarity (measures the angle between two vectors, not their magnitude — good for text where document length varies).
- **Limitation to be upfront about if asked:** TF-IDF is a bag-of-words approach — it doesn't understand context or synonyms (e.g., "ML" vs "machine learning" are treated as different terms). A natural follow-up improvement is using word embeddings (e.g., sentence-transformers) for semantic matching instead of exact keyword overlap. Mentioning this proactively shows you understand the technique's boundaries, which interviewers respect.
- **Extension idea if asked "what would you improve":** Weight keyword importance differently (e.g., skills vs soft skills), or fine-tune on a labeled dataset of "good match" vs "bad match" resume-JD pairs for a supervised scoring model instead of pure similarity.
