# 🧠 Smart ATS — Resume & JD Matcher

Smart ATS is a **Streamlit-based application** that evaluates how well a resume matches a given Job Description (JD).  
It uses a **hybrid approach** that combines keyword extraction (ATS-style) with **semantic similarity** powered by embeddings — all without requiring any paid API keys.

---

## Features
1. **ATS-style Keyword Extraction**  
   - Uses TF-IDF to extract top n-grams from the JD.  
   - Expands bigrams into unigrams for broader coverage.

2. **Hybrid Matching (Exact + Semantic)**  
   - First checks for exact matches of JD terms in the resume.  
   - Then applies sentence-transformer embeddings (`all-MiniLM-L6-v2`) to find semantically similar matches.  

3. **Sentence-Level Similarity**  
   - Compares every JD sentence against resume sentences to calculate semantic alignment.

4. **Scoring System**  
   - Final score (0–100) = **70% semantic similarity + 30% keyword coverage**.  

5. **Detailed Report**  
   - Outputs a downloadable `match_report.json` containing scores, matched and missing skills, and debug info.  

6. **Resume Bullet Suggestions**  
   - Generates contextual, professional bullet points to strengthen your resume based on missing skills.

---

## Tech Stack
- **Python+**
- [Streamlit](https://streamlit.io/) – UI framework  
- [PyPDF2](https://pypi.org/project/PyPDF2/) – Resume text extraction  
- [scikit-learn](https://scikit-learn.org/stable/) – TF-IDF keyword extraction  
- [SentenceTransformers](https://www.sbert.net/) – Semantic embeddings  
- **NumPy, Regex, JSON** utilities  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/smart-ats.git
cd smart-ats
pip install -r requirements.txt
```

---

##  Usage
Run the Streamlit app:

```bash
streamlit run app.py
```

Then:
1. Paste the Job Description.  
2. Upload your resume (PDF) or paste text.  
3. Adjust the semantic similarity threshold (default: 0.68).  
4. Click **Analyze Fit** to view results.  

---

## Output
- **Overall Match Score (0–100)**  
- **Top JD Keywords**  
- **Matched vs. Missing Skills**  
- **Suggested Resume Bullets**  
- Downloadable JSON report  

---

## Notes
- Works fully **offline** (CPU-only).  
- PDF parsing quality depends on resume formatting.  
- Suggested bullets are templates — customize with your own metrics.  

---

## License
MIT License © 2025
