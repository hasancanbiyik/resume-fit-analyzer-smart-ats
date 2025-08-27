# Resume Fit Analyzer / Smart ATS

Smart ATS is a Streamlit-based application that evaluates how well a resume matches a given Job Description (JD).
It combines ATS-style keyword extraction with semantic similarity powered by embeddings — all offline, free, and API-keyless.

## Features

### ATS-Style Keyword Extraction
- TF-IDF extracts top keywords/phrases from the JD.
- Expands bigrams into unigrams for broader coverage.

### Hybrid Matching (Exact + Semantic)
- First checks for exact term matches in the resume.
- Then applies embeddings (**all-MiniLM-L6-v2**) for semantic matches (e.g., “NLP” ≈ “natural language processing”).

### Sentence-Level Similarity
- Compares JD sentences against resume sentences using cosine similarity.

### Scoring System
- Final score (0–100) = 70% semantic similarity + 30% keyword coverage (tunable).

### Detailed Report
- Downloadable `match_report.json` with scores, ratios, matched/missing skills, and debug info.

### Resume Bullet Suggestions
- Generates professional, context-aware resume bullets from missing skills (template-based, no LLM).

## Tech Stack

- **Python+**
- **Streamlit** – Interactive UI
- **PyPDF2** – PDF text extraction
- **scikit-learn** – TF-IDF keyword extraction
- **SentenceTransformers** – Embeddings (all-MiniLM-L6-v2)
- **NumPy, Regex, JSON, Pandas** utilities

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/smart-ats.git
cd smart-ats

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:

1. Paste the Job Description.
2. Upload your resume (PDF) or paste text.
3. Adjust the semantic similarity threshold (default: 0.70).
4. Click **Analyze Fit** to view results.

## Output

- Overall Match Score (0–100)
- Top JD Keywords (ATS-style)
- Matched vs. Missing Skills
- Suggested Resume Bullets (editable templates)
- Downloadable JSON report

## Notes

- Works fully offline (CPU-only).
- PDF parsing quality depends on resume formatting.
- Bullet suggestions are templates — customize with real metrics.
- Hidden `.venv/` folder is used for dependencies (add to `.gitignore`).

## Roadmap

- Sample JD picker for quick testing.
- Slider for score weighting (semantic vs coverage).
- Optional local LLM (via Ollama) for rewriting bullets.

## License

MIT License © 2025
