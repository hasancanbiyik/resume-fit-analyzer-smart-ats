import os
import streamlit as st
import logging
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime
import time
import functools
import PyPDF2 as pdf
import re, string, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from html import escape

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.68"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "25"))
DEFAULT_PORT = int(os.getenv("PORT", "8501"))

# ------------------------------
# Utility Functions
# ------------------------------

def sanitize_text_input(text: str, max_length: int = 50000) -> str:
    """Sanitize and validate text input."""
    if not text or not isinstance(text, str):
        return ""
    
    # Limit length
    text = text[:max_length]
    
    # Remove potentially harmful content
    text = escape(text)  # HTML escape
    
    # Basic cleanup
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Remove control chars
    
    return text.strip()

def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file before processing."""
    if not uploaded_file:
        return False, "No file uploaded"
    
    # Check file size (10MB limit)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File size exceeds 10MB limit"
    
    # Check file type
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, "Valid file"

def input_pdf_text(uploaded_file) -> Tuple[str, Optional[str]]:
    """Extract text from PDF with detailed error reporting."""
    # Validate file first
    is_valid, validation_message = validate_file(uploaded_file)
    if not is_valid:
        return "", validation_message
    
    try:
        reader = pdf.PdfReader(uploaded_file)
        if reader.is_encrypted:
            return "", "Encrypted PDFs are not supported. Please upload an unencrypted version."
        
        text = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                text.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                continue
                
        full_text = "\n".join(text)
        if len(full_text.strip()) < 100:
            return full_text, "Warning: Very little text was extracted. Please ensure your PDF contains readable text."
        
        return full_text, None
        
    except pdf.errors.PdfReadError as e:
        logger.error(f"PDF read error: {e}")
        return "", "Unable to read PDF file. It may be corrupted or in an unsupported format."
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {e}")
        return "", "An unexpected error occurred while processing your PDF."

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text: str):
    # light sentence split
    parts = re.split(r"[.\n\r]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]

def monitor_performance(func_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{func_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

# ------------------------------
# Model Loading
# ------------------------------

@st.cache_resource
def load_embedder():
    """Load embedding model with error handling and progress."""
    try:
        with st.spinner("Loading AI model (first time may take 30+ seconds)..."):
            model = SentenceTransformer(DEFAULT_EMBED_MODEL)
            logger.info(f"Successfully loaded model: {DEFAULT_EMBED_MODEL}")
            return model
    except Exception as e:
        logger.error(f"Failed to load model {DEFAULT_EMBED_MODEL}: {e}")
        st.error(f"Failed to load AI model. Please refresh the page or contact support.")
        st.stop()

# ------------------------------
# Core Analysis Functions
# ------------------------------

def extract_top_keywords(jd_text: str, top_k: int = 20):
    # Use TF-IDF to find salient n-grams in the JD
    doc = jd_text if jd_text.strip() else ""
    sents = split_sentences(doc) or [doc]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(sents)
    scores = np.asarray(X.mean(axis=0)).ravel()
    feats = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]

    out = []
    for idx in order:
        token = feats[idx]
        if any(ch.isdigit() for ch in token):
            continue
        if len(token) < 3:
            continue
        t = token.strip()
        if t not in out:
            out.append(t)
        if len(out) >= top_k:
            break
    # also expand bigrams into unigrams so coverage finds them
    expanded = set(out)
    for k in out:
        if " " in k:
            for p in k.split():
                if len(p) > 2:
                    expanded.add(p)
    return list(expanded)[:top_k]

def _present(resume_norm: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in resume_norm
    return re.search(rf"\b{re.escape(phrase)}\b", resume_norm) is not None

def exact_coverage(resume_text: str, jd_terms):
    rnorm = normalize(resume_text)
    matched, missing = [], []
    for kw in jd_terms:
        hit = _present(rnorm, kw)
        (matched if hit else missing).append(kw)
    cov = len(matched) / max(len(jd_terms), 1)
    return cov, matched, missing

def semantic_coverage(resume_text: str, jd_terms, model: SentenceTransformer, threshold: float = 0.68):
    """
    For each remaining JD term/phrase, mark as matched if ANY resume sentence is semantically similar.
    """
    if not jd_terms:
        return 0.0, [], []

    res_sents = split_sentences(resume_text) or [resume_text]
    emb_res = model.encode(res_sents, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    emb_terms = model.encode(jd_terms, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    sims = util.cos_sim(emb_terms, emb_res).cpu().numpy()  # [terms x sentences]

    matched_terms, missing_terms = [], []
    for i, term in enumerate(jd_terms):
        best = sims[i].max()
        if best >= threshold:
            matched_terms.append(term)
        else:
            missing_terms.append(term)

    coverage_ratio = len(matched_terms) / max(len(jd_terms), 1)
    return float(coverage_ratio), matched_terms, missing_terms

@monitor_performance("semantic_analysis")
def semantic_score(resume_text: str, jd_text: str, model: SentenceTransformer):
    # Embed JD sentences and Resume sentences, then average best-match per JD sentence
    jd_sents = split_sentences(jd_text) or [jd_text]
    res_sents = split_sentences(resume_text) or [resume_text]

    if not jd_sents or not res_sents:
        return 0.0, []

    emb_jd = model.encode(jd_sents, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    emb_res = model.encode(res_sents, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    sims_matrix = util.cos_sim(emb_jd, emb_res).cpu().numpy()  # [len(jd) x len(res)]
    per_jd_max = sims_matrix.max(axis=1)                        # best resume sentence for each JD sentence
    score = float(per_jd_max.mean())                            # 0..1
    return score, per_jd_max.tolist()

# ------------------------------
# Bullet Generation Functions
# ------------------------------

ACTION_VERBS = [
    "Developed", "Implemented", "Optimized", "Automated", "Designed",
    "Built", "Enhanced", "Led", "Streamlined", "Architected", 
    "Delivered", "Created", "Managed", "Analyzed", "Integrated"
]

IMPACT_TEMPLATES = [
    "improving {metric} by {value}%",
    "reducing {process} time by {value}%", 
    "increasing {outcome} efficiency by {value}%",
    "enhancing {quality} with {value}% fewer issues",
    "achieving {value}% improvement in {performance}",
    "delivering {value}% faster {process}"
]

METRIC_MAPPINGS = {
    "data": ["accuracy", "processing speed", "data quality"],
    "machine learning": ["model performance", "prediction accuracy", "training efficiency"],
    "automation": ["manual effort", "processing time", "workflow efficiency"],
    "api": ["response time", "throughput", "system reliability"],
    "database": ["query performance", "data retrieval", "system uptime"],
    "cloud": ["deployment speed", "infrastructure costs", "scalability"],
    "analytics": ["reporting accuracy", "insight generation", "dashboard performance"]
}

def get_relevant_metrics(skill: str) -> list:
    """Get contextually relevant metrics for a skill."""
    skill_lower = skill.lower()
    for key, metrics in METRIC_MAPPINGS.items():
        if key in skill_lower or any(k in skill_lower for k in key.split()):
            return metrics
    return ["performance", "efficiency", "quality"]

def best_resume_context(resume_text: str, query: str, model: SentenceTransformer, top_k: int = 2):
    """Return top_k resume sentences most similar to the query term/phrase."""
    res_sents = split_sentences(resume_text) or [resume_text]
    if not res_sents or not query.strip():
        return []
    
    try:
        emb_res = model.encode(res_sents, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        emb_q = model.encode([query], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        sims = util.cos_sim(emb_q, emb_res).cpu().numpy()[0]
        
        # Only return contexts with reasonable similarity (>0.3)
        valid_indices = [(i, sim) for i, sim in enumerate(sims) if sim > 0.3]
        if not valid_indices:
            return []
            
        # Sort by similarity and take top_k
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in valid_indices[:top_k]]
        
        return [res_sents[i] for i in top_indices]
    except Exception:
        return []

def titleize_skill(s: str) -> str:
    """Properly capitalize skill names."""
    s = s.strip()
    # Handle acronyms and special cases
    upper_cases = {"nlp", "ml", "ai", "llm", "sql", "aws", "gcp", "api", "rag", "etl", "ci/cd", "rest", "json", "xml"}
    special_cases = {
        "javascript": "JavaScript",
        "typescript": "TypeScript", 
        "nodejs": "Node.js",
        "reactjs": "React.js",
        "postgresql": "PostgreSQL",
        "mongodb": "MongoDB"
    }
    
    s_lower = s.lower()
    if s_lower in upper_cases:
        return s.upper()
    elif s_lower in special_cases:
        return special_cases[s_lower]
    else:
        return s.title()

def extract_context_keywords(context: str) -> str:
    """Extract the most relevant keywords from context for bullet generation."""
    if not context:
        return "existing systems"
    
    # Simple keyword extraction - look for technical terms, tools, domains
    words = context.lower().split()
    technical_indicators = [
        "system", "application", "platform", "framework", "tool", "service",
        "database", "api", "model", "algorithm", "pipeline", "dashboard"
    ]
    
    relevant_words = []
    for word in words:
        # Clean word of punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        if (len(clean_word) > 3 and 
            (any(indicator in clean_word for indicator in technical_indicators) or
             clean_word in context[:50])):  # Prioritize words from beginning of context
            relevant_words.append(clean_word)
            if len(relevant_words) >= 3:
                break
    
    if relevant_words:
        return " ".join(relevant_words[:2])  # Use up to 2 most relevant words
    
    # Fallback: use first few meaningful words
    meaningful_words = [w for w in words[:5] if len(w) > 3]
    return " ".join(meaningful_words[:2]) if meaningful_words else "existing systems"

def generate_impact_phrase(skill: str) -> str:
    """Generate contextually appropriate impact phrase."""
    import random
    
    relevant_metrics = get_relevant_metrics(skill)
    template = random.choice(IMPACT_TEMPLATES)
    
    # Select appropriate metric and value ranges
    metric = random.choice(relevant_metrics)
    
    # Value ranges based on metric type
    if "time" in template or "speed" in template:
        value = random.choice([15, 20, 25, 30, 35, 40])
    elif "cost" in template:
        value = random.choice([10, 15, 20, 25])
    else:
        value = random.choice([20, 25, 30, 35, 40, 45])
    
    # Format the template
    try:
        if "{metric}" in template and "{value}" in template:
            return template.format(metric=metric, value=value)
        elif "{process}" in template and "{value}" in template:
            return template.format(process=metric.split()[0] if metric.split() else "processing", value=value)
        elif "{outcome}" in template and "{value}" in template:
            return template.format(outcome=metric.split()[0] if metric.split() else "system", value=value)
        elif "{quality}" in template and "{value}" in template:
            return template.format(quality=metric, value=value)
        elif "{performance}" in template and "{value}" in template:
            return template.format(performance=metric, value=value)
        else:
            return template.format(value=value)
    except:
        return f"improving {metric} by {value}%"

def suggest_bullets(missing_terms, resume_text, model, max_bullets=4):
    """Create up to max_bullets professional bullets grounded in resume context."""
    if not missing_terms or not resume_text.strip():
        return []
    
    bullets = []
    # Filter and prioritize terms
    terms = [t for t in missing_terms if len(t.strip()) > 2 and not t.isdigit()]
    
    # Prioritize multi-word terms (likely more specific/valuable)
    terms.sort(key=lambda x: (len(x.split()), len(x)), reverse=True)
    terms = terms[:max_bullets]
    
    import random
    used_actions = set()
    
    for i, term in enumerate(terms):
        # Get contextual information from resume
        contexts = best_resume_context(resume_text, term, model, top_k=2)
        
        # Select unused action verb
        available_actions = [a for a in ACTION_VERBS if a not in used_actions]
        if not available_actions:
            available_actions = ACTION_VERBS
            used_actions.clear()
        
        action = random.choice(available_actions)
        used_actions.add(action)
        
        # Process skill name
        skill = titleize_skill(term)
        
        # Generate impact phrase
        impact = generate_impact_phrase(term)
        
        # Extract relevant context
        if contexts:
            context_summary = extract_context_keywords(contexts[0])
        else:
            context_summary = "enterprise applications"
        
        # Generate bullet with better structure
        bullets.append(f"{action} {skill} solutions for {context_summary}, {impact}")
    
    return bullets

def display_bullet_suggestions(missing, resume_text, model):
    """Display the bullet suggestions section in Streamlit."""
    st.subheader("✏️ Professional Resume Bullets")
    st.caption("AI-generated bullets based on missing skills and your experience context")
    
    if not missing:
        st.success("🎉 No major missing skills detected — your resume looks well-aligned!")
        return
    
    with st.spinner("Generating contextual bullets..."):
        bullets = suggest_bullets(missing, resume_text, model, max_bullets=4)
    
    if bullets:
        st.markdown("**Suggested additions to strengthen your resume:**")
        for i, bullet in enumerate(bullets, 1):
            st.markdown(f"{i}. {bullet}")
        
        # Download functionality
        bullet_text = "\n".join(f"• {bullet}" for bullet in bullets)
        st.download_button(
            "📄 Download as Text File",
            data=bullet_text,
            file_name=f"resume_bullets_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            help="Download bullets as a text file for easy copy-paste"
        )
        
        with st.expander("💡 How to use these bullets"):
            st.write("""
            **Tips for using these AI-generated bullets:**
            - Replace placeholder percentages with your actual achievements
            - Customize the context to match your specific projects
            - Ensure each bullet starts with a strong action verb
            - Quantify impact wherever possible with real metrics
            - Tailor the language to match the job description tone
            """)
    else:
        st.info("Unable to generate contextual bullets. Consider adding more detailed experience descriptions to your resume.")

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Smart ATS (No API Key)", layout="wide")
st.title("🧠 Smart ATS — No API Key Needed")
st.caption("Embeddings (semantic) + ATS coverage (hybrid). 100% free, CPU-only.")

col1, col2 = st.columns(2)
with col1:
    jd = st.text_area("Paste the Job Description", height=240, placeholder="Paste JD here...")
    # Sanitize JD input
    jd = sanitize_text_input(jd)

with col2:
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    resume_text = ""
    error_message = None
    
    if uploaded_file is not None:
        resume_text, error_message = input_pdf_text(uploaded_file)
        if error_message:
            if "Warning:" in error_message:
                st.warning(error_message)
            else:
                st.error(error_message)
                resume_text = ""  # Clear text if there was an error
    else:
        resume_text = st.text_area("…or paste resume text", height=240, placeholder="Paste resume text here...")
        resume_text = sanitize_text_input(resume_text)

threshold = st.slider(
    "Semantic coverage threshold",
    0.50, 0.85, DEFAULT_THRESHOLD, 0.01,
    help="Higher = stricter matching. Typical 0.65–0.75."
)

analyze = st.button("🔎 Analyze Fit")

if analyze:
    if not jd.strip():
        st.error("Please provide the Job Description.")
        st.stop()
    if not resume_text.strip():
        st.error("Please upload or paste your resume.")
        st.stop()

    with st.spinner("Loading model and scoring…"):
        model = load_embedder()

        # 1) Extract JD terms (ATS-style)
        jd_keywords = extract_top_keywords(jd, top_k=DEFAULT_TOP_K)

        # 2) Exact coverage first
        cov_exact, matched_exact, missing_exact = exact_coverage(resume_text, jd_keywords)

        # 3) Semantic coverage on the remaining missing terms
        cov_sem, matched_sem, missing_sem = semantic_coverage(resume_text, missing_exact, model, threshold=threshold)

        # Combine matched/missing for final coverage
        matched = matched_exact + matched_sem
        missing = missing_sem
        coverage_ratio = len(matched) / max(len(jd_keywords), 1)

        # 4) Sentence-level semantic similarity (overall JD ↔ resume)
        sem_ratio, jd_sentence_scores = semantic_score(resume_text, jd, model)

        # Final score (0–100): balance meaning + keywords
        final = 100.0 * (0.7 * sem_ratio + 0.3 * coverage_ratio)

    st.subheader("📊 Match Score")
    st.metric("Overall (0–100)", value=round(final, 1))
    st.progress(min(max(int(final), 0), 100))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top JD Keywords (ATS-style)**")
        st.write(", ".join(jd_keywords) if jd_keywords else "—")
    with c2:
        st.markdown("**Matched in Resume**")
        st.write(", ".join(matched) if matched else "—")

    st.markdown("**Missing / Underrepresented Skills**")
    if missing:
        st.warning(", ".join(missing))
        display_bullet_suggestions(missing, resume_text, model)
    else:
        st.success("No major missing skills detected.")
        # Still show bullet suggestions even when no missing skills
        display_bullet_suggestions(missing, resume_text, model)

    # Download JSON report
    report = {
        "score": final,
        "semantic_ratio": sem_ratio,
        "coverage_ratio": coverage_ratio,
        "jd_keywords": jd_keywords,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "debug": {
            "jd_sentence_match_scores": jd_sentence_scores,
            "resume_chars": len(resume_text),
            "jd_chars": len(jd),
            "threshold": threshold,
            "cov_exact": cov_exact,
            "cov_sem_on_missing": cov_sem
        }
    }
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name="match_report.json",
        mime="application/json"
    )

st.divider()
with st.expander("How this works"):
    st.write("""
**Hybrid coverage**:
1) We check **exact keyword/phrase hits** (precise, zero ambiguity).
2) For remaining terms, we use **semantic embeddings** to see if any resume sentence means roughly the same thing (cosine ≥ threshold).

**Semantic similarity**:
- Independently, we compare every JD sentence to the best matching resume sentence and average these scores.

**Final score**:
- 70% sentence-level semantic similarity + 30% term coverage.
    """)
