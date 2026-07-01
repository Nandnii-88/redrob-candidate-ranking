#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge
========================================================================
Produces a top-100 ranked CSV from candidates.jsonl for the "Senior AI
Engineer — Founding Team" job description.

Design summary (see README.md / deck for full rationale):

  final_score = behavioral_modifier(candidate) * (
        0.30 * semantic_fit          (TF-IDF cosine sim: profile text vs JD)
      + 0.25 * skill_fit             (explicit must-have / nice-to-have skills,
                                       trust-weighted by endorsements + duration)
      + 0.20 * title_and_role_fit    (regex/keyword classifier over title +
                                       career history, screens out keyword
                                       stuffers and off-track titles)
      + 0.15 * career_substance_fit  (production vs research-only, consulting
                                       penalty, seniority trajectory)
      + 0.10 * logistics_fit         (location, notice period, relocation)
  )

  Honeypots (internally-inconsistent profiles) are detected with deterministic
  rules and hard-capped near zero so they never reach the top 100.

Everything here is pure-Python + scikit-learn TF-IDF (CPU only, no network,
no GPU). Full 100K-candidate run completes in well under the 5-minute / 16GB
budget (see benchmark in README.md).

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import gzip
import json
import re
import sys
import time
from datetime import date, datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Job description (condensed into a query document + explicit rule signals)
# --------------------------------------------------------------------------

JD_QUERY_TEXT = """
Senior AI Engineer founding team AI engineering ranking retrieval matching
embeddings sentence-transformers OpenAI embeddings BGE E5 vector database
Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS hybrid search
production deployment embedding drift index refresh retrieval quality
regression evaluation framework NDCG MRR MAP A/B testing offline online
correlation Python production code quality learning to rank XGBoost neural
LLM fine-tuning LoRA QLoRA PEFT recommendation system search marketplace
recruiting HR tech NLP information retrieval distributed systems large scale
inference shipped end to end ranking search recommendation system real users
scale product company mentoring evaluation infrastructure recruiter
engagement metrics
"""

MUST_HAVE_SKILLS = {
    "embeddings": 1.0, "sentence-transformers": 1.0, "sentence transformers": 1.0,
    "openai embeddings": 0.9, "bge": 0.8, "e5": 0.6, "vector search": 1.0,
    "vector database": 1.0, "pinecone": 0.8, "weaviate": 0.8, "qdrant": 0.8,
    "milvus": 0.8, "opensearch": 0.7, "elasticsearch": 0.9, "faiss": 0.9,
    "hybrid search": 0.9, "semantic search": 0.9, "retrieval": 1.0,
    "ranking": 1.0, "learning to rank": 0.9, "ndcg": 0.8, "mrr": 0.7,
    "map": 0.5, "a/b testing": 0.7, "python": 1.0, "rag": 0.9,
    "recommendation system": 0.9, "recommender systems": 0.9, "nlp": 0.8,
    "information retrieval": 0.9,
}

NICE_TO_HAVE_SKILLS = {
    "lora": 0.6, "qlora": 0.6, "peft": 0.5, "fine-tuning llms": 0.7,
    "fine-tuning": 0.5, "xgboost": 0.5, "distributed systems": 0.5,
    "inference optimization": 0.5, "open source": 0.3, "kubernetes": 0.2,
    "spark": 0.2, "airflow": 0.2,
}

# Titles that strongly indicate fit vs. strongly indicate a trap/mismatch.
GOOD_TITLE_PATTERNS = re.compile(
    r"\b(ai|ml|machine learning|nlp|search|ranking|retrieval|recommend\w*|"
    r"applied scientist|research engineer)\b.*\b(engineer|scientist|lead|"
    r"architect)\b|\bsenior\s+(ai|ml|software)\s+engineer\b",
    re.IGNORECASE,
)
ENGINEER_TITLE_PATTERN = re.compile(
    r"\b(engineer|developer|scientist|architect|programmer)\b", re.IGNORECASE
)
BAD_TITLE_PATTERNS = re.compile(
    r"\b(hr\s*manager|recruiter|marketing|sales|content writer|business "
    r"analyst|product manager|project manager|hr business partner|talent "
    r"acquisition|account manager|customer success)\b",
    re.IGNORECASE,
)

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

PURE_RESEARCH_PATTERNS = re.compile(
    r"\b(research scientist|phd researcher|postdoc|research fellow|academic"
    r"\s+lab|research only)\b",
    re.IGNORECASE,
)
PRODUCTION_PATTERNS = re.compile(
    r"\b(deployed|production|shipped|launched|scale[d]?|real users|live "
    r"system|served \d|users? per|traffic|latency|throughput)\b",
    re.IGNORECASE,
)
FRAMEWORK_TUTORIAL_PATTERNS = re.compile(
    r"\b(langchain tutorial|built a demo|how i used|tutorial project)\b",
    re.IGNORECASE,
)

PREFERRED_LOCATIONS = {"pune", "noida", "delhi", "delhi ncr", "gurugram",
                        "gurgaon", "new delhi"}
TIER1_INDIA_LOCATIONS = {"bangalore", "bengaluru", "mumbai", "hyderabad",
                          "chennai", "pune", "noida", "delhi", "gurugram",
                          "gurgaon", "new delhi"}


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def candidate_text_blob(c):
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts)


def skill_fit_score(c):
    skills = c.get("skills", [])
    skill_index = {}
    for s in skills:
        name = s.get("name", "").lower().strip()
        if not name:
            continue
        # trust weight: real usage (duration + endorsements), not just listed
        dur = s.get("duration_months", 0) or 0
        end = s.get("endorsements", 0) or 0
        prof = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85,
                "expert": 1.0}.get(s.get("proficiency", ""), 0.5)
        trust = min(1.0, 0.3 + 0.5 * min(dur / 24.0, 1.0) + 0.2 * min(end / 20.0, 1.0))
        skill_index[name] = prof * trust

    # No real candidate has all 30 must-have keywords (they're near-synonyms
    # covering many possible tech-stack choices, by design — see JD: "we
    # don't care which model/vector-db, we care about the operational
    # experience"). We saturate against a realistic target: ~4-5 strong,
    # well-evidenced hits on the core retrieval/ranking/embeddings stack is
    # a full-credit match.
    MUST_SATURATION = 5.5
    must_hit = 0.0
    for name, weight in MUST_HAVE_SKILLS.items():
        if name in skill_index:
            must_hit += weight * skill_index[name]
        else:
            # partial credit for substring match (e.g. "vector database" in a longer name)
            for k, v in skill_index.items():
                if name in k or k in name:
                    must_hit += weight * v * 0.6
                    break
    must_score = min(1.0, must_hit / MUST_SATURATION)

    NICE_SATURATION = 1.8
    nice_hit = sum(w * skill_index.get(n, 0.0) for n, w in NICE_TO_HAVE_SKILLS.items())
    nice_score = min(1.0, nice_hit / NICE_SATURATION)

    # Keyword-stuffer penalty: many must-have names *listed* but low average
    # trust (i.e. low endorsements/duration => likely padded skill list).
    listed_musthaves = [n for n in MUST_HAVE_SKILLS if n in skill_index]
    if len(listed_musthaves) >= 6:
        avg_trust = np.mean([skill_index[n] for n in listed_musthaves])
        if avg_trust < 0.45:
            must_score *= 0.5  # stuffed list, low real signal

    return 0.85 * must_score + 0.15 * nice_score


def title_role_fit(c):
    title = (c.get("profile", {}).get("current_title") or "")
    titles_all = [title] + [j.get("title", "") for j in c.get("career_history", [])]
    blob = " ".join(titles_all)

    if BAD_TITLE_PATTERNS.search(title):
        base = 0.05
    elif GOOD_TITLE_PATTERNS.search(blob):
        base = 1.0
    elif ENGINEER_TITLE_PATTERN.search(title):
        base = 0.55
    else:
        base = 0.25

    return base


def career_substance_fit(c):
    descs = " ".join(j.get("description", "") for j in c.get("career_history", []))
    companies = [j.get("company", "").lower() for j in c.get("career_history", [])]
    current_company = (c.get("profile", {}).get("current_company") or "").lower()
    all_companies = set(companies + [current_company])

    score = 0.5
    if PRODUCTION_PATTERNS.search(descs):
        score += 0.30
    if PURE_RESEARCH_PATTERNS.search(descs) and not PRODUCTION_PATTERNS.search(descs):
        score -= 0.40
    if FRAMEWORK_TUTORIAL_PATTERNS.search(descs):
        score -= 0.20

    consulting_only = all_companies and all_companies.issubset(CONSULTING_FIRMS | {""})
    # remove blank entries before checking "only"
    real_companies = {x for x in all_companies if x}
    if real_companies and real_companies.issubset(CONSULTING_FIRMS):
        score -= 0.35

    # job-hopper penalty: many short (<18mo) stints
    durations = [j.get("duration_months", 0) or 0 for j in c.get("career_history", [])]
    if len(durations) >= 3:
        short = sum(1 for d in durations if d < 18)
        if short / len(durations) > 0.6:
            score -= 0.15

    return float(np.clip(score, 0.0, 1.0))


def experience_fit(c):
    yrs = c.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= yrs <= 9:
        return 1.0
    if 4 <= yrs < 5 or 9 < yrs <= 11:
        return 0.7
    if 2 <= yrs < 4 or 11 < yrs <= 14:
        return 0.4
    return 0.15


def logistics_fit(c):
    loc = (c.get("profile", {}).get("location") or "").lower().strip()
    country = (c.get("profile", {}).get("country") or "").lower().strip()
    sig = c.get("redrob_signals", {})
    notice = sig.get("notice_period_days", 60) or 60
    relocate = sig.get("willing_to_relocate", False)

    if loc in PREFERRED_LOCATIONS:
        loc_score = 1.0
    elif loc in TIER1_INDIA_LOCATIONS or "india" in country:
        loc_score = 0.75 if (loc in TIER1_INDIA_LOCATIONS or relocate) else 0.55
    elif relocate:
        loc_score = 0.4
    else:
        loc_score = 0.15

    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.65
    else:
        notice_score = 0.35

    return 0.7 * loc_score + 0.3 * notice_score


def behavioral_modifier(c):
    sig = c.get("redrob_signals", {})
    today = date(2026, 6, 30)

    last_active = parse_date(sig.get("last_active_date"))
    if last_active is None:
        recency_score = 0.6
    else:
        days = (today - last_active).days
        if days <= 14:
            recency_score = 1.0
        elif days <= 30:
            recency_score = 0.9
        elif days <= 90:
            recency_score = 0.6
        elif days <= 180:
            recency_score = 0.35
        else:
            recency_score = 0.15

    resp = sig.get("recruiter_response_rate", 0.5)
    resp = 0.5 if resp is None else resp
    open_flag = 1.0 if sig.get("open_to_work_flag", False) else 0.85
    interview_rate = sig.get("interview_completion_rate", 0.7)
    interview_rate = 0.7 if interview_rate is None else interview_rate

    verified = (
        0.5
        + 0.2 * (1 if sig.get("verified_email") else 0)
        + 0.15 * (1 if sig.get("verified_phone") else 0)
        + 0.15 * (1 if sig.get("linkedin_connected") else 0)
    )

    raw = (
        0.35 * recency_score
        + 0.30 * resp
        + 0.15 * open_flag
        + 0.10 * interview_rate
        + 0.10 * verified
    )
    # Map to a multiplier in [0.55, 1.15] so behavior nudges rank without
    # letting a high-engagement irrelevant candidate dominate skill fit.
    return 0.55 + 0.60 * raw


def is_honeypot(c):
    """Deterministic checks for internally-impossible profiles."""
    p = c.get("profile", {})
    yrs = p.get("years_of_experience", 0) or 0
    current_year = 2026

    # Skill: "expert" proficiency claimed with ~0 duration.
    expert_zero_dur = sum(
        1 for s in c.get("skills", [])
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) <= 2
    )
    if expert_zero_dur >= 3:
        return True

    # Career: tenure at a role longer than time since a very recently
    # "founded" feeling company, or total career_history duration grossly
    # exceeding stated years_of_experience.
    total_months = sum(j.get("duration_months", 0) or 0 for j in c.get("career_history", []))
    if yrs > 0 and total_months > (yrs * 12 + 24):
        return True

    # Education ends after experience supposedly already started for 5+ yrs,
    # or end_year in the far future relative to today with high experience.
    edu = c.get("education", [])
    for e in edu:
        end_year = e.get("end_year")
        if end_year and yrs >= 5 and end_year > current_year:
            return True

    # Overlapping full-time roles (more than ~1 concurrent "is_current" or
    # heavily overlapping date ranges) is a soft signal; skip — too noisy
    # without full interval-overlap logic given time budget.

    return False


def score_candidate(c):
    if is_honeypot(c):
        return -1.0, "honeypot: internally inconsistent profile (likely synthetic trap)"
    return None, None


def build_reasoning(c, scores):
    p = c.get("profile", {})
    title = p.get("current_title", "Unknown title")
    yrs = p.get("years_of_experience", 0)
    loc = p.get("location", "Unknown")
    sig = c.get("redrob_signals", {})
    resp = sig.get("recruiter_response_rate")
    notice = sig.get("notice_period_days")

    bits = [f"{title}, {yrs} yrs, based in {loc}."]
    if scores["skill"] > 0.55:
        bits.append("Strong match on must-have retrieval/ranking/embeddings skills.")
    elif scores["skill"] > 0.25:
        bits.append("Partial skill match on core retrieval/ranking stack.")
    else:
        bits.append("Weak overlap with required embeddings/retrieval skill set.")

    if scores["career"] > 0.7:
        bits.append("Career history shows production deployment experience.")
    elif scores["career"] < 0.35:
        bits.append("Limited evidence of production (vs. research-only/consulting) work.")

    if resp is not None:
        bits.append(f"Recruiter response rate {resp:.0%}, notice period {notice}d.")

    return " ".join(bits)[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    opener = gzip.open if args.candidates.endswith(".gz") else open
    candidates = []
    with opener(args.candidates, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    texts = [candidate_text_blob(c) for c in candidates]
    vectorizer = TfidfVectorizer(
        max_features=40000, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    matrix = vectorizer.fit_transform(texts + [JD_QUERY_TEXT])
    jd_vec = matrix[-1]
    cand_matrix = matrix[:-1]
    sims = (cand_matrix @ jd_vec.T).toarray().ravel()
    sim_min, sim_max = sims.min(), sims.max()
    sims_norm = (sims - sim_min) / (sim_max - sim_min + 1e-9)
    print(f"TF-IDF semantic fit computed in {time.time()-t1:.1f}s", file=sys.stderr)

    t2 = time.time()
    results = []
    for i, c in enumerate(candidates):
        honeypot_score, honeypot_reason = score_candidate(c)
        if honeypot_score is not None:
            results.append({
                "candidate_id": c["candidate_id"],
                "score": -1.0,
                "reasoning": honeypot_reason,
                "honeypot": True,
            })
            continue

        skill = skill_fit_score(c)
        title = title_role_fit(c)
        career = career_substance_fit(c)
        exp = experience_fit(c)
        logi = logistics_fit(c)
        semantic = sims_norm[i]

        base = (
            0.30 * semantic + 0.25 * skill + 0.20 * title
            + 0.15 * career + 0.05 * exp + 0.05 * logi
        )
        mod = behavioral_modifier(c)
        final = base * mod

        reasoning = build_reasoning(
            c, {"skill": skill, "career": career, "title": title}
        )
        results.append({
            "candidate_id": c["candidate_id"],
            "score": final,
            "reasoning": reasoning,
            "honeypot": False,
        })
    print(f"Scored all candidates in {time.time()-t2:.1f}s", file=sys.stderr)

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[: args.top_k]

    # normalize scores into a clean 0-1 descending range for readability,
    # preserving strict ordering (ties broken by candidate_id ascending,
    # already guaranteed by the sort above).
    max_s = top[0]["score"] if top else 1.0
    min_s = top[-1]["score"] if top else 0.0
    span = max(max_s - min_s, 1e-6)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_norm = 1.01
        rows = []
        for r in top:
            norm = 0.05 + 0.94 * (r["score"] - min_s) / span
            norm = min(norm, prev_norm)  # enforce non-increasing
            prev_norm = norm
            rows.append([r["candidate_id"], f"{norm:.4f}", r["reasoning"]])

        # Rounding can create score ties between rows that weren't adjacent by
        # candidate_id pre-rounding. Re-sort within each tied-score run by
        # candidate_id ascending so the spec's tiebreak rule is satisfied.
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1][1] == rows[i][1]:
                j += 1
            if j > i:
                rows[i:j + 1] = sorted(rows[i:j + 1], key=lambda x: x[0])
            i = j + 1

        for rank, row in enumerate(rows, start=1):
            writer.writerow([row[0], rank, row[1], row[2]])

    n_honeypots = sum(1 for r in top if r["honeypot"])
    print(f"Wrote top {len(top)} to {args.out} ({time.time()-t0:.1f}s total). "
          f"Honeypots in top {args.top_k}: {n_honeypots}", file=sys.stderr)


if __name__ == "__main__":
    main()
