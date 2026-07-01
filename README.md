# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A transparent, rule-grounded + TF-IDF semantic ranker for the "Senior AI
Engineer — Founding Team" job description. Built to be **explainable and
fast**, not a black box: every score component maps to a sentence a human
recruiter would actually say.

## TL;DR

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

- Runtime on the full 100,000-candidate pool: **~65–90 seconds**, CPU only,
  no GPU, no network calls, well inside the 5-minute / 16GB budget.
- 0 honeypots surfaced in the top 100 on the released pool (see `data/` for
  the analysis used to design the honeypot filter).

## Why this architecture

The JD is explicit that this is not a keyword-matching exercise: *"the
'right answer' is not 'find candidates whose skills section contains the
most AI keywords' — that's a trap we've explicitly built into the
dataset."* The dataset itself contains keyword stuffers, behavioral twins,
plain-language Tier-5s (great candidates who don't use the buzzwords), and
~80 honeypots with internally impossible profiles.

So instead of a single similarity score, the ranker scores **five
independent, human-auditable signals** and combines them with a behavioral
modifier, then runs a deterministic honeypot filter before anything is
ranked:

```
final_score = behavioral_modifier(candidate) × (
      0.30 × semantic_fit            TF-IDF cosine similarity, candidate text vs JD
    + 0.25 × skill_fit               must-have/nice-to-have skills, trust-weighted
                                      by endorsement count + duration (not just "listed")
    + 0.20 × title_and_role_fit      regex/keyword screen — the decisive signal
                                      against keyword-stuffer traps
    + 0.15 × career_substance_fit    production vs. research-only, consulting-only
                                      penalty, job-hopper penalty
    + 0.05 × experience_fit          5–9 yr band, soft-decaying outside it
    + 0.05 × logistics_fit           location (Pune/Noida/Tier-1 India), notice period
)
```

honeypots are detected with deterministic checks (e.g. "expert" proficiency
claimed with ~0 months' duration on 3+ skills; total career-history months
grossly exceeding stated years of experience) and forced to the bottom of
the ranking before the behavioral modifier is even computed.

### Why TF-IDF and not embeddings/LLM re-ranking

The submission spec is explicit: **no GPU, no hosted LLM calls during
ranking, 5-minute / 16GB CPU budget for 100K candidates**. An LLM-per-candidate
re-ranker cannot fit that budget; a `sentence-transformers` encoder forward
pass over 100K profiles is borderline on CPU-only hardware and adds a large,
hard-to-audit dependency for relatively little gain over a well-tuned
lexical signal *combined with explicit structured features* — which is what
actually catches the traps the JD describes (title mismatches, consulting-only
careers, research-without-production careers). TF-IDF cosine similarity is
used as one input among five, not the whole system, specifically so it can't
be gamed by keyword stuffing alone (the `title_and_role_fit` and `skill_fit`
trust-weighting components exist precisely to counter that).

If you want to swap in a local embedding model (e.g. a quantized `BGE-small`
ONNX export), the `semantic_fit` component is isolated in `rank.py` behind
a single function boundary — see `candidate_text_blob()` / the TF-IDF block
in `main()`.

### Skill-fit trust weighting (anti keyword-stuffing)

A skill being *listed* is weak evidence. A skill being listed **with real
endorsements and a multi-year duration** is strong evidence. Each skill's
contribution is scaled by:

```
trust = 0.3 + 0.5 × min(duration_months / 24, 1) + 0.2 × min(endorsements / 20, 1)
```

Candidates with 6+ must-have skills *listed* but a low average trust score
(the classic "I added 15 AI buzzwords to my skill list yesterday" pattern)
get an explicit penalty.

### Title/role fit (the decisive anti-trap signal)

The JD explicitly flags: *"A candidate who has all the AI keywords listed
as skills but whose title is 'Marketing Manager' is not a fit, no matter
how perfect their skill list looks."* `title_and_role_fit` checks the
current title and full career-history title trail against:

- **Bad-title patterns** (HR Manager, Recruiter, Marketing, Sales, Content
  Writer, Product/Project Manager, etc.) → hard floor, regardless of skill
  list.
- **Good-title patterns** (AI/ML/NLP/search/ranking/recommendation engineer
  or scientist titles) → full credit.
- Generic "Engineer/Developer/Scientist" titles get partial credit, refined
  by the other four components.

### Career-substance fit (research-only / consulting-only penalties)

Directly encodes two explicit JD disqualifiers:

- *"If you've spent your career in pure research environments… without any
  production deployment — we will not move forward."* → regex screen for
  production-language (deployed, shipped, scale, latency, real users) vs.
  research-only language, with a penalty when the latter dominates.
- *"People who have only worked at consulting firms (TCS, Infosys, Wipro,
  Accenture, Cognizant, Capgemini, etc.) in their entire career."* →
  explicit company-name set checked against the candidate's full employer
  history; only penalized if **every** employer is in that set (the JD
  carves out an exception for candidates currently at one of these firms
  with prior product-company experience, which this naturally allows since
  the penalty only fires when the *entire* history is consulting-only).

### Behavioral modifier (not a hard filter)

*"A perfect-on-paper candidate who hasn't logged in for 6 months and has a
5% recruiter response rate is, for hiring purposes, not actually
available. Down-weight them appropriately."* — implemented exactly as a
**down-weighting multiplier** (range ≈ 0.55–1.15), combining recency of
`last_active_date`, `recruiter_response_rate`, `open_to_work_flag`,
`interview_completion_rate`, and identity-verification signals. It is a
modifier, not a gate, so a highly engaged but skill-mismatched candidate
still can't outrank a strong-fit candidate who is merely moderately
engaged.

## Repository layout

```
.
├── rank.py                      # the full ranking pipeline (single file, audit-friendly)
├── requirements.txt
├── submission_metadata.yaml     # filled-in copy of the hackathon template
├── submission.csv               # top-100 output for the released JD + candidate pool
├── sandbox/
│   └── app.py                   # Streamlit sandbox — upload a small sample, get ranked CSV
├── notebooks/
│   └── eda.md                   # notes on honeypot patterns found during data exploration
└── README.md
```

## Reproducing the submission

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv   # -> "Submission is valid."
```

No pre-computation step is required — `rank.py` is the entire pipeline,
start to finish, in one process.

## Honeypot filter details

Three deterministic checks, applied before scoring (a flagged candidate is
sent to the bottom of the ranking regardless of skill/title match):

1. **Expert-with-no-time**: 3+ skills marked `proficiency: expert` with
   `duration_months ≤ 2`. Genuine expertise takes time; this combination is
   not internally consistent.
2. **Career-time overrun**: total `career_history` duration exceeds the
   candidate's stated `years_of_experience` by more than 2 years — i.e. the
   timeline as described is not physically possible without unexplained
   overlapping full-time roles.
3. **Education-after-experience impossibility**: a degree `end_year` in the
   future relative to a candidate already claiming 5+ years of experience.

These are intentionally conservative (low false-positive rate) rather than
exhaustive, in line with the spec's note that *"we expect a good ranking
system to naturally avoid them; you don't need to special-case them"* — the
filter is a backstop, not the primary defense (title/skill/career scoring
naturally pushes most honeypots down regardless).

## Known limitations / what we'd improve with more time

- The TF-IDF semantic component does not capture true paraphrase-level
  similarity (e.g. "built a system that finds similar job postings" vs.
  "recommendation system") as well as a dense embedding model would. We
  scoped this out due to the no-GPU / 5-minute budget for 100K candidates;
  a precomputed local ONNX embedding index would be the natural next step
  and is compatible with the architecture (swap the TF-IDF block for an
  embedding lookup at the same point in `main()`).
- Overlapping-employment honeypots (two concurrent `is_current: true` roles,
  or heavily overlapping date ranges) are not currently checked — flagged as
  a TODO in `rank.py`.
- Company-size/industry signals are read but not yet incorporated into
  `career_substance_fit`; a more thorough version would weight prior
  product-company experience by company size and known-AI-native-company
  lists.
