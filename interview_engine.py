"""
Core interview engine.
Talks to a local Ollama server for LLM calls, and uses faster-whisper for STT.

Question types supported:
  - fixed:                 static question, scored against rubric, optional probes
  - fixed_resume_aware:    static question, but probes may reference resume content
  - generated_from_resume: LLM generates N questions from resume text (cached on disk
                            keyed by resume hash, so repeat runs on the same resume
                            don't burn extra LLM calls)
  - eligibility:           static question, classified yes/no/unclear (NOT rubric-scored),
                            flagged for human review if it matches a disqualifying_answer.
                            The AI never makes the eligibility decision itself.

Design intent: the model NEVER outputs a hire/no-hire decision, and NEVER outputs
an eligibility pass/fail decision. It outputs scores/classifications + evidence;
a human makes every actual call.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"  # pull with: ollama pull llama3.1:8b-instruct-q4_K_M

QUESTION_BANK_PATH = Path(__file__).parent / "question_bank.json"
RESUME_QUESTION_CACHE_DIR = Path(__file__).parent / "cache" / "resume_questions"
RESUME_QUESTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

INTERVIEWER_PERSONA_PROMPT = """You are conducting a screening interview on behalf of a company \
for a role in the defense sector. Your tone is warm, professional, and conversational — like a \
thoughtful human recruiter, not a quiz bot.

Ground rules:
- You are an AI. Never pretend otherwise if asked.
- Never give the candidate feedback on how they're doing during the interview — stay neutral.
- Never answer questions about salary, team specifics, security clearance outcomes, or company \
strategy — say a human recruiter will follow up on those.
- Keep your own turns short. You are here to listen, not to talk.
- Do not improvise new interview questions outside the provided question bank, except for the \
bounded follow-up probes you're explicitly allowed.
- You do not make or imply any hire/no-hire or eligibility decision at any point.
"""


def load_question_bank() -> dict:
    with open(QUESTION_BANK_PATH) as f:
        return json.load(f)


@dataclass
class Turn:
    role: str  # "interviewer" | "candidate"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class QuestionState:
    question_id: str
    prompt: str
    section: str
    qtype: str  # fixed | fixed_resume_aware | generated_from_resume | eligibility
    rubric: Optional[dict] = None
    max_probes: int = 0
    probes_used: int = 0
    answers: list = field(default_factory=list)
    score: Optional[int] = None
    evidence: str = ""
    # eligibility-only fields
    disqualifying_answer: Optional[str] = None
    classified_answer: Optional[str] = None  # "yes" | "no" | "unclear"
    eligibility_flag: bool = False


@dataclass
class Session:
    session_id: str
    bank: dict
    resume_text: str = ""
    started_at: float = field(default_factory=time.time)
    time_budget_minutes: int = 30
    current_q_index: int = 0
    questions: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    finished: bool = False

    def time_remaining_minutes(self) -> float:
        elapsed = (time.time() - self.started_at) / 60.0
        return max(0.0, self.time_budget_minutes - elapsed)


SESSIONS: dict[str, Session] = {}


async def call_llm(system: str, user: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()


def _resume_cache_key(resume_text: str, section_cfg: dict) -> Path:
    h = hashlib.sha256((resume_text + json.dumps(section_cfg, sort_keys=True)).encode()).hexdigest()
    return RESUME_QUESTION_CACHE_DIR / f"{h}.json"


GENERATE_QUESTIONS_SYSTEM_PROMPT = """You generate technical screening interview questions based \
on a candidate's resume. Respond ONLY with strict JSON, no markdown, no preamble:
{
  "questions": [
    {"prompt": "<question text>"},
    {"prompt": "<question text>"}
  ]
}
"""


async def generate_resume_questions(resume_text: str, section_cfg: dict) -> list[dict]:
    """Generate (or load from cache) technical questions derived from the resume."""
    cache_path = _resume_cache_key(resume_text, section_cfg)
    if cache_path.exists():
        return json.loads(cache_path.read_text())["questions"]

    count = section_cfg.get("count", 2)
    instructions = section_cfg["generation_instructions"].format(count=count)
    user_prompt = f"{instructions}\n\nCandidate's resume:\n{resume_text}"

    raw = await call_llm(GENERATE_QUESTIONS_SYSTEM_PROMPT, user_prompt)
    try:
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])[:count]
    except json.JSONDecodeError:
        # Fail safe: generic fallback questions if generation breaks, rather than crashing the session
        questions = [
            {"prompt": "Tell me about a technology on your resume you'd say you know best, and a tricky problem you solved with it."},
            {"prompt": "Pick a project from your resume and walk me through a technical tradeoff you made on it."},
        ][:count]

    cache_path.write_text(json.dumps({"questions": questions}, indent=2))
    return questions


async def build_question_states(bank: dict, resume_text: str) -> list[QuestionState]:
    """Flatten the question bank's sections into a single ordered list of QuestionState."""
    qstates: list[QuestionState] = []

    for section_cfg in bank["sections"]:
        section_name = section_cfg["section"]
        qtype = section_cfg["type"]

        if qtype in ("fixed", "fixed_resume_aware"):
            for q in section_cfg["questions"]:
                qstates.append(QuestionState(
                    question_id=q["id"],
                    prompt=q["prompt"],
                    section=section_name,
                    qtype=qtype,
                    rubric=q.get("rubric"),
                    max_probes=q.get("max_probes", 0),
                ))

        elif qtype == "generated_from_resume":
            if resume_text:
                generated = await generate_resume_questions(resume_text, section_cfg)
            else:
                # No resume provided — fall back to generic technical prompts
                generated = [
                    {"prompt": "Tell me about a technical project you're proud of and a hard decision you made on it."},
                    {"prompt": "Describe a time you had to debug something non-trivial. What was your process?"},
                ][: section_cfg.get("count", 2)]

            for i, gq in enumerate(generated):
                qstates.append(QuestionState(
                    question_id=f"gen_{i+1}",
                    prompt=gq["prompt"],
                    section=section_name,
                    qtype=qtype,
                    rubric=section_cfg.get("rubric"),
                    max_probes=section_cfg.get("max_probes", 1),
                ))

        elif qtype == "eligibility":
            for q in section_cfg["questions"]:
                qstates.append(QuestionState(
                    question_id=q["id"],
                    prompt=q["prompt"],
                    section=section_name,
                    qtype=qtype,
                    max_probes=0,
                    disqualifying_answer=q.get("disqualifying_answer"),
                ))

    return qstates


async def new_session(resume_text: str = "") -> Session:
    bank = load_question_bank()
    sid = str(uuid.uuid4())
    qstates = await build_question_states(bank, resume_text)
    s = Session(
        session_id=sid,
        bank=bank,
        resume_text=resume_text,
        time_budget_minutes=bank.get("total_time_minutes", 30),
        questions=qstates,
    )
    SESSIONS[sid] = s
    return s


# --- Scoring / classification for normal (non-eligibility) questions ---------------

DECIDE_SYSTEM_PROMPT = INTERVIEWER_PERSONA_PROMPT + """
Right now your only job is to decide, for the CURRENT question, whether the candidate's answer \
is specific and complete enough to move on, or whether a brief follow-up probe would get useful \
additional signal.

Respond ONLY with strict JSON, no markdown, no preamble:
{
  "action": "next_question" | "probe",
  "probe_question": "<a short, natural follow-up question, only if action is probe, else empty string>",
  "reasoning": "<one sentence, internal use only>"
}

Rules:
- If the candidate already gave a specific, concrete answer, action must be "next_question".
- Only probe if the answer was vague, generic, or avoided specifics.
- If resume context is provided and the answer seems to inflate or contradict it, a probe asking \
for clarification is appropriate.
- Keep probe_question short and conversational.
"""

SCORE_SYSTEM_PROMPT = """You are scoring a single interview answer against a rubric. \
You are NOT making a hire/no-hire decision — only scoring this one answer.

Respond ONLY with strict JSON, no markdown:
{
  "score": <integer 1-5>,
  "evidence": "<1-2 sentence justification quoting/paraphrasing specifics from the candidate's answer>",
  "flag": "<empty string, or a short note if something seems contradictory, evasive, or like resume inflation>"
}

Score conservatively. A score of 5 requires real specificity. A generic or rehearsed-sounding \
answer should score no higher than 3.
"""

ELIGIBILITY_CLASSIFY_PROMPT = """Classify the candidate's spoken answer to a yes/no eligibility \
question. You are NOT deciding eligibility — only classifying what they said.

Respond ONLY with strict JSON, no markdown:
{
  "answer": "yes" | "no" | "unclear",
  "raw_note": "<short paraphrase of what they actually said, for human review>"
}

If the answer is hedged, partial, or doesn't clearly map to yes/no, use "unclear" — do not guess.
"""


async def decide_next_action(question: QuestionState, latest_answer: str, resume_text: str = "") -> dict:
    if question.probes_used >= question.max_probes:
        return {"action": "next_question", "probe_question": "", "reasoning": "probe budget exhausted"}

    resume_context = f"\n\nCandidate's resume (for context):\n{resume_text}" if resume_text and question.qtype == "fixed_resume_aware" else ""
    user_prompt = (
        f"Original question: {question.prompt}\n\n"
        f"Candidate's answer: {latest_answer}\n\n"
        f"Probes used so far: {question.probes_used}/{question.max_probes}"
        f"{resume_context}"
    )
    raw = await call_llm(DECIDE_SYSTEM_PROMPT, user_prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"action": "next_question", "probe_question": "", "reasoning": "parse_error_fallback"}


async def score_question(question: QuestionState) -> dict:
    combined_answer = "\n---\n".join(question.answers)
    rubric = question.rubric or {"1": "Weak", "3": "Adequate", "5": "Strong"}
    user_prompt = (
        f"Question: {question.prompt}\n\n"
        f"Rubric:\n{json.dumps(rubric, indent=2)}\n\n"
        f"Candidate's full answer (including any follow-ups):\n{combined_answer}"
    )
    raw = await call_llm(SCORE_SYSTEM_PROMPT, user_prompt)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"score": None, "evidence": "Scoring failed to parse — needs manual review.", "flag": "parse_error"}
    question.score = result.get("score")
    question.evidence = result.get("evidence", "")
    return result


async def classify_eligibility_answer(question: QuestionState, answer_text: str) -> dict:
    user_prompt = f"Question: {question.prompt}\n\nCandidate's answer: {answer_text}"
    raw = await call_llm(ELIGIBILITY_CLASSIFY_PROMPT, user_prompt)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"answer": "unclear", "raw_note": answer_text}

    question.classified_answer = result.get("answer", "unclear")
    question.evidence = result.get("raw_note", "")
    if question.disqualifying_answer and question.classified_answer == question.disqualifying_answer:
        question.eligibility_flag = True
    elif question.classified_answer == "unclear":
        question.eligibility_flag = True  # flag for human follow-up, not a disqualification
    return result


def build_final_report(session: Session) -> dict:
    scored_qs = [q for q in session.questions if q.qtype != "eligibility" and q.score is not None]
    scores = [q.score for q in scored_qs]
    avg = round(sum(scores) / len(scores), 2) if scores else None

    technical_flags = [
        {"question_id": q.question_id, "section": q.section, "evidence": q.evidence}
        for q in scored_qs if q.score is not None and q.score <= 2
    ]

    eligibility_results = [
        {
            "question_id": q.question_id,
            "prompt": q.prompt,
            "classified_answer": q.classified_answer,
            "evidence": q.evidence,
            "flagged_for_human_review": q.eligibility_flag,
        }
        for q in session.questions if q.qtype == "eligibility"
    ]

    return {
        "session_id": session.session_id,
        "role": session.bank.get("role"),
        "duration_minutes": round((time.time() - session.started_at) / 60.0, 1),
        "average_technical_score": avg,
        "per_question": [
            {
                "question_id": q.question_id,
                "section": q.section,
                "prompt": q.prompt,
                "score": q.score,
                "evidence": q.evidence,
                "probes_used": q.probes_used,
            }
            for q in scored_qs
        ],
        "low_score_flags": technical_flags,
        "eligibility": {
            "note": "Eligibility answers are CLASSIFIED ONLY, not adjudicated. A human on the "
                    "security/HR team must make the actual eligibility determination. Sensitive "
                    "fields (drug history, financial status) should be access-restricted per PDPA.",
            "results": eligibility_results,
        },
        "note": "This report contains no hire/no-hire or eligibility recommendation. For human review only.",
        "transcript": [{"role": t.role, "text": t.text, "ts": t.ts} for t in session.transcript],
    }
