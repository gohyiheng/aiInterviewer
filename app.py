"""
FastAPI server for the AI screening interviewer.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Requires:
    - Ollama running locally with the model pulled (see interview_engine.py)
    - faster-whisper installed (pip install faster-whisper)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# On Windows, pip-installed nvidia-*-cu12 packages drop their DLLs (cublas64_12.dll,
# cudnn64_9.dll, etc.) inside site-packages, but Windows won't find them unless that
# folder is explicitly registered as a DLL search path. Register them BEFORE importing
# faster_whisper, since it loads the CUDA libraries at import/model-init time.
if sys.platform == "win32":
    import importlib.util
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                bin_dir = os.path.join(loc, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)

from faster_whisper import WhisperModel

import interview_engine as engine
import resume_parser

app = FastAPI(title="AI Screening Interviewer")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Override with env vars if CUDA DLLs still aren't resolving on Windows, e.g.:
#   set WHISPER_DEVICE=cpu
#   set WHISPER_COMPUTE_TYPE=int8
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
whisper_model = WhisperModel("small", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _next_question_payload(session, force_finish_outro: str | None = None):
    """Build the JSON the frontend needs for whatever the current question is,
    skipping the section intro_note exactly once per eligibility section."""
    q = session.questions[session.current_q_index]
    payload_extra = {}
    if q.qtype == "eligibility" and not getattr(session, "_eligibility_intro_said", False):
        bank = session.bank
        for section_cfg in bank["sections"]:
            if section_cfg["section"] == q.section and "intro_note" in section_cfg:
                payload_extra["section_intro"] = section_cfg["intro_note"]
                session.transcript.append(engine.Turn(role="interviewer", text=section_cfg["intro_note"]))
        session._eligibility_intro_said = True
    return q, payload_extra


@app.post("/session/start")
async def start_session(resume: UploadFile = File(None)):
    resume_text = ""
    if resume is not None:
        suffix = Path(resume.filename).suffix.lower()
        tmp_path = UPLOADS_DIR / f"resume_{Path(resume.filename).stem}{suffix}"
        with open(tmp_path, "wb") as f:
            f.write(await resume.read())
        try:
            resume_text = resume_parser.extract_text(str(tmp_path))
        except ValueError as e:
            raise HTTPException(400, str(e))

    session = await engine.new_session(resume_text=resume_text)
    session._eligibility_intro_said = False
    bank = session.bank
    intro = bank["intro_script"]
    session.transcript.append(engine.Turn(role="interviewer", text=intro))

    first_q, extra = _next_question_payload(session)
    session.transcript.append(engine.Turn(role="interviewer", text=first_q.prompt))

    return {
        "session_id": session.session_id,
        "intro": intro,
        "question": first_q.prompt,
        "question_index": 0,
        "total_questions": len(session.questions),
        "time_remaining_minutes": session.time_remaining_minutes(),
        "resume_uploaded": bool(resume_text),
        **extra,
    }


def _transcribe(audio_path: str) -> str:
    segments, _info = whisper_model.transcribe(audio_path, language="en", vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


@app.post("/session/{session_id}/answer")
async def submit_answer(
    session_id: str,
    audio: UploadFile = File(None),
    text_answer: str = Form(None),
):
    session = engine.SESSIONS.get(session_id)
    if not session or session.finished:
        raise HTTPException(404, "Session not found or already finished")

    if session.current_q_index >= len(session.questions):
        raise HTTPException(400, "No more questions in this session")

    try:
        if audio is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
            try:
                tmp.write(await audio.read())
                tmp.close()  # must close before another process (whisper) reads it on Windows
                answer_text = _transcribe(tmp.name)
            finally:
                try:
                    Path(tmp.name).unlink()
                except OSError:
                    pass
        elif text_answer:
            answer_text = text_answer
        else:
            raise HTTPException(400, "Provide either audio or text_answer")

        if not answer_text:
            raise HTTPException(
                422,
                "Transcription came back empty — check the recording captured audio, "
                "and that ffmpeg is installed and on PATH (faster-whisper needs it to decode webm).",
            )

        question = session.questions[session.current_q_index]
        question.answers.append(answer_text)
        session.transcript.append(engine.Turn(role="candidate", text=answer_text))

        if session.time_remaining_minutes() <= 0:
            return await _advance_or_finish(session, force_finish=True)

        if question.qtype == "eligibility":
            await engine.classify_eligibility_answer(question, answer_text)
            return await _advance_or_finish(session)

        decision = await engine.decide_next_action(question, answer_text, resume_text=session.resume_text)

        if decision["action"] == "probe" and decision.get("probe_question"):
            question.probes_used += 1
            session.transcript.append(engine.Turn(role="interviewer", text=decision["probe_question"]))
            return {
                "session_id": session.session_id,
                "type": "probe",
                "question": decision["probe_question"],
                "question_index": session.current_q_index,
                "time_remaining_minutes": session.time_remaining_minutes(),
            }

        await engine.score_question(question)
        return await _advance_or_finish(session)

    except HTTPException:
        raise
    except httpx.ConnectError:
        raise HTTPException(
            503,
            "Could not reach Ollama at http://localhost:11434 — make sure Ollama is running "
            "(check the system tray icon, or run `ollama serve`).",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()  # full traceback in the uvicorn console for debugging
        raise HTTPException(500, f"{type(e).__name__}: {e}")


async def _advance_or_finish(session: engine.Session, force_finish: bool = False):
    session.current_q_index += 1
    if force_finish or session.current_q_index >= len(session.questions):
        session.finished = True
        outro = session.bank["outro_script"]
        session.transcript.append(engine.Turn(role="interviewer", text=outro))
        report = engine.build_final_report(session)
        out_path = RESULTS_DIR / f"{session.session_id}.json"
        out_path.write_text(json.dumps(report, indent=2))
        return {
            "session_id": session.session_id,
            "type": "finished",
            "outro": outro,
            "report": report,
        }

    next_q, extra = _next_question_payload(session)
    session.transcript.append(engine.Turn(role="interviewer", text=next_q.prompt))
    return {
        "session_id": session.session_id,
        "type": "next_question",
        "question": next_q.prompt,
        "question_index": session.current_q_index,
        "total_questions": len(session.questions),
        "time_remaining_minutes": session.time_remaining_minutes(),
        **extra,
    }


@app.get("/session/{session_id}/report")
async def get_report(session_id: str):
    out_path = RESULTS_DIR / f"{session_id}.json"
    if not out_path.exists():
        raise HTTPException(404, "Report not found (session may not be finished yet)")
    return JSONResponse(json.loads(out_path.read_text()))
