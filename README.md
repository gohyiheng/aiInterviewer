# Self-Hosted AI Screening Interviewer

Built for: RTX 3060 Mobile (6GB VRAM), Ryzen 5800H, 32GB RAM, running on Windows.

## What this does
- Runs a 15-30 min structured screening interview against a mix of fixed and
  resume-generated questions
- Optional resume upload (.pdf/.docx/.txt) — used to:
  1. Generate 1-2 personalized technical questions targeting the specific tools/
     projects on that resume (cached on disk by resume hash, so re-running the
     same resume doesn't re-burn LLM calls — see `cache/resume_questions/`)
  2. Let the "experience" question's follow-up probes reference resume content
- Adapts with up to N follow-up probes per question if an answer is vague
- Scores technical/experience answers against a rubric (1-5) with evidence
- Runs a separate **eligibility** section (citizenship, drug history, financial/debt)
  for defense-sector roles — answers are **classified** (yes/no/unclear), never
  adjudicated. A human always makes the actual eligibility call.
- Produces a JSON scorecard + full transcript for a human to review
- Does **not** make a hire/no-hire or eligibility decision itself, anywhere

## Stack
- **LLM**: Ollama (local), `llama3.1:8b-instruct-q4_K_M` — fits in 6GB VRAM
- **STT**: faster-whisper (`small` model) — runs on GPU, transcribes candidate audio
- **TTS**: Browser's native Web Speech API — zero local compute, keeps your GPU free
- **Resume parsing**: pdfplumber (.pdf) / python-docx (.docx) — plain text only
- **Backend**: FastAPI

## Important: eligibility questions need legal review before real use
The eligibility section asks about citizenship, drug history, and debt/bankruptcy.
Before using this on real candidates:
- **Citizenship** is a standard bona fide requirement for defense-sector roles — fine
  to ask directly, but confirm the exact wording/scope with your legal/HR team
  (e.g. citizenship vs. PR vs. existing clearance status may matter differently).
- **Drug history and financial/debt status are sensitive personal data under
  Singapore's PDPA.** You need:
  - Explicit, separate consent language before this section (not just the general
    "this is recorded" disclosure) — the current `intro_note` in `question_bank.json`
    is a starting point, not a compliance-checked consent flow.
  - Restricted access to these specific fields in storage — they're currently
    written into the same `results/<session_id>.json` as everything else; for
    production use, split eligibility results into a separate, access-controlled
    store before this goes near real candidates.
  - A defined retention/deletion policy specifically for this data.
- The AI **classifies** answers (yes/no/unclear) and flags them — it does not
  decide pass/fail. Treat "unclear" as "needs human follow-up," not as a negative
  signal — people sometimes just answer ambiguously on a single take.

## Setup (Windows)

### 0. Install ffmpeg (required — faster-whisper needs it to decode the browser's webm/opus audio)
Easiest route on Windows via winget:
```powershell
winget install ffmpeg
```
Then close and reopen PowerShell so PATH updates, and verify with:
```powershell
ffmpeg -version
```
If that doesn't print a version, faster-whisper will fail to transcribe and you'll
see a 500 error on every answer submission.

### 1. Install Ollama and pull the model
Download and run the installer from https://ollama.com/download/windows (no admin
required, installs to your user profile). Ollama runs as a background service after
install — check it's up with:
```powershell
curl http://localhost:11434
```
Then pull the model (Command Prompt or PowerShell):
```powershell
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 2. Python environment
Use the official Python installer from python.org (3.10-3.12 recommended), and make
sure "Add python.exe to PATH" is checked during install. Then, in PowerShell:
```powershell
cd ai-interviewer
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> If PowerShell blocks the activation script with an execution-policy error, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then retry.

### 3. GPU support for faster-whisper (optional but recommended on your 3060)
faster-whisper's CUDA path needs NVIDIA's CUDA + cuDNN DLLs visible on Windows.
The simplest route on Windows is usually:
```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
This puts the required DLLs inside your venv so Windows can find them without a
separate system-wide CUDA toolkit install. If you still hit a "cannot find cudnn"
or similar DLL error at runtime, switch `device="cuda"` to `device="cpu"` via the
`WHISPER_DEVICE` env var (see below) — slower, but with only a handful of short
questions it's still fine for a 20-30 min screen, and your 5800H handles `small`-
model transcription in a few seconds per answer.

### 4. Run it
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```
To force CPU-based transcription instead of CUDA:
```powershell
$env:WHISPER_DEVICE="cpu"
$env:WHISPER_COMPUTE_TYPE="int8"
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser (Chrome/Edge have the best
Web Speech API support). Allow microphone access when prompted.

## Customizing for a real role
Edit `question_bank.json`:
- `intro_script` / `outro_script` — say this is AI-conducted, per most jurisdictions' disclosure rules
- `questions[]` — swap in your own questions, rubrics, time budgets, max_probes
- `total_time_minutes` — hard cutoff; the session force-finishes when time runs out

## Where results go
Each finished session writes a scorecard to `results/<session_id>.json`:
```json
{
  "session_id": "...",
  "average_score": 3.4,
  "per_question": [ { "question_id": "...", "score": 4, "evidence": "...", "probes_used": 1 } ],
  "low_score_flags": [ ... ],
  "transcript": [ ... ]
}
```
Hook this up to your ATS however you like — e.g. a cron job that watches `results/`
and pushes to Greenhouse/Lever/etc.

## Important things I have NOT built in, that you should not skip
1. **Candidate disclosure** — tell candidates up front (in the invite email, not just
   the intro script) that an AI is conducting this screen. Several jurisdictions
   (e.g. NYC Local Law 144) require this plus a published bias audit if you use
   this to actually filter candidates at scale.
2. **Bias audit** — before using this on real candidates, test it against a diverse
   set of mock transcripts (accents, communication styles, non-native English) and
   check score distributions don't skew on anything other than answer quality.
3. **Fallback for tech issues** — there's no reconnect/resume logic if the browser
   tab closes mid-interview. For a real screen, add session persistence (e.g.
   resume by session_id if transcript isn't finished) before sending this to candidates.
4. **Accessibility** — Web Speech API TTS quality varies; consider Piper TTS
   (also fully local, CPU-only) if you need more natural/consistent voice output,
   or a text-only mode as an explicit candidate option.

## Performance notes for your hardware
- `llama3.1:8b-instruct-q4_K_M` uses roughly 5-5.5GB VRAM — leaves headroom for whisper's `small` model alongside it
- If you see VRAM contention (OOM or slowdowns) when both run together, drop whisper to `device="cpu"` — your 5800H can handle short-clip transcription fine, just with added latency (~2-4s per answer instead of <1s)
- Expect ~1-3s LLM response latency per turn on this setup — acceptable for a conversational screen
