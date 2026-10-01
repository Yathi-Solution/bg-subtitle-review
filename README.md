# Telugu → English Subtitle Review Pipeline

Turns a Telugu **audio recording** or **`.sbv`/`.srt` subtitle file** into reviewed
Telugu subtitles, an AI English draft, and — after human review — final English
subtitles plus an Excel review workbook. Comes with a CLI and a responsive web UI
(works on desktop, Android, and iOS browsers).

> The AI output is a **first-pass draft for human review** — a reviewer must
> check every cue before publishing.

## Workflow (human in the loop at every stage)

```
audio (.mp3 …) ── speech-to-text ──┐
                                   ├─► 1. Telugu subtitles: human edits cues, listens to
.sbv / .srt ───────────────────────┘      low-confidence ones, downloads SBV/SRT
                                              │
                                              ▼  translate (Azure OpenAI)
                                   2. English review: human writes corrections,
                                      optional AI review flags likely errors
                                              │
                                              ▼
                                   English .srt/.sbv (corrections win) + workbook
```

Why this shape — each step follows published practice:

| Step | Practice | Source |
|---|---|---|
| Fix the Telugu transcript before translating | Recognition errors carry straight into the translation, so the source is corrected first | — |
| Cues ≤ 7 s, ≥ 5/6 s; ≤ 42 chars/line, 2 lines; ≤ 20 chars/s | Subtitle timing and reading-speed limits | [Netflix Timed Text Style Guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements), [English (USA)](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide) |
| A human post-edits every cue; the AI never overwrites | Full human post-editing of machine translation | [ISO 18587:2017](https://www.iso.org/standard/62970.html) |
| AI review uses accuracy / terminology / linguistic / style categories with minor / major / critical severity | MQM error typology | [themqm.org](https://themqm.org/error-types-2/typology/) |
| An LLM flags error spans instead of rewriting everything | GEMBA-MQM (Microsoft, WMT 2023) | [ACL Anthology](https://aclanthology.org/2023.wmt-1.64/) |

### Speech-to-text: Azure AI Speech (Central India)

Audio is transcribed by **Azure AI Speech fast transcription** with the Telugu
(`te-IN`) model. It returns every word with its timestamp; the app groups words
into subtitle-sized cues at the speaker's pauses and carries the recognizer's
confidence so the editor can sort and highlight the weakest cues for review.

- **Region:** Azure Speech is *not* available in South India; **Central India**
  supports fast transcription, and audio is processed only in the resource's region
  ([regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions)).
- **Tier:** Standard (S0). The Free (F0) tier has no fast transcription
  ([quotas](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits)).
- **Cost:** ₹34.40 ($0.36) per audio hour, billed per second — see [Cost per file](#cost-per-file).
- **Limits:** under 5 hours and 500 MB per file (uploads are compressed to mono MP3 first).
- **Optional model:** `AZURE_SPEECH_MODEL=MAI-Transcribe-2` switches to Microsoft's
  newer model, which lists Telugu, on the same resource — it is in *public preview*
  (no SLA) ([MAI-Transcribe](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)).
  LLM speech's other modes do not support Telugu.

## What you get per job

| Download | Purpose |
|---|---|
| Telugu `.sbv` / `.srt` | The (human-corrected) Telugu subtitles, in the formats ticked on upload |
| English `.srt` / `.sbv` | Final English: the human correction where given, else the AI draft. Sanskrit terms marked `*like this*` become `<i>italics</i>` in SRT (SBV has no styling) |
| `*_master_review.xlsx` | Cue #, timecode, Telugu, AI English, Human Review Correction; an **AI Review** sheet when that pass was run |
| `*_discourse_brief.md` | Context the model builds before translating |
| `*_translation_raw.json` | Machine-readable translations |

Jobs and their files are kept **24 hours** after upload (`RETENTION_HOURS`), are
listed under *Recent jobs*, survive server restarts, and are then deleted
automatically. A job can also be deleted immediately from its page.

## Cost per file

Measured on 26 Sep 2026 with a real 26.6-minute Telugu discourse (367 cues).
Prices are Azure's official list prices — the same data as the
[Speech pricing](https://azure.microsoft.com/en-us/pricing/details/speech/) and
[Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/)
pages, read from the [Azure Retail Prices API](https://prices.azure.com/api/retail/prices)
because those pages show prices only after sign-in / region selection.

| Step | Azure service (region) | List price | Usage for this file | Cost |
|---|---|---|---|---|
| Speech-to-text | AI Speech fast transcription (Central India) | ₹34.40 ($0.36) per audio hour, billed per second | 26.6 min of audio | **₹15.25** ($0.16) |
| Translation | OpenAI gpt-5, Global Standard (South India) | per 1M tokens: input ₹119.43 ($1.25), output ₹955.46 ($10.00) | 56K input + 71K output tokens, 11 calls, ~4 min | **₹74.72** ($0.78) |
| AI review *(optional)* | same gpt-5 deployment | same | 119K input + 135K output tokens, 37 calls, ~6 min | **₹142.72** ($1.49) |
| **Total** | | | | **≈ ₹90 ($0.94)** without AI review · **≈ ₹233 ($2.44)** with it |

- **Scales with audio length:** roughly ₹200 per audio hour without AI review, ₹525 with it.
- **Reasoning tokens drive the cost:** gpt-5 "thinks" before answering, and that
  hidden reasoning (83% of translation output, 93% of review output here) is billed as
  output. Token counts vary a little between runs.
- **Re-running costs again:** each re-translation or AI review is billed in full.
  Editing, saving and downloading are free.
- **No fixed fees:** both Azure resources are pay-per-use, and Render's free plan
  costs nothing. The Speech Free (F0) tier's 5 hours/month don't cover fast transcription.
- The price list also shows a *Fast Transcription Promo* meter (₹9.55 / $0.10 per
  hour, since 1 Sep 2026) without stating who qualifies; the table uses the regular
  rate. Your Azure invoice shows which meter was billed.
- Prices change — check the official pages above before budgeting.

## Project layout

```
pipeline.py    Parsing, SBV/SRT writing, translation, AI review, workbook (Azure OpenAI)
transcribe.py  Audio -> timed Telugu cues (Azure AI Speech fast transcription)
cli.py         Command-line interface
app.py         FastAPI backend: jobs, editor API, downloads, 24 h cleanup
static/        Responsive single-page web UI (upload, cue editor, review editor)
render.yaml    One-click Render deployment blueprint
.env.example   Template for environment variables (copy to .env)
```

## 1. Setup (local)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env with your real values
```

Fill in `.env`:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview   # an API version, not a model date
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_SPEECH_KEY=...                          # Speech resource (S0) for audio uploads
AZURE_SPEECH_REGION=centralindia
APP_PASSWORD=choose-a-long-random-password
```

`.env` is git-ignored and must never be committed. Audio decoding uses `ffmpeg`
from your PATH, or the copy bundled with the `imageio-ffmpeg` package.

## 2. Run the web app

```bash
python app.py
# open http://localhost:8000
```

1. Enter the `APP_PASSWORD`.
2. Upload an audio file (tick SBV and/or SRT; optionally list names/terms that
   occur in the recording) or a Telugu `.sbv`/`.srt`.
3. **Telugu subtitles** tab: play cues, fix recognition errors (start with
   *Least confident first*), adjust timings, insert/delete cues, save, download.
4. **Save & translate** → **English review** tab: type corrections where needed.
   Reading-speed and line-length warnings are shown per cue.
5. Optional **Run AI review**: flags likely errors with a suggested fix; choose
   *Use suggestion* only where you agree.
6. Download the English `.srt`/`.sbv` and the workbook.

## 3. Or use the CLI

```bash
python cli.py transcribe episode.mp3 --formats sbv,srt --out-dir out_v4_1_lean
python cli.py translate "CH 05 _EP 11.sbv" --out-dir out_v4_1_lean
python cli.py validate-workbook "CH 05 _EP 11.sbv" out_v4_1_lean/CH_05__EP_11_master_review.xlsx
```

Options: `--model` (Azure deployment override), `--chunk-size` (default 40),
`--overlap` (default 3), `--skill` (skill markdown file), `--hint` (names/terms for
speech-to-text).

## 4. Deploy to Render

1. Push this repo to GitHub.
2. In [Render](https://render.com): **New + → Blueprint**, pick this repo.
   Render reads `render.yaml`.
3. When prompted, set the secret env vars: `AZURE_OPENAI_ENDPOINT`,
   `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_SPEECH_KEY`, `APP_PASSWORD`.
4. Deploy. Your app is live at `https://<name>.onrender.com` over HTTPS.

> **Free plan caveat:** Render's free disk is wiped on every restart, deploy and
> idle spin-down (about 15 minutes without traffic), so a job can disappear before
> its 24 hours — download files when a job finishes. Full 24-hour retention needs a
> paid plan with a persistent disk (set `JOBS_DIR` to the disk's mount path). A step
> that is running when the server restarts is marked as interrupted and can be re-run.

## 5. Deploy to a Windows VM (current production setup)

The app also runs on an Azure Windows Server VM as two always-on Windows services:
`SubtitleReview` (uvicorn) and `Caddy` (HTTPS reverse proxy). The VM's disk is
persistent, so the full 24-hour job retention works here — unlike Render's free plan.

| | |
|---|---|
| Host | Azure Windows Server VM, South India (shared with a PostgreSQL instance) |
| App directory | `E:\apps\bg-subtitle-review` — a persistent data disk |
| Virtualenv | `E:\apps\bg-subtitle-review\myenv` |
| App service | `SubtitleReview` — uvicorn on `127.0.0.1:8000`, not reachable from outside |
| Proxy service | `Caddy` — terminates TLS on 443, forwards to 8000, renews its certificate automatically |
| Public URL | `https://bg-subtitle.southindia.cloudapp.azure.com` |
| Secrets | `E:\apps\bg-subtitle-review\.env` — a plain VM has no App Settings |
| Logs | `E:\apps\bg-subtitle-review\logs\`, `C:\tools\caddy-*.log` (rotated at 10 MB) |

> **Never install to `D:`.** On Azure Windows VMs that letter is the *temporary*
> disk: it is wiped whenever the VM is deallocated, resized, or moved to another
> host. Check with `vol D:` — a label of "Temporary Storage" or the presence of
> `D:\DATALOSS_WARNING_README.txt` confirms it.

### Operations cheat sheet

Run these from an **Administrator** Command Prompt.

```cmd
:: status and health
sc query SubtitleReview
sc query Caddy
curl http://127.0.0.1:8000/healthz

:: start / stop / restart
net stop SubtitleReview
net start SubtitleReview
C:\tools\nssm.exe restart SubtitleReview

:: read the logs
type E:\apps\bg-subtitle-review\logs\err.log
type C:\tools\caddy-err.log

:: deploy an update
cd /d E:\apps\bg-subtitle-review
net stop SubtitleReview
git pull
myenv\Scripts\pip.exe install -r requirements.txt
net start SubtitleReview

:: change a secret (a restart is required to load it)
notepad E:\apps\bg-subtitle-review\.env
C:\tools\nssm.exe restart SubtitleReview

:: inspect or remove a service
C:\tools\nssm.exe edit SubtitleReview
C:\tools\nssm.exe remove SubtitleReview confirm
```

`/healthz` returns `{"ok":true,"configured":true}` once the service is up and has
found its `.env`. `"configured":false` means `APP_PASSWORD` is unset — almost always
a working-directory problem (see Gotchas).

No cleanup job is needed: the app deletes jobs older than `RETENTION_HOURS` itself.

### Rebuilding the VM from scratch

1. **Python 3.12** — install with `InstallAllUsers=1 PrependPath=1`, then reopen the shell.
2. **Code and dependencies** — `git clone` into `E:\apps\bg-subtitle-review`, then
   `python -m venv myenv` and `myenv\Scripts\pip.exe install -r requirements.txt`.
   `ffmpeg` comes bundled via `imageio-ffmpeg`; nothing extra to install.
3. **Secrets** — create `.env` (see [Setup](#1-setup-local)).
4. **Test manually** — `myenv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000`,
   then `curl http://127.0.0.1:8000/healthz`. Stop it before installing the service.
5. **Install NSSM** to `C:\tools`, then register the service. `AppDirectory` is
   mandatory:

   ```cmd
   C:\tools\nssm.exe install SubtitleReview "E:\apps\bg-subtitle-review\myenv\Scripts\python.exe" "-m uvicorn app:app --host 127.0.0.1 --port 8000"
   C:\tools\nssm.exe set SubtitleReview AppDirectory "E:\apps\bg-subtitle-review"
   C:\tools\nssm.exe set SubtitleReview AppStdout "E:\apps\bg-subtitle-review\logs\out.log"
   C:\tools\nssm.exe set SubtitleReview AppStderr "E:\apps\bg-subtitle-review\logs\err.log"
   C:\tools\nssm.exe set SubtitleReview AppRotateFiles 1
   C:\tools\nssm.exe set SubtitleReview AppRotateBytes 10485760
   C:\tools\nssm.exe set SubtitleReview Start SERVICE_AUTO_START
   C:\tools\nssm.exe start SubtitleReview
   ```

6. **Networking** — give the VM a DNS label (portal → VM → Overview → DNS name),
   open TCP 80 and 443 in *both* the Azure NSG and the Windows firewall
   (`netsh advfirewall firewall add rule name="HTTPS" dir=in action=allow protocol=TCP localport=443`).
7. **Caddy** — download `caddy.exe` to `C:\tools`, write `C:\tools\Caddyfile`:

   ```
   bg-subtitle.southindia.cloudapp.azure.com {
       reverse_proxy 127.0.0.1:8000
   }
   ```

   then register it the same way (`AppDirectory` `C:\tools`, auto-start, log files).

### Gotchas

- **`AppDirectory` must be the app folder.** `app.py` and `pipeline.py` both call
  `load_dotenv()`, which searches upward from the *working directory*. Without it the
  service starts in `System32`, never finds `.env`, and every job fails on missing
  credentials while `/healthz` reports `"configured":false`.
- **The Caddyfile hostname must match the VM's DNS name exactly.** A mismatch means
  Caddy holds no certificate for the name the browser asks for, and the handshake
  fails with `ERR_SSL_PROTOCOL_ERROR` — which looks like a broken app but is not.
- **Port 80 must be open in the NSG,** not just the Windows firewall. Let's Encrypt
  validates over HTTP; without it Caddy can never issue or renew a certificate.
- **Keep uvicorn on `127.0.0.1`.** The password is sent in a plain header, so all
  traffic must pass through Caddy's TLS.
- **Run one instance only,** and leave PostgreSQL's port 5432 closed to the internet.
- **A full disk takes PostgreSQL down too,** since both share this VM. Watch free
  space; audio uploads can reach `MAX_AUDIO_BYTES` (300 MB) each.

## Security notes

- Access is gated by a single shared `APP_PASSWORD`. Use a long random value and
  always serve over HTTPS (Render does this automatically). Everyone with the
  password sees the same job list.
- The audio player uses a per-job signed token (derived from `APP_PASSWORD`),
  because browsers cannot send the password header for `<audio>` elements.
- API keys live only in server environment variables, never in the repo.
- If a key is ever committed or shared, **rotate it immediately**.
