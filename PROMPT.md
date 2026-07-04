# Agent Prompt: Create a LinguAfrika NLLB Translation Server

Use this prompt to scaffold a new translation model repository. Fill in the
`[PLACEHOLDERS]` before handing it to the agent.

---

## Prompt

You are creating a standalone Python Flask REST API server that serves a
fine-tuned NLLB (No Language Left Behind) translation model for LinguAfrika.
The model is a PEFT/LoRA adapter hosted on Hugging Face and must be downloaded
locally before the server starts.

### Model Details (fill these in before running)

| Variable | Value |
|---|---|
| `HF_MODEL_ID` | `[drakensberg85/English-Hausa_NLLB_FT_model · Hugging Face]` |
| `MODEL_FOLDER` | `[e.g. English-Hausa_NLLB_FT_model]` |
| `SOURCE_LANGUAGE` | `[e.g. English]` |
| `TARGET_LANGUAGE` | `[e.g. Hausa]` |
| `SOURCE_LANG_CODE` | `[e.g. eng_Latn]` |
| `TARGET_LANG_CODE` | `[e.g. hau_Latn]` |
| `PORT` | `[e.g. 5003 — use a different port per model]` |
| `REPO_NAME` | `[e.g. lla-llm-Hausa]` |

> Common NLLB language codes: `eng_Latn` (English), `yor_Latn` (Yoruba),
> `ibo_Latn` (Igbo), `hau_Latn` (Hausa), `swh_Latn` (Swahili),
> `wol_Latn` (Wolof), `zul_Latn` (Zulu).

---

### Task

Create a new project directory named `[REPO_NAME]` with the following files.
Do not add any files beyond what is listed here.

```
[REPO_NAME]/
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

### File 1 — `app.py`

Create a Flask REST API with the following behaviour. Do NOT use
`flask.json.JSONEncoder` — it was removed in Flask 2.3. Do NOT set
`app.json_encoder`. Use `app.config['JSON_AS_ASCII'] = False` for Unicode.

**Model loading** (run once at startup, not per request):
- The tokenizer is loaded from `./models/[MODEL_FOLDER]/` using
  `AutoTokenizer.from_pretrained`.
- The base model `facebook/nllb-200-distilled-600M` is loaded with
  `AutoModelForSeq2SeqLM.from_pretrained`. Use `torch.float16` on CUDA,
  `torch.float32` on CPU.
- The PEFT adapter is applied on top with
  `PeftModel.from_pretrained(base_model, "./models/[MODEL_FOLDER]/")`
- Call `model.eval()` after loading.
- If `torch.cuda.is_available()` is true use `device_map="auto"` for the base
  model load; otherwise omit it and call `model.to("cpu")` afterwards.

**Translation logic** (`translate` method):
- Set `tokenizer.src_lang = source_lang` before tokenising.
- Tokenise with `padding=True, truncation=True, max_length=512`.
- Resolve the forced BOS token with
  `tokenizer.convert_tokens_to_ids(target_lang)`.
- Generate with `num_beams=5, length_penalty=1.0, early_stopping=True,
  max_length=512`.
- Decode with `skip_special_tokens=True`.
- Apply `unicodedata.normalize('NFC', decoded_text)` before returning.

**Endpoints** (all return JSON):

| Method | Path | Description |
|---|---|---|
| GET | `/` | Identity — model name, org, capabilities, endpoint list |
| GET | `/health` | `{"status": "healthy", "model": "...", "device": "..."}` |
| POST | `/translate` | Single text translation |
| POST | `/translate/batch` | List of texts translation |
| GET | `/languages` | Supported language codes |
| GET | `/help` | Full API documentation with curl examples |

`/translate` request body:
```json
{ "text": "Hello world", "source_lang": "[SOURCE_LANG_CODE]", "target_lang": "[TARGET_LANG_CODE]" }
```
`source_lang` and `target_lang` are optional; default to `[SOURCE_LANG_CODE]`
and `[TARGET_LANG_CODE]`.

`/translate/batch` request body:
```json
{ "texts": ["Hello", "Goodbye"], "source_lang": "[SOURCE_LANG_CODE]", "target_lang": "[TARGET_LANG_CODE]" }
```

Return HTTP 400 for missing/invalid input. Return HTTP 500 for translation
errors. Always include `Content-Type: application/json; charset=utf-8` on
translation responses.

Server startup reads `PORT` (default `[PORT]`), `HOST` (default `0.0.0.0`),
and `DEBUG` (default `false`) from environment variables.

---

### File 2 — `Dockerfile`

```dockerfile
FROM python:3.10

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models ./models

EXPOSE [PORT]

ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=[PORT]
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
```

The model files must already be present in `./models/[MODEL_FOLDER]/` on the
host before building the image (they are copied in at build time).

---

### File 3 — `requirements.txt`

Pin these exact versions (they are tested and compatible):

```
Flask==3.1.1
transformers==4.53.2
peft==0.16.0
torch==2.7.1
accelerate==1.9.0
huggingface-hub==0.33.4
safetensors==0.5.3
tokenizers==0.21.2
numpy==2.1.2
requests==2.32.4
Werkzeug==3.1.3
```

---

### File 4 — `.gitignore`

```
.venv/
__pycache__/
*.pyc
models/
*.safetensors
*.bin
.env
```

The `models/` directory is excluded from git because the weights are large.
Users download them separately (see README instructions below).

---

### File 5 — `README.md`

Write a clear, concise README with these sections in order. Use plain markdown
(no HTML tags). Do not add emojis unless they appear in an example response.

1. **Title**: `# LinguAfrika [SOURCE_LANGUAGE]-[TARGET_LANGUAGE] Translation Server`
2. **One-line description**: what the repo does and which model it serves.
3. **Prerequisites**: Python 3.10+, Docker (optional), Git.
4. **Model download** — show the exact `huggingface-cli` command to download
   the model weights into `./models/[MODEL_FOLDER]/`:
   ```bash
   huggingface-cli download [HF_MODEL_ID] --local-dir ./models/[MODEL_FOLDER]/
   ```
5. **Running locally (Python)**:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
   Server starts at `http://localhost:[PORT]`.
6. **Running with Docker**:
   ```bash
   docker build -t lla-[TARGET_LANGUAGE_LOWERCASE] .
   docker run -p [PORT]:[PORT] lla-[TARGET_LANGUAGE_LOWERCASE]
   ```
7. **API reference** — one section per endpoint with method, path, example
   request, and example response. Show at least:
   - `GET /health`
   - `POST /translate` (with curl example)
   - `POST /translate/batch` (with curl example)
   - `GET /languages`
8. **Language codes** — table of the codes supported by this model.
9. **Environment variables** — `PORT`, `HOST`, `DEBUG`.
10. **Error responses** — list the HTTP codes and example error JSON.

---

### Constraints and pitfalls to avoid

- **Do not** import `JSONEncoder` from `flask.json` — it does not exist in
  Flask 2.3+. The fix is to simply not subclass it.
- **Do not** use `app.json_encoder = ...` — it is also removed.
- **Do not** use `device_map="auto"` when loading the PEFT adapter; only use
  it for the base model on CUDA.
- **Do not** call `model.to(device)` after using `device_map="auto"` — it
  conflicts. Only call `.to(device)` on CPU.
- The `default()` method of a JSONEncoder only fires for types Python cannot
  serialise natively. Strings are always serialisable, so a custom encoder
  trying to normalise strings in `default()` is dead code. Do not add it.
- **Do not** create any extra files (test scripts, helper modules, Makefile,
  docker-compose) unless explicitly asked.

---

### Verification checklist (agent should confirm before finishing)

- [ ] `from flask.json import JSONEncoder` does NOT appear anywhere.
- [ ] `app.json_encoder` is NOT set anywhere.
- [ ] `requirements.txt` pins Flask to 3.1.1 or later.
- [ ] The Dockerfile `EXPOSE` and `CMD` use port `[PORT]`.
- [ ] The README `huggingface-cli download` command uses the correct `[HF_MODEL_ID]`.
- [ ] Default `source_lang` in `/translate` is `[SOURCE_LANG_CODE]`.
- [ ] Default `target_lang` in `/translate` is `[TARGET_LANG_CODE]`.
- [ ] `.gitignore` excludes `models/` and `.venv/`.
