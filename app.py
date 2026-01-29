import os
import re
import hmac
import json
import uuid
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, make_response
from openai import OpenAI
import tempfile
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from rq.job import Job

from jobs import get_queue, get_redis
from storage_pg import fetch_blob

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def _get_openai_api_key() -> str:
    """
    Render/hosting env vars sometimes include quotes or a 'Bearer ' prefix.
    The OpenAI SDK may validate the key format, so we normalize it here.
    """
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if (len(key) >= 2) and ((key[0] == key[-1] == '"') or (key[0] == key[-1] == "'")):
        key = key[1:-1].strip()
    # Remove any stray whitespace/newlines inside the key.
    key = re.sub(r"\s+", "", key)
    return key

def _openai_client() -> OpenAI:
    # Create the client lazily so the app can boot even if key is missing.
    timeout_s = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "240") or "240")
    retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2") or "2")
    return OpenAI(api_key=_get_openai_api_key(), timeout=timeout_s, max_retries=retries)

def _looks_like_openai_key(key: str) -> bool:
    """
    Best-effort validation to replace the opaque:
    'The string did not match the expected pattern.'
    """
    if not key:
        return False
    # Common OpenAI key prefixes:
    # - sk-...
    # - sk-proj-...
    # (Keep permissive; just prevent obvious misconfigurations like 'Bearer ...' or JSON blobs.)
    return bool(re.match(r"^sk-(proj-)?[A-Za-z0-9_\-]{10,}$", key))

def _looks_like_model_name(model: str) -> bool:
    # Model IDs are typically lowercase with digits, dots and hyphens (be permissive).
    if not model:
        return False
    if len(model) > 120:
        return False
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$", model))

def _detect_suspicious_openai_env() -> str | None:
    """
    The OpenAI SDK may read env vars like OPENAI_BASE_URL/OPENAI_API_BASE.
    If these are set to a non-URL value, it can trigger regex/pattern validation errors.
    """
    for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        v = (os.environ.get(k) or "").strip()
        if v and not re.match(r"^https?://", v, flags=re.IGNORECASE):
            return f"{k} is set but does not look like a URL. Value starts with: {v[:24]!r}"
    return None

# === Auth (team-only) ===

APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()
APP_SECRET_KEY = os.environ.get("APP_SECRET_KEY", "").strip() or "dev-insecure-secret"
AUTH_COOKIE_NAME = "sg_auth"
AUTH_MAX_AGE_SECONDS = 60 * 60 * 24 * 14  # 14 days

_serializer = URLSafeTimedSerializer(APP_SECRET_KEY, salt="script-generator-web")

def _auth_enabled() -> bool:
    return bool(APP_PASSWORD)

def _is_authed() -> bool:
    if not _auth_enabled():
        return True
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        return False
    try:
        payload = _serializer.loads(token, max_age=AUTH_MAX_AGE_SECONDS)
        return isinstance(payload, dict) and payload.get("v") == 1
    except (BadSignature, SignatureExpired):
        return False

def _set_auth_cookie(resp):
    if not _auth_enabled():
        return resp
    token = _serializer.dumps({"v": 1, "ts": datetime.utcnow().isoformat()})
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        max_age=AUTH_MAX_AGE_SECONDS,
        httponly=True,
        samesite="Lax",
        secure=bool(os.environ.get("COOKIE_SECURE", "")),  # set to "1" on HTTPS if you want
    )
    return resp

def _clear_auth_cookie(resp):
    resp.delete_cookie(AUTH_COOKIE_NAME)
    return resp

def require_auth(view_func):
    @wraps(view_func)
    def _wrapped(*args, **kwargs):
        if _is_authed():
            return view_func(*args, **kwargs)
        return redirect(url_for("login", next=request.path))
    return _wrapped

def _safe_temp_path(filename: str) -> str:
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise FileNotFoundError("Invalid filename")
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found")
    return filepath

# === Postprocessing Functions (from original script) ===

def replace_em_dashes_with_colons(text: str) -> str:
    text = re.sub(r"\s*—\s*", ". ", text)
    text = re.sub(r"\s*–\s*", ". ", text)
    text = re.sub(r"\s+-\s+", ". ", text)
    text = re.sub(r"\s*--\s*", ". ", text)
    text = re.sub(r"(\w)[—–-](\w)", r"\1. \2", text)
    text = re.sub(r"\.\s*\.", ". ", text)
    return text

def replace_long_parentheticals(text: str, max_words: int = 10) -> str:
    def _process_line(line: str) -> str:
        def _repl(match: re.Match) -> str:
            content = match.group(1).strip()
            if len(content.split()) > max_words:
                return f". {content}"
            return f"({content})"
        return re.sub(r"\(([^)]*)\)", _repl, line)
    lines = text.splitlines()
    return "\n".join(_process_line(ln) for ln in lines)

def merge_comma_linebreaks(text: str) -> str:
    lines = text.splitlines()
    merged = []
    for line in lines:
        if merged:
            prev = merged[-1].rstrip()
            if prev.endswith(","):
                merged[-1] = prev + " " + line.lstrip()
                continue
        merged.append(line)
    return "\n".join(merged)

def sanitize_punctuation(text: str) -> str:
    text = replace_em_dashes_with_colons(text)
    text = replace_long_parentheticals(text, max_words=10)
    text = merge_comma_linebreaks(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r":(?=\w)", ": ", text)
    text = re.sub(r"\.(?=\w)", ". ", text)
    return text

def fix_spanish_common_errors(text: str) -> str:
    rules = [
        (r"Gracias por confiar estos trámites conmigo", "Gracias por confiarme estos trámites"),
        (r"Gracias por confiar estos trámites", "Gracias por confiarme estos trámites"),
        (r"\bdel resto día\b", "del resto del día"),
        (r"\bjunto Volotea\b", "con Volotea"),
        (r"\bcheck\.\s*in\b", "check-in"),
        (r"\bcheck\s*[- ]\s*in\b", "check-in"),
        (r"\babordo\b", "a bordo"),
        (r"\bfuera\s+horarios\b", "fuera de los horarios"),
    ]
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def soften_compliance_language(text: str) -> str:
    lines = text.splitlines()
    softened = []
    for ln in lines:
        if re.search(r"(certificad|nunca\s+compart|no\s+almacenamos|cumple.*protocol|cumple.*norma|plataforma[s]?\s+protegida[s]?|seguridad\s+financiera|datos\s+personales\s+no\s+)", ln, flags=re.IGNORECASE):
            softened.append("El pago se realiza en un entorno protegido.")
        else:
            softened.append(ln)
    return "\n".join(softened)

def prune_excessive_closings(text: str) -> str:
    closing_used = False
    more_used = False
    closing_patterns = [
        r"^Gracias por elegir Volotea",
        r"^Gracias por su llamada",
        r"^Le deseo un buen día",
        r"^Que tenga un buen día",
        r"^Ha sido un placer",
        r"^Gracias por confiar(me| en)",
        r"^Gracias por contactar",
        r"^Estamos para ayudarle",
        r"^Muchas?\s+gracias",
        r"^Gracias por su tiempo",
    ]
    anything_else_patterns = [
        r"¿.*(algo|alguna cosa).*(más).*(ayudarle|ayudarla|ayudarlo)\??",
        r"¿Puedo ayudarle en algo más\??",
        r"¿Necesita algo más\??",
        r"¿Hay algo más.*\??",
    ]
    lines = text.splitlines()
    pruned = []
    for ln in lines:
        if any(re.search(p, ln, flags=re.IGNORECASE) for p in anything_else_patterns):
            if more_used:
                continue
            more_used = True
            pruned.append(ln)
            continue
        if any(re.search(p, ln, flags=re.IGNORECASE) for p in closing_patterns):
            if closing_used:
                continue
            closing_used = True
            pruned.append(ln)
            continue
        pruned.append(ln)
    return "\n".join(pruned)

def ends_with_sentence_punct(line: str) -> bool:
    return bool(re.search(r"[\.!\?:;]\s*$", line))

def merge_until_sentence_punct(text: str) -> str:
    lines = text.splitlines()
    merged_lines = []
    buffer = ""
    for line in lines:
        if not buffer:
            buffer = line.strip()
        else:
            buffer = buffer.rstrip()
            if not ends_with_sentence_punct(buffer):
                buffer = (buffer + " " + line.lstrip()).strip()
            else:
                merged_lines.append(buffer)
                buffer = line.strip()
    if buffer:
        merged_lines.append(buffer)
    return "\n".join(merged_lines)

def ensure_sentence_final_punct(text: str) -> str:
    lines = text.splitlines()
    fixed = []
    for ln in lines:
        if ln.strip() and not ends_with_sentence_punct(ln):
            fixed.append(ln.rstrip() + ".")
        else:
            fixed.append(ln)
    return "\n".join(fixed)

def capitalize_sentence_starts(text: str) -> str:
    def _cap_line(ln: str) -> str:
        s = ln.lstrip()
        prefix = ln[:len(ln) - len(s)]
        if not s:
            return ln
        if s[0] in ("¿", "¡"):
            if len(s) > 1:
                return prefix + s[0] + s[1].upper() + s[2:]
            return ln
        return prefix + s[0].upper() + s[1:]
    return "\n".join(_cap_line(ln) for ln in text.splitlines())

def dedupe_punctuation_artifacts(text: str) -> str:
    text = re.sub(r"\.\s+\.", ". ", text)
    text = re.sub(r"([!?;:])\s*\1+", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

def drop_listing_like_lines(text: str) -> str:
    out = []
    for ln in text.splitlines():
        words = [w for w in re.findall(r"\w+", ln)]
        commas = ln.count(",")
        semis = ln.count(";")
        caps = sum(1 for w in words if len(w) > 1 and w[0].isupper())
        if commas >= 3 or semis >= 2:
            continue
        if len(words) > 26:
            continue
        if caps > max(6, len(words) // 3):
            continue
        out.append(ln)
    return "\n".join(out)

def postprocess(text: str) -> str:
    text = sanitize_punctuation(text)
    text = fix_spanish_common_errors(text)
    text = soften_compliance_language(text)
    text = prune_excessive_closings(text)
    text = merge_until_sentence_punct(text)
    text = ensure_sentence_final_punct(text)
    text = capitalize_sentence_starts(text)
    text = dedupe_punctuation_artifacts(text)
    text = drop_listing_like_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# === Script Generation Function ===

def _word_count(t: str) -> int:
    return len([w for w in t.split() if w.strip()])

def _line_count(t: str) -> int:
    return sum(1 for ln in t.splitlines() if ln.strip())

def _generate_chat(messages, *, model: str, temperature: float, top_p: float, max_tokens: int,
                   presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> str:
    response = _openai_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

_LANG_CODE_TO_NAME: dict[str, str] = {
    "AFR": "Afrikaans",
    "ARA": "Arabic",
    "HYE": "Armenian",
    "ASM": "Assamese",
    "AZE": "Azerbaijani",
    "BEL": "Belarusian",
    "BEN": "Bengali",
    "BOS": "Bosnian",
    "BUL": "Bulgarian",
    "CAT": "Catalan",
    "CEB": "Cebuano",
    "NYA": "Chichewa",
    "HRV": "Croatian",
    "CES": "Czech",
    "DAN": "Danish",
    "NLD": "Dutch",
    "ENG": "English",
    "EST": "Estonian",
    "FIL": "Filipino",
    "FIN": "Finnish",
    "FRA": "French",
    "GLG": "Galician",
    "KAT": "Georgian",
    "DEU": "German",
    "ELL": "Greek",
    "GUJ": "Gujarati",
    "HAU": "Hausa",
    "HEB": "Hebrew",
    "HIN": "Hindi",
    "HUN": "Hungarian",
    "ISL": "Icelandic",
    "IND": "Indonesian",
    "GLE": "Irish",
    "ITA": "Italian",
    "JPN": "Japanese",
    "JAV": "Javanese",
    "KAN": "Kannada",
    "KAZ": "Kazakh",
    "KIR": "Kirghiz",
    "KOR": "Korean",
    "LAV": "Latvian",
    "LIN": "Lingala",
    "LIT": "Lithuanian",
    "LTZ": "Luxembourgish",
    "MKD": "Macedonian",
    "MSA": "Malay",
    "MAL": "Malayalam",
    "CMN": "Mandarin Chinese",
    "MAR": "Marathi",
    "NEP": "Nepali",
    "NOR": "Norwegian",
    "PUS": "Pashto",
    "FAS": "Persian",
    "POL": "Polish",
    "POR": "Portuguese",
    "PAN": "Punjabi",
    "RON": "Romanian",
    "RUS": "Russian",
    "SRP": "Serbian",
    "SND": "Sindhi",
    "SLK": "Slovak",
    "SLV": "Slovenian",
    "SOM": "Somali",
    "SPA": "Spanish",
    "SWA": "Swahili",
    "SWE": "Swedish",
    "TAM": "Tamil",
    "TEL": "Telugu",
    "THA": "Thai",
    "TUR": "Turkish",
    "UKR": "Ukrainian",
    "URD": "Urdu",
    "VIE": "Vietnamese",
    "CYM": "Welsh",
}

def _normalize_language(lang: str) -> str:
    lang = (lang or "").strip()
    if not lang:
        return "English"
    # If frontend sends a code like "ENG", convert to a friendly name.
    if len(lang) == 3 and lang.isalpha():
        return _LANG_CODE_TO_NAME.get(lang.upper(), lang.upper())
    return lang

def _maybe_generate_auto_brief(*, language: str, accent: str, script_type: str, details: str, model: str) -> str:
    sys = {
        "role": "system",
        "content": (
            "You create ultra-brief, brand-safe, practical micro-briefs for voiceover scripts. "
            "Return ONLY the brief, no titles, no quotes, no bullets."
        )
    }
    usr = {
        "role": "user",
        "content": (
            f"Language: {language}\n"
            f"Accent / dialect: {accent or 'not specified'}\n"
            f"Script type: {script_type}\n"
            f"User details:\n{details}\n\n"
            "Write one fresh brief (2–4 sentences) that is specific, recordable, and ready to feed into a script generator."
        )
    }
    return _generate_chat([sys, usr], model=model, temperature=0.7, top_p=0.95, max_tokens=220)

def _build_messages(*, script_type: str, language: str, accent: str, brief: str, details: str) -> list[dict]:
    """
    Build a single-speaker (solo) voiceover script prompt.
    """
    language = (language or "English (en-US)").strip()
    accent = (accent or "").strip()
    script_type = (script_type or "voiceover").strip()
    brief = (brief or "").strip()
    details = (details or "").strip()

    base_system = (
        "You are a senior voiceover/copywriting scriptwriter.\n"
        "Output rules (strict): plain UTF-8 text only; no Markdown; no headings; "
        "no labels like 'Speaker:'; no dialogue format; no stage directions like [pause] or (laughs).\n"
        "Write a single-speaker script meant for one voice actor.\n"
        "Use natural pacing and paragraph breaks.\n"
    )

    sys = {"role": "system", "content": base_system}
    usr = {
        "role": "user",
        "content": (
            f"LANGUAGE: {language}\n"
            f"ACCENT / DIALECT: {accent or 'not specified'}\n"
            f"SCRIPT TYPE: {script_type}\n\n"
            f"BRIEF:\n{brief}\n\n"
            f"DETAILS / REQUIREMENTS:\n{details}\n\n"
            "Write a script that is ready to record.\n"
            "No lists or bullets.\n"
        ),
    }
    return [sys, usr]

def generate_script(*, script_type: str, language: str, accent: str, brief: str, details: str,
                    model: str, target_words: int, target_minutes: int | None = None) -> tuple[str, dict]:
    """
    Returns (script_text, meta_dict).
    """
    # Basic safety bounds (keeps Render requests reasonable)
    target_words = int(target_words or 1200)
    target_words = max(200, min(target_words, 8000))

    # Temperature/top_p (mirrors your scripts: random but bounded)
    temperature = 0.8
    top_p = 0.95

    # Auto-brief if Brief is empty
    used_brief = (brief or "").strip()
    if not used_brief:
        used_brief = _maybe_generate_auto_brief(
            language=language, accent=accent, script_type=script_type, details=details, model=model
        )

    messages = _build_messages(
        script_type=script_type,
        language=language,
        accent=accent,
        brief=used_brief,
        details=details,
    )

    # First pass (keep outputs bounded; avoids slow/OOM on small instances)
    max_tokens = min(3500, max(800, int(target_words * 1.7)))
    raw = _generate_chat(
        messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=0.3,
        frequency_penalty=0.2,
    )

    script_text = raw

    # Light continuation loop to reach approximate target words (kept conservative)
    loops = 0
    max_loops = 1
    while _word_count(script_text) < target_words and loops < max_loops:
        tail = script_text[-1200:]
        cont_user = (
            f"Continue exactly where you left off. Current word count is about {_word_count(script_text)}. "
            f"Keep writing until you reach at least {target_words} words total. "
            "Do not repeat or summarize. Maintain the same tone and format."
        )
        messages = messages + [{"role": "assistant", "content": tail}, {"role": "user", "content": cont_user}]
        part = _generate_chat(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=0.3,
            frequency_penalty=0.2,
        )
        script_text += "\n" + part
        loops += 1

    # Understated estimated minutes (intentionally optimistic)
    # Example: 180 wpm makes the estimate lower than typical VO pacing.
    estimated_minutes = max(1, int(round(_word_count(script_text) / 180)))

    meta = {
        "script_type": script_type,
        "language": language,
        "accent": accent,
        "temperature": temperature,
        "top_p": top_p,
        "target_minutes": int(target_minutes) if target_minutes is not None else None,
        "target_words": target_words,
        "word_count": _word_count(script_text),
        "line_count": _line_count(script_text),
        "brief_used": used_brief,
        "estimated_minutes": estimated_minutes,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return script_text, meta

# === Flask Routes ===

@app.route('/')
@require_auth
def index():
    return render_template('index.html', auth_enabled=_auth_enabled())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not _auth_enabled():
        return redirect(url_for("index"))
    if request.method == 'GET':
        return render_template('login.html', error=None)

    password = (request.form.get("password") or "").strip()
    if not hmac.compare_digest(password, APP_PASSWORD):
        return render_template('login.html', error="Invalid password.")

    nxt = request.args.get("next") or url_for("index")
    resp = make_response(redirect(nxt))
    return _set_auth_cookie(resp)

@app.route('/logout')
def logout():
    resp = make_response(redirect(url_for("login")))
    return _clear_auth_cookie(resp)

@app.route('/generate', methods=['POST'])
@require_auth
def generate():
    try:
        # Async mode: enqueue a background job and return job_id quickly.
        # We still validate env vars early so the user gets a fast, readable error.
        api_key = _get_openai_api_key()
        if not api_key:
            return jsonify({'error': 'Missing OPENAI_API_KEY in environment variables.'}), 500
        if not _looks_like_openai_key(api_key):
            safe_prefix = api_key[:8] + "..." if len(api_key) > 8 else api_key
            return jsonify({'error': f"OPENAI_API_KEY looks invalid. Key starts with: {safe_prefix!r}"}), 500
        suspicious = _detect_suspicious_openai_env()
        if suspicious:
            return jsonify({'error': suspicious}), 500

        data = request.json or {}
        script_type = (data.get("script_type") or "").strip()
        language = _normalize_language(data.get('language') or 'English')
        accent = (data.get('accent') or '').strip()
        brief = (data.get('brief') or '').strip()
        details = (data.get('details') or '').strip()
        target_minutes = int(data.get("target_minutes") or 5)
        v3 = bool(data.get("v3"))

        model = (os.environ.get("OPENAI_MODEL") or "gpt-4.1").strip()
        if not _looks_like_model_name(model):
            return jsonify({'error': f"OPENAI_MODEL looks invalid: {model!r}"}), 500

        if not script_type:
            return jsonify({'error': 'Please enter Script type.'}), 400

        job_id = uuid.uuid4().hex
        payload = {
            "script_type": script_type,
            "language": language,
            "accent": accent,
            "brief": brief,
            "details": details,
            "target_minutes": target_minutes,
            "model": model,
            "v3": v3,
        }

        q = get_queue()
        job = q.enqueue("tasks.run_generate_job", payload, job_id=job_id)
        job.meta.update({"status": "queued", "stage": "queued", "progress": 0, "message": "Queued…"})
        job.save_meta()

        return jsonify({"success": True, "job_id": job_id})
        
    except Exception as e:
        # Log full traceback in Render logs for debugging.
        app.logger.exception("Error in /generate")
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>', methods=['GET'])
@require_auth
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        return jsonify({"error": "Job not found"}), 404

    status = job.get_status()
    meta = job.meta or {}

    resp = {
        "job_id": job_id,
        "status": meta.get("status") or status,
        "stage": meta.get("stage"),
        "progress": meta.get("progress", 0),
        "message": meta.get("message"),
        "current_segment": meta.get("current_segment"),
        "total_segments": meta.get("total_segments"),
        "preview": meta.get("preview"),
        "result": meta.get("result"),
    }

    if status == "failed":
        # Do not return full traceback to the browser.
        resp["error"] = meta.get("error") or "Job failed."
        return jsonify(resp), 500

    return jsonify(resp)

@app.route('/download/<filename>')
@require_auth
def download(filename):
    try:
        filepath = _safe_temp_path(filename)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/download-meta/<filename>')
@require_auth
def download_meta(filename):
    try:
        filepath = _safe_temp_path(filename)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/download-job/<path:job_file>')
@require_auth
def download_job(job_file: str):
    """
    Download artifacts stored in Postgres by job id.
    Supported:
      /download-job/<job_id>.txt
      /download-job/<job_id>.meta.json
      /download-job/<job_id>.outline.json
    """
    try:
        if job_file.endswith(".txt"):
            job_id = job_file[:-4]
            content, filename = fetch_blob(job_id, "txt")
            resp = make_response(content)
            resp.headers["Content-Type"] = "text/plain; charset=utf-8"
            resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            return resp

        if job_file.endswith(".meta.json"):
            job_id = job_file[: -len(".meta.json")]
            content, filename = fetch_blob(job_id, "meta")
            resp = make_response(content)
            resp.headers["Content-Type"] = "application/json; charset=utf-8"
            resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            return resp

        if job_file.endswith(".outline.json"):
            job_id = job_file[: -len(".outline.json")]
            content, filename = fetch_blob(job_id, "outline")
            resp = make_response(content)
            resp.headers["Content-Type"] = "application/json; charset=utf-8"
            resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            return resp

        return jsonify({"error": "Unsupported file type"}), 400
    except FileNotFoundError:
        return jsonify({"error": "Job not found"}), 404
    except Exception as e:
        app.logger.exception("Error in /download-job")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


