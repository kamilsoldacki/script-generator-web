import os
import re
import hmac
import json
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, make_response
from openai import OpenAI
import tempfile
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

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
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def _maybe_generate_auto_brief(*, language: str, accent: str, script_kind: str, details: str, model: str) -> str:
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
            f"Script kind: {script_kind}\n"
            f"User details:\n{details}\n\n"
            "Write one fresh brief (2–4 sentences) that is specific, recordable, and ready to feed into a script generator."
        )
    }
    return _generate_chat([sys, usr], model=model, temperature=0.7, top_p=0.95, max_tokens=220)

def _build_messages(*, preset: str, language: str, accent: str, brief: str, details: str,
                    scenes_count: int | None = None, min_turns: int | None = None, max_turns: int | None = None,
                    dialect: str | None = None) -> list[dict]:
    """
    Presets:
    - generic_agent_lines
    - generic_monologue
    - generic_dialogue_scenes
    - volotea_vera_es (legacy)
    """
    language = (language or "English (en)").strip()
    accent = (accent or "").strip()
    brief = (brief or "").strip()
    details = (details or "").strip()

    if preset == "volotea_vera_es":
        # Legacy Volotea/Vera preset (kept from original)
        client_brief = brief or details
        return [
            {
                "role": "system",
                "content": (
                    "Eres Vera, asistente virtual de Volotea. Atiendes soporte telefónico en directo. "
                    "Hablas únicamente en español europeo con acento neutro de España. Usa 'usted' con respeto y cercanía. "
                    "Tu estilo es sereno, preciso y bajo control (matiz de piloto de avión), cálido y tranquilizador, "
                    "especialmente ante retrasos, cambios o incidencias. Conoces procesos y operación sin revelar detalles internos."
                )
            },
            {
                "role": "system",
                "content": (
                    "Reglas de salida: produce solo las intervenciones de la agente (tus palabras). "
                    "No escribas líneas del cliente. Sin acotaciones, efectos, etiquetas, encabezados, listas, numeración ni markdown. "
                    "Texto plano UTF-8. Cada intervención en su propia línea. No generes párrafos. "
                    "Frases cortas. Una idea por frase. Máximo dos frases por intervención. "
                    "Longitud objetivo por intervención: entre seis y dieciséis palabras. "
                    "No uses raya larga (—) ni guiones como pausa; usa '.' o ':'. "
                    "Paréntesis solo para incisos muy breves."
                )
            },
            {
                "role": "user",
                "content": (
                    f"TAREA:\n"
                    f"Usando este perfil de voz y contexto:\n{client_brief}\n\n"
                    f"Genera un guion conversacional extenso en {language}, compuesto únicamente por líneas que diría la agente por teléfono.\n"
                    "Solo texto plano; una línea = una intervención.\n"
                    "Longitud objetivo: entre mil doscientas y dos mil palabras.\n"
                )
            }
        ]

    base_system = (
        "You are a senior voiceover/copywriting scriptwriter.\n"
        "Output rules (strict): plain UTF-8 text only; no Markdown; no headings unless explicitly requested; "
        "no stage directions like [pause] or (laughs) unless requested.\n"
        "Write in the requested language and keep the style natural for native speakers.\n"
    )

    if preset == "generic_agent_lines":
        sys = {
            "role": "system",
            "content": base_system + (
                "You are writing ONLY the AGENT's lines for call center / customer support training.\n"
                "Formatting: each agent line as its own paragraph separated by a blank line.\n"
                "Each line should start by briefly acknowledging what the customer likely said, then give a next step.\n"
                "Vary length: mix short confirmations with longer helpful explanations.\n"
            )
        }
        usr = {
            "role": "user",
            "content": (
                f"LANGUAGE: {language}\n"
                f"ACCENT / DIALECT: {accent or 'not specified'}\n"
                "SCRIPT KIND: Call center agent-only lines\n\n"
                f"BRIEF:\n{brief}\n\n"
                f"DETAILS / REQUIREMENTS:\n{details}\n\n"
                "Hard rules:\n"
                "- Do NOT include the customer's lines.\n"
                "- No lists or bullets.\n"
                "- Avoid real personal data; if verification is needed, use safe placeholders (e.g., last four digits).\n"
            )
        }
        return [sys, usr]

    if preset == "generic_dialogue_scenes":
        sc = scenes_count or 10
        mi = min_turns or 6
        ma = max_turns or 10
        dialect_note = ""
        if dialect == "najdi":
            dialect_note = "Dialect: Najdi Arabic (Riyadh area). Avoid Hijazi vocabulary."
        elif dialect == "hijazi":
            dialect_note = "Dialect: Hijazi Arabic (Jeddah/Mecca area). Avoid Najdi vocabulary."

        sys = {
            "role": "system",
            "content": base_system + (
                "You write short, realistic dialogue scenes between a caller and an agent.\n"
                "No extra commentary.\n"
            )
        }
        usr = {
            "role": "user",
            "content": (
                f"LANGUAGE: {language}\n"
                f"ACCENT / DIALECT: {accent or 'not specified'}\n"
                f"{dialect_note}\n"
                "SCRIPT KIND: Dialogue scenes (caller + agent)\n\n"
                f"BRIEF:\n{brief}\n\n"
                f"DETAILS / REQUIREMENTS:\n{details}\n\n"
                "Format (strict):\n"
                f"- Produce exactly {sc} scenes.\n"
                "- Each scene starts with a header line:\n"
                "### Scene {number} — {domain} — {intent1}/{intent2}\n"
                "- Then only dialogue lines:\n"
                "Caller: ...\n"
                "Agent: ...\n"
                f"- Turns per scene: between {mi} and {ma} turns (each turn = Caller line + Agent line).\n"
                "- Plain text only.\n"
            )
        }
        return [sys, usr]

    # generic_monologue (default)
    sys = {
        "role": "system",
        "content": base_system + (
            "You write a one-speaker monologue meant for a voice actor.\n"
            "No dialogue labels. No scene directions.\n"
            "Use natural pacing and paragraph breaks.\n"
        )
    }
    usr = {
        "role": "user",
        "content": (
            f"LANGUAGE: {language}\n"
            f"ACCENT / DIALECT: {accent or 'not specified'}\n"
            "SCRIPT KIND: One-speaker monologue\n\n"
            f"BRIEF:\n{brief}\n\n"
            f"DETAILS / REQUIREMENTS:\n{details}\n\n"
            "Output: continuous text with paragraph breaks; ready to record.\n"
            "No lists or bullets.\n"
        )
    }
    return [sys, usr]

def generate_script(*, preset: str, language: str, accent: str, brief: str, details: str,
                    model: str, target_words: int, auto_brief: bool,
                    scenes_count: int | None = None, min_turns: int | None = None, max_turns: int | None = None,
                    dialect: str | None = None) -> tuple[str, dict]:
    """
    Returns (script_text, meta_dict).
    """
    # Basic safety bounds (keeps Render requests reasonable)
    target_words = int(target_words or 1200)
    target_words = max(200, min(target_words, 8000))

    # Temperature/top_p (mirrors your scripts: random but bounded)
    temperature = 0.8
    top_p = 0.95

    # Auto-brief if requested (or if brief empty and auto enabled)
    used_brief = (brief or "").strip()
    if auto_brief and not used_brief:
        used_brief = _maybe_generate_auto_brief(
            language=language, accent=accent, script_kind=preset, details=details, model=model
        )

    messages = _build_messages(
        preset=preset,
        language=language,
        accent=accent,
        brief=used_brief,
        details=details,
        scenes_count=scenes_count,
        min_turns=min_turns,
        max_turns=max_turns,
        dialect=dialect,
    )

    # First pass
    max_tokens = 4500 if target_words <= 2000 else 9000
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

    # Optional postprocess for legacy Volotea preset
    if preset == "volotea_vera_es":
        script_text = postprocess(script_text)

    # Light continuation loop to reach approximate target words (kept conservative)
    loops = 0
    max_loops = 3
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

    # Final polish for Volotea preset: second pass QC (kept from original idea, but shorter)
    if preset == "volotea_vera_es":
        review_messages = [
            {
                "role": "system",
                "content": (
                    "Eres un revisor de calidad para un guion VO/TTS en español de España. "
                    "Devuelve únicamente líneas válidas de la agente; texto plano; una línea = una oración. "
                    "Elimina líneas fuera de dominio o demasiado largas."
                ),
            },
            {
                "role": "user",
                "content": "Limpia este guion manteniendo el sentido y el orden:\n\n" + script_text,
            },
        ]
        reviewed = _generate_chat(
            review_messages,
            model=model,
            temperature=0.1,
            top_p=0.8,
            max_tokens=3000,
            presence_penalty=0.0,
            frequency_penalty=0.2,
        )
        script_text = postprocess(reviewed)

    meta = {
        "preset": preset,
        "language": language,
        "accent": accent,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "target_words": target_words,
        "word_count": _word_count(script_text),
        "line_count": _line_count(script_text),
        "auto_brief": bool(auto_brief),
        "brief_used": used_brief,
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
        return render_template('login.html', error="Nieprawidłowe hasło.")

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
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({'error': 'Brak OPENAI_API_KEY w zmiennych środowiskowych'}), 500

        data = request.json or {}
        preset = (data.get("preset") or "generic_agent_lines").strip()
        language = (data.get('language') or 'Polish (pl-PL)').strip()
        accent = (data.get('accent') or '').strip()
        brief = (data.get('brief') or '').strip()
        details = (data.get('details') or '').strip()
        target_words = int(data.get("target_words") or 1200)
        auto_brief = bool(data.get("auto_brief") or False)

        scenes_count = data.get("scenes_count")
        min_turns = data.get("min_turns")
        max_turns = data.get("max_turns")
        dialect = data.get("dialect")

        model = (data.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-4.1").strip()

        if not (brief or details):
            return jsonify({'error': 'Wpisz przynajmniej "Szczegóły" albo "Brief".'}), 400

        script_text, meta = generate_script(
            preset=preset,
            language=language,
            accent=accent,
            brief=brief,
            details=details,
            model=model,
            target_words=target_words,
            auto_brief=auto_brief,
            scenes_count=int(scenes_count) if scenes_count else None,
            min_turns=int(min_turns) if min_turns else None,
            max_turns=int(max_turns) if max_turns else None,
            dialect=str(dialect).strip() if dialect else None,
        )
        
        # Save to temp file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_preset = re.sub(r"[^a-zA-Z0-9_-]+", "_", preset)[:40]
        filename = f"script_{safe_preset}_{timestamp}.txt"
        meta_filename = f"script_{safe_preset}_{timestamp}.meta.json"
        
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        meta_path = os.path.join(temp_dir, meta_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(script_text)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'meta_filename': meta_filename,
            'preview': script_text[:500] + '...' if len(script_text) > 500 else script_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


