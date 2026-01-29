import json
import re
import time
from datetime import datetime
from typing import Any

from rq import get_current_job

from storage_pg import save_script

# Import generation helpers from the existing app module.
# This avoids duplicating the OpenAI prompt logic.
from app import generate_script, _maybe_generate_auto_brief, _generate_chat  # type: ignore


def _safe_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s or "").strip("_")
    return s[:40] or "script"


def _job_set_meta(job, **updates):
    job.meta.update(updates)
    job.save_meta()


def _estimate_segments(target_minutes: int) -> int:
    # 90 min -> 15 segments of 6 min, capped.
    seg_minutes = 6
    total = max(1, int(round(target_minutes / seg_minutes)))
    return min(30, max(1, total))

_V3_SYSTEM_PROMPT = (
    "You are a dialogue enhancer for speech generation.\n"
    "\n"
    "PRIMARY GOAL: Add expressive, voice-only audio tags in square brackets [] to make the text feel emotional and engaging,\n"
    "while STRICTLY preserving the original wording and meaning.\n"
    "\n"
    "HARD RULES (MUST):\n"
    "- Do NOT add, remove, or reorder ANY words from the original text.\n"
    "- Do NOT add new sentences or new lines.\n"
    "- Do NOT put original words inside brackets. Brackets are ONLY for NEW audio tags.\n"
    "- Tags must be VOICE-ONLY / AUDIBLE (emotion/delivery + vocal non-verbal). No music/SFX.\n"
    "- Do NOT use tags like [music], [standing], [grinning], [pacing].\n"
    "- Do NOT introduce or imply sensitive topics (politics, religion, hate, profanity, NSFW).\n"
    "\n"
    "QUALITY REQUIREMENTS (MUST):\n"
    "- Be emotionally accurate to the text.\n"
    "- Use a DIVERSE set of tags; do not overuse a single tag.\n"
    "- Limit pause tags: use at most ONE pause tag per paragraph; prefer no pause tags if not needed.\n"
    "- Add at least ONE expressive NON-PAUSE tag per paragraph, unless the paragraph is extremely short.\n"
    "- If the text is tense/fast/exciting (e.g., sports commentary), prefer energetic delivery tags\n"
    "  (e.g., [excited], [hyped], [breathless], [shouting], [gasps], [cheering], [laughing], [exhales sharply]).\n"
    "- Avoid sad/soft tags unless the text clearly calls for it (loss, heartbreak, disappointment).\n"
    "\n"
    "EMPHASIS:\n"
    "- You MAY add emphasis using punctuation, ellipses, and capitalization, but do not change words.\n"
    "\n"
    "OUTPUT: Return ONLY the enhanced text."
)

def _v3_enhance_text(*, language: str, text: str, model: str) -> str:
    """
    Post-processes text by injecting voice-only audio tags in [].
    Must not change wording; only tags + light emphasis are allowed.
    """
    text = (text or "").strip()
    if not text:
        return text

    sys = {"role": "system", "content": _V3_SYSTEM_PROMPT}
    usr = {
        "role": "user",
        "content": (
            f"LANGUAGE: {language}\n"
            "Enhance the following text with audio tags.\n\n"
            f"{text}\n"
        ),
    }

    wc = len([w for w in text.split() if w.strip()])
    max_tokens = min(3500, max(700, int(wc * 2.3)))
    try:
        out = _generate_chat([sys, usr], model=model, temperature=0.25, top_p=0.9, max_tokens=max_tokens)
        return out.strip()
    except Exception:
        return text

def _build_outline(*, language: str, accent: str, script_type: str, brief: str, details: str, target_minutes: int, segments: int, model: str) -> list[dict[str, Any]]:
    """
    Returns a list of segments: [{title, goal, minutes, key_points[]}].
    Best-effort: falls back to a simple outline on parse failure.
    """
    sys = {
        "role": "system",
        "content": "You are an expert script outliner for long-form voiceover. Return ONLY valid JSON.",
    }
    usr = {
        "role": "user",
        "content": (
            f"LANGUAGE: {language}\n"
            f"ACCENT/DIALECT: {accent or 'not specified'}\n"
            f"SCRIPT TYPE: {script_type}\n"
            f"TARGET LENGTH MINUTES: {target_minutes}\n"
            f"SEGMENTS: {segments}\n\n"
            f"BRIEF:\n{brief}\n\n"
            f"DETAILS:\n{details}\n\n"
            "Create a segment-by-segment outline for a single-speaker script.\n"
            "Return JSON with this exact shape:\n"
            "{\n"
            '  "segments": [\n'
            '    {"title": "…", "goal": "…", "minutes": 6, "key_points": ["…", "..."]}\n'
            "  ]\n"
            "}\n"
            "Rules: minutes should roughly sum to TARGET LENGTH MINUTES; keep titles short."
        ),
    }
    try:
        raw = _generate_chat([sys, usr], model=model, temperature=0.4, top_p=0.9, max_tokens=1200)
        obj = json.loads(raw)
        segs = obj.get("segments") if isinstance(obj, dict) else None
        if isinstance(segs, list) and segs:
            cleaned = []
            for i, s in enumerate(segs[:segments]):
                if not isinstance(s, dict):
                    continue
                cleaned.append(
                    {
                        "title": str(s.get("title") or f"Segment {i+1}").strip(),
                        "goal": str(s.get("goal") or "").strip(),
                        "minutes": int(s.get("minutes") or max(1, target_minutes // max(1, segments))),
                        "key_points": [str(x).strip() for x in (s.get("key_points") or []) if str(x).strip()],
                    }
                )
            if cleaned:
                return cleaned
    except Exception:
        pass

    # Fallback
    minutes_each = max(2, int(round(target_minutes / max(1, segments))))
    return [
        {"title": f"Segment {i+1}", "goal": f"Continue the {script_type} script", "minutes": minutes_each, "key_points": []}
        for i in range(segments)
    ]


def _update_continuity(*, language: str, last_segment_text: str, continuity: str, model: str) -> str:
    """
    Keeps a small continuity doc: names, choices, terminology, do/don't.
    """
    sys = {
        "role": "system",
        "content": "You maintain a continuity note for a long voiceover script. Return ONLY plain text.",
    }
    usr = {
        "role": "user",
        "content": (
            f"LANGUAGE: {language}\n\n"
            "Update the continuity note based on the latest segment.\n"
            "Keep it under 12 lines. Include: key terms, audience, tone, any specific promises, and any recurring wording.\n\n"
            f"CURRENT CONTINUITY NOTE:\n{continuity or '(none yet)'}\n\n"
            f"LATEST SEGMENT:\n{last_segment_text[-2200:]}\n"
        ),
    }
    try:
        return _generate_chat([sys, usr], model=model, temperature=0.2, top_p=0.9, max_tokens=260)
    except Exception:
        return continuity


def run_generate_job(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Background job: generates a long script in segments and stores results in Postgres.
    Returns a result dict saved in job.meta['result'].
    """
    job = get_current_job()
    if job is None:
        raise RuntimeError("RQ job context missing")

    started = time.time()
    _job_set_meta(job, status="running", stage="init", progress=0, message="Starting…")

    script_type = (payload.get("script_type") or "").strip()
    language = (payload.get("language") or "English").strip()
    accent = (payload.get("accent") or "").strip()
    brief = (payload.get("brief") or "").strip()
    details = (payload.get("details") or "").strip()
    target_minutes = int(payload.get("target_minutes") or 5)
    model = (payload.get("model") or "gpt-4.1").strip()
    v3 = bool(payload.get("v3"))

    try:
        # If both empty, seed details so auto-brief has something to work with.
        if not brief and not details:
            details = (
                "No user-provided details. Invent a simple, realistic scenario that fits the script type, "
                "and choose a clear audience, tone, and a few key points that are easy to record."
            )

        if not brief:
            _job_set_meta(job, stage="planning", progress=3, message="Creating a brief…")
            brief = _maybe_generate_auto_brief(language=language, accent=accent, script_type=script_type, details=details, model=model)

        total_segments = _estimate_segments(target_minutes)
        _job_set_meta(job, stage="planning_outline", total_segments=total_segments, current_segment=0, progress=6, message="Creating outline…")
        outline = _build_outline(
            language=language,
            accent=accent,
            script_type=script_type,
            brief=brief,
            details=details,
            target_minutes=target_minutes,
            segments=total_segments,
            model=model,
        )

        _job_set_meta(job, stage="generating", progress=8, message="Generating…")

        full_text_parts: list[str] = []
        seam_tail = ""
        continuity_note = ""

        for i, seg in enumerate(outline):
            seg_minutes = int(seg.get("minutes") or max(2, int(round(target_minutes / max(1, total_segments)))))
            seg_words = int(seg_minutes * 150)
            seg_title = str(seg.get("title") or f"Segment {i+1}")
            seg_goal = str(seg.get("goal") or "")
            seg_points = seg.get("key_points") or []

            seg_details = (
                f"{details}\n\n"
                f"SEGMENT TITLE: {seg_title}\n"
                f"SEGMENT GOAL: {seg_goal}\n"
                f"KEY POINTS: {', '.join([str(x) for x in seg_points]) if seg_points else 'not specified'}\n\n"
            )
            if continuity_note:
                seg_details += f"CONTINUITY NOTE (keep consistent):\n{continuity_note}\n\n"
            if seam_tail:
                seg_details += (
                    "SEAM (last part of previous segment; continue smoothly, do not repeat):\n"
                    f"{seam_tail}\n"
                )

            _job_set_meta(
                job,
                current_segment=i + 1,
                progress=int(10 + (75 * (i / max(1, total_segments)))),
                message=f"Generating segment {i+1}/{total_segments}: {seg_title}",
            )

            seg_text, _meta = generate_script(
                script_type=script_type,
                language=language,
                accent=accent,
                brief=brief,
                details=seg_details,
                model=model,
                target_words=seg_words,
                target_minutes=seg_minutes,
            )

            seg_text_raw = seg_text.strip()
            seg_text_final = seg_text_raw

            if v3 and seg_text_raw:
                seg_text_final = _v3_enhance_text(language=language, text=seg_text_raw, model=model)

            full_text_parts.append(seg_text_final)

            # Keep continuity + seam based on the raw text (pre-v3) so tags don't pollute generation.
            seam_tail = seg_text_raw[-1200:]
            continuity_note = _update_continuity(language=language, last_segment_text=seg_text_raw, continuity=continuity_note, model=model)

            if i == 0 or i == total_segments - 1:
                preview = "\n\n".join(full_text_parts)[:600]
                _job_set_meta(job, preview=preview)

        _job_set_meta(job, stage="uploading", progress=90, message="Uploading…")

        full_text = "\n\n".join(full_text_parts).strip() + "\n"
        word_count = len([w for w in full_text.split() if w.strip()])
        estimated_minutes = max(1, int(round(word_count / 180)))

        meta = {
            "job_id": job.id,
            "script_type": script_type,
            "language": language,
            "accent": accent,
            "target_minutes": target_minutes,
            "segments": total_segments,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": int(time.time() - started),
            "brief_used": brief,
            "continuity_note": continuity_note,
            "word_count": word_count,
            "estimated_minutes": estimated_minutes,
            "v3": v3,
        }

        save_script(
            job_id=job.id,
            script_type=script_type,
            language=language,
            accent=accent,
            target_minutes=target_minutes,
            script_txt=full_text,
            meta=meta,
            outline={"segments": outline},
        )

        result = {
            "success": True,
            "job_id": job.id,
            "txt_url": f"/download-job/{job.id}.txt",
            "meta_url": f"/download-job/{job.id}.meta.json",
            "outline_url": f"/download-job/{job.id}.outline.json",
            "preview": full_text[:500] + ("..." if len(full_text) > 500 else ""),
            "word_count": word_count,
            "estimated_minutes": estimated_minutes,
        }

        _job_set_meta(job, status="finished", stage="done", progress=100, message="Done.", result=result)
        return result
    except Exception as e:
        _job_set_meta(job, status="failed", stage="error", progress=100, message="Failed.", error=str(e))
        raise

