import argparse
import base64
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Tuple

import requests
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from PIL import Image

# ---------- PROMPT DEFAULTS ----------
DEFAULT_VISION_OCR_PROMPT = (
    "Extract only the readable text that is physically present in the image, in natural reading order. "
    "The output language and script must match the visible source text exactly. "
    "If the visible text is Chinese, output Chinese only except for short visible terms such as HP, LV, names, numbers, symbols, or UI labels. "
    "Return only the base source text. Preserve paragraph breaks and intentional textual layout when they affect meaning, but do not reproduce every visual wrap from the image; merge wrapped lines that belong to the same sentence or paragraph. "
    "Preserve visible textual styling and marks when possible, including brackets, parentheses, quote marks, emphasis, strikethrough/deleted text, symbols, and emotes/emoticons. "
    "For Chinese/Japanese/Korean text with small pronunciation guides above or beside the main characters, OCR only the main/base characters and ignore the pronunciation guides. "
    "For pinyin-over-Chinese images, output the Chinese characters only; do not output the pinyin unless the pinyin is standalone text with no matching Chinese base text. "
    "Do not translate, summarize, explain, annotate, transliterate, romanize, or add pronunciation guides. "
    "Do not output duplicate reading lines such as pinyin, romaji, furigana, Jyutping, or Latin readings when they are attached to the same base text. "
    "Never invent filler, examples, schedules, notices, continuation text, or unrelated content that is not visible in the image."
)
DEFAULT_VISION_OCR_USER_PROMPT = (
    "OCR this image/chunk. Return only the main/base source text. "
    "Keep the original visible language/script. Ignore pinyin/romaji/furigana/Jyutping pronunciation guides attached to base characters. "
    "Do not translate or add any unrelated text."
)
# ---------------------------------------


@dataclass
class PipelineConfig:
    """All tunable pipeline settings in one place.

    Create a module-level instance and mutate its fields before calling
    the pipeline functions.  CLI and GUI both populate the same struct.
    """

    # Image splitting
    smart_line_chunking: bool = True
    target_chunk_height: int = 1200
    min_chunk_height: int = 600
    max_foreground_ratio: float = 0.01
    min_gap_rows: int = 12
    cut_padding_rows: int = 4
    min_ocr_foreground_pixels: int = 24
    min_ocr_foreground_ratio: float = 0.0002
    overlap_percent: float = 3.0
    min_overlap_pixels: int = 80

    # Deduplication
    sim_threshold: float = 0.85
    min_dedupe_length: int = 30
    enable_dedupe: bool = False

    # API / concurrency
    max_workers: int = 4
    api_call_delay_seconds: float = 2.0
    temperature: float = 0.0
    max_output_tokens: int = 8192
    streaming: bool = True
    request_prep_lock_microseconds: int = 250

    # Thinking / reasoning
    send_thinking_parameters: bool = False
    gemini_thinking_level: str = "minimal"
    openai_thinking_effort: str = "none"

    # Output / post-processing
    cleanup: bool = True
    flatten_to_white_background: bool = True
    skip_pinyin_lines: bool = False
    skip_romanization_lines: bool = False

    # Paths / defaults
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    max_chunk_bytes: int = 10 * 1024 * 1024  # 10 MB; Vision hard limit is 20 MB
    default_output_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    chunks_dir_name: str = "OCR Chunks"
    ocr_results_dir_name: str = "OCR Results"
    use_existing_chunks: bool = False
    default_ocr_model: str = "gemini-3.1-flash-lite"
    google_cloud_vision_model: str = "google-cloud-vision"

    # LM Studio / local endpoints
    lmstudio_base_url: str = "http://localhost:1234/v1"

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"

    # Output / resume
    output_format: str = "html"          # html, txt, md, json
    resume_processing: bool = True       # skip images whose output already exists
    dry_run: bool = False                # print chunk plan without calling APIs
    truncation_retries: int = 2          # retry count before auto re-chunking
    export_epub: bool = False            # create EPUB after processing all images

    # Text cleanup
    cleanup_rules: list = field(default_factory=list)  # [{pattern, replacement, enabled}]
    cleanup_only: bool = False           # skip OCR, clean existing .html files

    # Prompts (overridable per call, but these are the module-level defaults)
    vision_ocr_prompt: str = DEFAULT_VISION_OCR_PROMPT
    vision_ocr_user_prompt: str = DEFAULT_VISION_OCR_USER_PROMPT


# Single module-level config instance – replaces the old global variables.
_config = PipelineConfig()

# Runtime state (not config)
_shutdown = False
_request_prep_lock = threading.Lock()
_stream_output_lock = threading.Lock()

# Truncation tracking and re-chunk metadata
_chunk_truncated: dict[str, int] = {}      # chunk_path → retry count
_chunk_meta: dict[str, tuple[str, int, int]] = {}  # chunk_path → (source_image, y0, y1)


def _reset_chunk_state():
    _chunk_truncated.clear()
    _chunk_meta.clear()


def _handle_sigint(sig, frame):
    global _shutdown
    print("\n[!] Interrupt received; finishing in-flight requests then exiting cleanly.")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_sigint)


# ---- Image splitting ----

def _percentile_from_histogram(histogram: list[int], percentile: float) -> float:
    total = sum(histogram)
    if total <= 0:
        return 0.0
    target = total * percentile
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return float(value)
    return 255.0


def _foreground_classifier(histogram: list[int], low: float | None = None, high: float | None = None) -> tuple[float, bool] | None:
    """Returns (threshold, dark_is_foreground) or None if contrast too low."""
    if low is None:
        low = _percentile_from_histogram(histogram, 0.05)
    if high is None:
        high = _percentile_from_histogram(histogram, 0.95)
    med = _percentile_from_histogram(histogram, 0.50)
    contrast = high - low
    if contrast < 18:
        return None

    if med >= 128:
        threshold = min(245.0, low + max(25.0, contrast * 0.35))
        return (threshold, True)
    else:
        threshold = max(10.0, high - max(25.0, contrast * 0.35))
        return (threshold, False)


def _row_foreground_counts(img: Image.Image) -> list[int]:
    """
    Count likely foreground/text pixels per row.

    - Transparent images: visible alpha pixels are foreground.
    - Light backgrounds: dark pixels are foreground.
    - Dark backgrounds: bright pixels are foreground.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    data = rgba.tobytes()

    alpha_values = data[3::4]
    if alpha_values and min(alpha_values) != max(alpha_values):
        counts = []
        for y in range(h):
            row_start = y * w * 4 + 3
            visible = 0
            for idx in range(row_start, row_start + w * 4, 4):
                if data[idx] > 16:
                    visible += 1
            counts.append(visible)
        if any(counts):
            return counts

    gray = img.convert("L")
    histogram = gray.histogram()
    classifier = _foreground_classifier(histogram)
    if classifier is None:
        return [w] * h

    threshold, dark_is_foreground = classifier
    raw = gray.tobytes()
    counts = []
    for y in range(h):
        row = raw[y * w:(y + 1) * w]
        if dark_is_foreground:
            counts.append(sum(1 for value in row if value <= threshold))
        else:
            counts.append(sum(1 for value in row if value >= threshold))
    return counts


def _flatten_to_white(img: Image.Image) -> Image.Image:
    """Composite transparent/palette images onto white for OCR-friendly chunks."""
    has_transparency = (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )
    if not has_transparency:
        return img.convert("RGB") if img.mode != "RGB" else img

    rgba = img.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[3])
    return background


def _likely_ocr_foreground_pixels(img: Image.Image) -> int:
    """Count likely OCR text pixels. Low-contrast all-white/all-dark crops are blank."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    data = rgba.tobytes()

    alpha_values = data[3::4]
    if alpha_values and max(alpha_values) <= 16:
        return 0

    gray = img.convert("L")
    histogram = gray.histogram()
    classifier = _foreground_classifier(histogram)

    if classifier is None:
        observed_low = next((value for value, count in enumerate(histogram) if count), 0)
        observed_high = next((value for value in range(255, -1, -1) if histogram[value]), 255)
        classifier = _foreground_classifier(histogram, low=float(observed_low), high=float(observed_high))
        if classifier is None:
            return 0

    threshold, dark_is_foreground = classifier
    if dark_is_foreground:
        return sum(1 for value in gray.tobytes() if value <= threshold)
    return sum(1 for value in gray.tobytes() if value >= threshold)


def _has_meaningful_ocr_foreground(img: Image.Image) -> bool:
    w, h = img.size
    foreground_pixels = _likely_ocr_foreground_pixels(img)
    required_pixels = max(_config.min_ocr_foreground_pixels, int(w * h * _config.min_ocr_foreground_ratio))
    return foreground_pixels >= required_pixels


def _find_smart_cut_points(foreground_counts: list[int], width: int, total_height: int) -> list[int]:
    max_foreground_pixels = max(0, int(width * _config.max_foreground_ratio))

    candidates = []
    in_gap = False
    gap_start = 0

    for y, foreground_count in enumerate(foreground_counts[:total_height]):
        if foreground_count <= max_foreground_pixels:
            if not in_gap:
                in_gap = True
                gap_start = y
        elif in_gap:
            if y - gap_start >= _config.min_gap_rows:
                candidates.append((gap_start + y) // 2)
            in_gap = False

    if in_gap and total_height - gap_start >= _config.min_gap_rows:
        candidates.append((gap_start + total_height) // 2)

    cuts = []
    last = 0
    while last + _config.target_chunk_height < total_height:
        target = last + _config.target_chunk_height
        min_next = last + _config.min_chunk_height
        reachable = [c for c in candidates if c > min_next]
        if not reachable:
            break

        best = min(
            reachable,
            key=lambda c: (
                abs(c - target),
                max(foreground_counts[max(0, c - _config.cut_padding_rows):min(total_height, c + _config.cut_padding_rows + 1)] or [width]),
            ),
        )
        if best >= total_height:
            break
        cuts.append(best)
        last = best

    for cut in cuts:
        start = max(0, cut - _config.cut_padding_rows)
        end = min(total_height, cut + _config.cut_padding_rows + 1)
        if any(count > max_foreground_pixels for count in foreground_counts[start:end]):
            return []

    return cuts


def _smart_chunk_ranges(img: Image.Image) -> list[tuple[int, int]]:
    w, h = img.size
    if h <= _config.target_chunk_height:
        return [(0, h)]

    foreground_counts = _row_foreground_counts(img)
    cuts = _find_smart_cut_points(foreground_counts, w, h)
    if not cuts:
        return []

    ranges = list(zip([0] + cuts, cuts + [h]))
    max_reasonable_height = max(_config.target_chunk_height * 2, _config.target_chunk_height + _config.min_chunk_height)
    if any((end_y - start_y) > max_reasonable_height for start_y, end_y in ranges):
        return []
    return ranges


def _overlap_chunk_ranges(total_height: int) -> list[tuple[int, int]]:
    overlap = max(int(_config.target_chunk_height * (_config.overlap_percent / 100.0)), _config.min_overlap_pixels)
    overlap = min(overlap, _config.target_chunk_height - 1)

    ranges = []
    start_y = 0
    while start_y < total_height:
        end_y = min(total_height, start_y + _config.target_chunk_height)
        ranges.append((start_y, end_y))
        if end_y >= total_height:
            break
        next_start = end_y - overlap
        if next_start <= start_y:
            next_start = end_y
        start_y = next_start
    return ranges


def split_image(path: str, output_dir: str) -> list[str]:
    """
    Slice an image into chunks.

    Smart chunking is enabled by default and tries to cut on clean horizontal
    gaps. If it cannot find reliable cuts, the splitter falls back to overlapped
    fixed-height chunks with an 80px floor.
    """
    img = Image.open(path)
    if _config.flatten_to_white_background:
        img = _flatten_to_white(img)
    w, h = img.size
    base = os.path.splitext(os.path.basename(path))[0]

    ranges = []
    if _config.smart_line_chunking:
        ranges = _smart_chunk_ranges(img)
        if ranges:
            print(f"  Smart chunking selected {len(ranges)} clean line-boundary chunk(s)")

    if not ranges:
        ranges = _overlap_chunk_ranges(h)
        print(f"  Fallback overlap chunking selected {len(ranges)} chunk(s)")

    chunk_files = []
    skipped_blank = 0
    for i, (y0, y1) in enumerate(ranges, start=1):
        chunk = img.crop((0, y0, w, y1))
        if not _has_meaningful_ocr_foreground(chunk):
            skipped_blank += 1
            continue
        fname = os.path.join(output_dir, f"{base}_chunk_{i:04d}.png")
        chunk.save(fname, format="PNG")
        chunk_files.append((fname, y0, y1))

    if skipped_blank:
        print(f"  Skipped {skipped_blank} whitespace-only chunk(s)")

    return chunk_files


def _image_chunks_dir(output_dir: str, job_name: str, image_base: str) -> str:
    """OCR Chunks/<job_name>/<image_base>/"""
    return os.path.join(output_dir, _config.chunks_dir_name, job_name, image_base)


def _results_dir(output_dir: str, job_name: str) -> str:
    """OCR Results/<job_name>/"""
    return os.path.join(output_dir, _config.ocr_results_dir_name, job_name)


def chunk_image_only(image_path: str, output_dir: str, job_name: str = "") -> list[str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    if job_name:
        chunk_dir = _image_chunks_dir(output_dir, job_name, base)
    else:
        chunk_dir = os.path.join(output_dir, f"{base}_chunks")
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    with Image.open(image_path) as img:
        iw, ih = img.size
    print(f"\n-> {base}")
    print(f"  Chunk-only mode: splitting {iw}x{ih}px image")
    chunk_info = split_image(image_path, chunk_dir)
    print(f"  Saved {len(chunk_info)} chunk(s) -> {chunk_dir}")
    return [info[0] for info in chunk_info]


def should_split(image_path: str) -> bool:
    with Image.open(image_path) as img:
        _, h = img.size
    return h > _config.min_chunk_height or os.path.getsize(image_path) > _config.max_chunk_bytes


# ---- Cloud Vision OCR ----

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
SPACE_BREAKS = {"SPACE", "SURE_SPACE", "EOL_SURE_SPACE"}
VISION_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _word_text(word: dict) -> str:
    text = ""
    for sym in word.get("symbols", []):
        text += sym.get("text", "")
        break_type = (
            sym.get("property", {})
               .get("detectedBreak", {})
               .get("type", "")
        )
        if break_type in SPACE_BREAKS:
            text += " "
    return text


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


_PINYIN_TONE_RE = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛÜ]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PINYIN_SYLLABLES = {
    "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao",
    "bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng", "cha", "chai",
    "chan", "chang", "chao", "che", "chen", "cheng", "chi", "chong", "chou", "chu", "chuai", "chuan", "chuang",
    "chui", "chun", "chuo", "ci", "cong", "cou", "cu", "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang",
    "dao", "de", "dei", "den", "deng", "di", "dia", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du",
    "duan", "dui", "dun", "duo", "e", "ei", "en", "eng", "er", "fa", "fan", "fang", "fei", "fen", "feng", "fo",
    "fou", "fu", "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua",
    "guai", "guan", "guang", "gui", "gun", "guo", "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng",
    "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo", "ji", "jia", "jian", "jiang",
    "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang", "kao",
    "ke", "ken", "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai",
    "lan", "lang", "lao", "le", "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu",
    "lo", "long", "lou", "lu", "luan", "lue", "lun", "luo", "lv", "lü", "ma", "mai", "man", "mang", "mao", "me",
    "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu", "na", "nai", "nan",
    "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu", "nong",
    "nou", "nu", "nuan", "nue", "nuo", "nv", "nü", "o", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen",
    "peng", "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu", "qi", "qia", "qian", "qiang", "qiao",
    "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun", "ran", "rang", "rao", "re", "ren", "reng",
    "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang", "sao", "se", "sen", "seng",
    "sha", "shai", "shan", "shang", "shao", "she", "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan",
    "shuang", "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai", "tan",
    "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun",
    "tuo", "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian", "xiang", "xiao",
    "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun", "ya", "yan", "yang", "yao", "ye", "yi",
    "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei",
    "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu",
    "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun",
    "zuo",
}


def _looks_like_pinyin_line(line: str) -> bool:
    if _CJK_RE.search(line):
        return False
    if re.search(r"\d", line):
        return False
    normalized = line.lower().replace("u:", "ü").replace("v", "ü")
    tokens = re.findall(r"[a-zü]+", normalized)
    if not tokens:
        return False
    pinyin_hits = sum(1 for token in tokens if token in _PINYIN_SYLLABLES)
    return bool(_PINYIN_TONE_RE.search(line)) or pinyin_hits / max(len(tokens), 1) >= 0.75


def _looks_like_romanization_line(line: str) -> bool:
    if _CJK_RE.search(line):
        return False
    if re.search(r"\d", line):
        return False
    letters = _ASCII_LETTER_RE.findall(line)
    if len(letters) < 3:
        return False
    meaningful = re.sub(r"[\s,.;:'\"!?()\[\]{}<>/\\|_~`^+\-]+", "", line)
    if not meaningful:
        return False
    letter_ratio = len(letters) / max(len(meaningful), 1)
    tokens = re.findall(r"[A-Za-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛÜ]+", line)
    return letter_ratio >= 0.75 and len(tokens) >= 2


def annotation_to_html_blocks(annotation: dict) -> list[str]:
    """
    Convert Cloud Vision OCR to simple paragraphs.

    The raw block hierarchy often fragments webnovel screenshots and creates
    chaotic heading guesses, so the default output uses Vision's full text and
    preserves its line breaks inside normal paragraph tags.
    """
    text = annotation.get("text", "")
    if not text.strip():
        return []

    paragraphs = []
    current = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _config.skip_pinyin_lines and _looks_like_pinyin_line(line):
            continue
        if _config.skip_romanization_lines and _looks_like_romanization_line(line):
            continue
        if line:
            current.append(_html_escape(line))
        elif current:
            paragraphs.append("<p>" + "<br>\n".join(current) + "</p>")
            current = []
    if current:
        paragraphs.append("<p>" + "<br>\n".join(current) + "</p>")
    return paragraphs


def _service_account_headers(credentials_path: str) -> dict[str, str]:
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=[VISION_SCOPE],
    )
    credentials.refresh(GoogleAuthRequest())
    return {"Authorization": f"Bearer {credentials.token}"}


def _image_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")
    return f"data:{mime};base64,{encoded}"


def _plain_text_to_blocks(text: str) -> list[str]:
    text = (text or "").strip()
    if not text or text.strip().lower() == "no":
        return []
    return annotation_to_html_blocks({"text": text})


def _extract_openai_text(result: dict) -> tuple[str, bool]:
    choices = result.get("choices", [])
    if not choices:
        return "", False
    truncated = choices[0].get("finish_reason") == "length"
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content, truncated
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts), truncated
    return "", truncated


def _extract_gemini_text(result: dict) -> tuple[str, bool]:
    parts = []
    truncated = False
    for candidate in result.get("candidates", []):
        if candidate.get("finishReason") == "MAX_TOKENS":
            truncated = True
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts), truncated


def _extract_anthropic_text(result: dict) -> tuple[str, bool]:
    parts = []
    stop_reason = result.get("stop_reason", "")
    truncated = stop_reason == "max_tokens"
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts), truncated


def _normalize_choice(value: str, allowed: tuple[str, ...], default: str) -> str:
    normalized = (value or default).strip().lower()
    return normalized if normalized in allowed else default


def _locked_request_settings() -> tuple[float, int, bool, bool, str, str]:
    """
    Snapshot shared request settings behind a tiny lock.

    The lock is deliberately held for microseconds and released before any HTTP
    call, so parallel OCR slots still overlap while avoiding mixed global reads.
    """
    with _request_prep_lock:
        temperature = float(_config.temperature)
        max_output_tokens = max(1, int(_config.max_output_tokens))
        streaming = bool(_config.streaming)
        thinking_enabled = bool(_config.send_thinking_parameters)
        gemini_level = _normalize_choice(
            _config.gemini_thinking_level,
            ("minimal", "low", "medium", "high"),
            "minimal",
        )
        openai_effort = _normalize_choice(
            _config.openai_thinking_effort,
            ("none", "low", "medium", "high", "xhigh"),
            "none",
        )
        if _config.request_prep_lock_microseconds > 0:
            time.sleep(_config.request_prep_lock_microseconds / 1_000_000.0)
    return temperature, max_output_tokens, streaming, thinking_enabled, gemini_level, openai_effort


def _is_openrouter_style(model: str, base_url: str) -> bool:
    lowered_url = (base_url or "").lower()
    lowered_model = (model or "").lower()
    return "openrouter" in lowered_url or lowered_model.startswith("openrouter/")


def _apply_openai_thinking(payload: dict, model: str, base_url: str, effort: str) -> None:
    effort = _normalize_choice(effort, ("none", "low", "medium", "high", "xhigh"), "none")
    if _is_openrouter_style(model, base_url) or effort in ("none", "xhigh"):
        payload["reasoning"] = {"enabled": True, "exclude": True, "effort": effort}
    else:
        payload["reasoning_effort"] = effort


def _iter_sse_json(response: requests.Response):
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _iter_httpx_sse_json(response: httpx.Response):
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _emit_stream_text(text: str) -> None:
    if not text:
        return
    with _stream_output_lock:
        sys.stdout.write(text)
        sys.stdout.flush()


def _extract_openai_stream_text(response: requests.Response) -> tuple[str, bool]:
    parts = []
    truncated = False
    for chunk in _iter_sse_json(response):
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(item["text"])
            if choice.get("finish_reason") == "length":
                truncated = True
    return "".join(parts), truncated


def _extract_openai_httpx_stream_text(response: httpx.Response) -> tuple[str, bool]:
    parts = []
    truncated = False
    for chunk in _iter_httpx_sse_json(response):
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if isinstance(content, str):
                parts.append(content)
                _emit_stream_text(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        text = item["text"]
                        parts.append(text)
                        _emit_stream_text(text)
            if choice.get("finish_reason") == "length":
                truncated = True
    if parts:
        _emit_stream_text("\n")
    return "".join(parts), truncated


def _extract_gemini_stream_text(response: requests.Response) -> tuple[str, bool]:
    parts = []
    truncated = False
    for chunk in _iter_sse_json(response):
        text, chunk_truncated = _extract_gemini_text(chunk)
        if text:
            parts.append(text)
        if chunk_truncated:
            truncated = True
    return "".join(parts), truncated


def _extract_gemini_httpx_stream_text(response: httpx.Response) -> tuple[str, bool]:
    parts = []
    truncated = False
    for chunk in _iter_httpx_sse_json(response):
        text, chunk_truncated = _extract_gemini_text(chunk)
        if text:
            parts.append(text)
            _emit_stream_text(text)
        if chunk_truncated:
            truncated = True
    if parts:
        _emit_stream_text("\n")
    return "".join(parts), truncated


def _extract_anthropic_httpx_stream_text(response: httpx.Response) -> tuple[str, bool]:
    parts = []
    truncated = False
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("event:") or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        try:
            chunk = json.loads(payload)
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        parts.append(text)
                        _emit_stream_text(text)
            elif chunk.get("type") == "message_delta":
                delta = chunk.get("delta", {})
                if isinstance(delta, dict) and delta.get("stop_reason") == "max_tokens":
                    truncated = True
        except json.JSONDecodeError:
            continue
    if parts:
        _emit_stream_text("\n")
    return "".join(parts), truncated


def _response_error_message(response: requests.Response, provider: str) -> str:
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                detail = error.get("message") or error.get("status") or ""
            elif isinstance(error, str):
                detail = error
    except (ValueError, RuntimeError):
        try:
            detail = (response.text or "").strip()
        except RuntimeError:
            detail = ""

    if detail:
        detail = detail.replace("\n", " ")[:500]
        return f"{provider} API error {response.status_code}: {detail}"
    return f"{provider} API error {response.status_code}: {response.reason}"


def _httpx_response_error_message(response: httpx.Response, provider: str) -> str:
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                detail = error.get("message") or error.get("status") or ""
            elif isinstance(error, str):
                detail = error
    except ValueError:
        detail = (response.text or "").strip()

    if detail:
        detail = detail.replace("\n", " ")[:500]
        return f"{provider} API error {response.status_code}: {detail}"
    return f"{provider} API error {response.status_code}: {response.reason_phrase}"


def _raise_for_status_clean(response: requests.Response, provider: str) -> None:
    if response.status_code >= 400:
        raise RuntimeError(_response_error_message(response, provider))


def _raise_httpx_for_status_clean(response: httpx.Response, provider: str) -> None:
    if response.status_code >= 400:
        try:
            response.read()
        except RuntimeError:
            pass
        raise RuntimeError(_httpx_response_error_message(response, provider))


def _ocr_image_google_cloud(image_path: str, api_key: str = "", credentials_path: str = "") -> list[str]:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "requests": [{
            "image": {"content": encoded},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }

    headers = {"Content-Type": "application/json"}
    params = {}
    if credentials_path:
        headers.update(_service_account_headers(credentials_path))
    else:
        params["key"] = api_key

    response = requests.post(VISION_URL, params=params, headers=headers, data=json.dumps(payload), timeout=60)
    _raise_for_status_clean(response, "Google Cloud Vision")

    result = response.json()
    api_error = result.get("responses", [{}])[0].get("error", {})
    if api_error:
        raise RuntimeError(
            f"Vision API error for {image_path!r}: "
            f"[{api_error.get('code')}] {api_error.get('message')}"
        )

    annotation = result.get("responses", [{}])[0].get("fullTextAnnotation", {})
    return annotation_to_html_blocks(annotation)


def _ocr_image_openai_compatible(
    image_path: str,
    api_key: str,
    model: str,
    base_url: str = "",
) -> list[str]:
    temperature, max_output_tokens, streaming, thinking_enabled, _, openai_effort = _locked_request_settings()
    system_prompt = _config.vision_ocr_prompt
    user_text = _config.vision_ocr_user_prompt
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            },
        ],
    }
    if streaming:
        payload["stream"] = True
    if thinking_enabled:
        _apply_openai_thinking(payload, model, base_url, openai_effort)

    def _warn_if_truncated(chunk_name, text, was_truncated):
        if was_truncated:
            _chunk_truncated[chunk_name] = _chunk_truncated.get(chunk_name, 0) + 1
            print(f"  Warning: {os.path.basename(chunk_name)} was truncated at {max_output_tokens} tokens. "
                  f"Consider increasing --max-output-tokens or reducing chunk height.")

    if streaming:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=httpx.Timeout(120.0, read=120.0)) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code >= 400 and thinking_enabled:
                    response.read()
                    payload.pop("reasoning", None)
                    payload.pop("reasoning_effort", None)
                    with client.stream("POST", url, headers=headers, json=payload) as retry_response:
                        _raise_httpx_for_status_clean(retry_response, "OpenAI-compatible")
                        text, truncated = _extract_openai_httpx_stream_text(retry_response)
                        _warn_if_truncated(image_path, text, truncated)
                        return _plain_text_to_blocks(text)
                _raise_httpx_for_status_clean(response, "OpenAI-compatible")
                text, truncated = _extract_openai_httpx_stream_text(response)
                _warn_if_truncated(image_path, text, truncated)
                return _plain_text_to_blocks(text)

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400 and thinking_enabled:
        payload.pop("reasoning", None)
        payload.pop("reasoning_effort", None)
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
    _raise_for_status_clean(response, "OpenAI-compatible")
    text, truncated = _extract_openai_text(response.json())
    _warn_if_truncated(image_path, text, truncated)
    return _plain_text_to_blocks(text)


def _ocr_image_gemini(
    image_path: str,
    api_key: str,
    model: str,
) -> list[str]:
    temperature, max_output_tokens, streaming, thinking_enabled, gemini_level, _ = _locked_request_settings()
    system_prompt = _config.vision_ocr_prompt
    user_text = _config.vision_ocr_user_prompt
    data_url = _image_data_url(image_path)
    header, encoded = data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")

    def _warn_if_truncated(text, was_truncated):
        if was_truncated:
            _chunk_truncated[image_path] = _chunk_truncated.get(image_path, 0) + 1
            print(f"  Warning: {os.path.basename(image_path)} was truncated at {max_output_tokens} tokens. "
                  f"Consider increasing --max-output-tokens or reducing chunk height.")

    def build_payload(use_thinking: bool) -> dict:
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"{system_prompt}\n\n{user_text}"},
                    {"inline_data": {"mime_type": mime, "data": encoded}},
                ]
            }],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
        }
        if use_thinking:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": gemini_level}
        return payload

    def generate_url(use_streaming: bool) -> str:
        method = "streamGenerateContent" if use_streaming else "generateContent"
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"

    def post_generate(use_streaming: bool, use_thinking: bool) -> requests.Response:
        url = generate_url(use_streaming)
        params = {}
        if use_streaming:
            params["alt"] = "sse"
        return requests.post(
            url,
            params=params,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=build_payload(use_thinking),
            timeout=120,
        )

    if streaming:
        stream_url = generate_url(True)
        stream_params = {"alt": "sse"}
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        with httpx.Client(timeout=httpx.Timeout(120.0, read=120.0)) as client:
            with client.stream(
                "POST",
                stream_url,
                params=stream_params,
                headers=headers,
                json=build_payload(thinking_enabled),
            ) as response:
                if response.status_code < 400:
                    text, truncated = _extract_gemini_httpx_stream_text(response)
                    _warn_if_truncated(text, truncated)
                    return _plain_text_to_blocks(text)
                response.read()

        print("  Gemini streaming rejected; retrying without streaming.")
        response = post_generate(False, thinking_enabled)
        if response.status_code >= 400 and thinking_enabled:
            print("  Gemini thinking params rejected; retrying without thinking params.")
            response = post_generate(False, False)
        _raise_for_status_clean(response, "Gemini")
        text, truncated = _extract_gemini_text(response.json())
        _warn_if_truncated(text, truncated)
        return _plain_text_to_blocks(text)

    response = post_generate(streaming, thinking_enabled)
    if response.status_code >= 400 and thinking_enabled:
        print("  Gemini thinking params rejected; retrying without thinking params.")
        response = post_generate(False, False)

    _raise_for_status_clean(response, "Gemini")
    text, truncated = _extract_gemini_text(response.json())
    _warn_if_truncated(text, truncated)
    return _plain_text_to_blocks(text)


def _ocr_image_anthropic(
    image_path: str,
    api_key: str,
    model: str,
) -> list[str]:
    temperature, max_tokens, streaming, _, _, _ = _locked_request_settings()
    system_prompt = _config.vision_ocr_prompt
    user_text = _config.vision_ocr_user_prompt

    def _warn_if_truncated(text, was_truncated):
        if was_truncated:
            _chunk_truncated[image_path] = _chunk_truncated.get(image_path, 0) + 1
            print(f"  Warning: {os.path.basename(image_path)} was truncated at {max_tokens} tokens. "
                  f"Consider increasing --max-output-tokens or reducing chunk height.")

    data_url = _image_data_url(image_path)
    header, encoded = data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}},
                ],
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    if streaming:
        payload["stream"] = True
        with httpx.Client(timeout=httpx.Timeout(120.0, read=120.0)) as client:
            with client.stream("POST", "https://api.anthropic.com/v1/messages", headers=headers, json=payload) as response:
                _raise_httpx_for_status_clean(response, "Anthropic")
                text, truncated = _extract_anthropic_httpx_stream_text(response)
                _warn_if_truncated(text, truncated)
                return _plain_text_to_blocks(text)

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=120,
    )
    _raise_for_status_clean(response, "Anthropic")
    text, truncated = _extract_anthropic_text(response.json())
    _warn_if_truncated(text, truncated)
    return _plain_text_to_blocks(text)


def ocr_image(
    image_path: str,
    api_key: str = "",
    credentials_path: str = "",
    model: str = "",
    base_url: str = "",
) -> list[str]:
    model = (model or _config.default_ocr_model).strip()
    base_url = (base_url or "").strip()
    if model == _config.google_cloud_vision_model:
        return _ocr_image_google_cloud(image_path, api_key, credentials_path)
    if not base_url and model.lower().startswith("lmstudio/"):
        return _ocr_image_openai_compatible(image_path, api_key or "lmstudio", model, _config.lmstudio_base_url)
    if not base_url and model.lower().startswith("ollama/"):
        return _ocr_image_openai_compatible(image_path, api_key or "ollama", model, _config.ollama_base_url)
    if not api_key:
        raise RuntimeError(f"API key is required for model {model!r}")
    if not base_url and model.lower().startswith("gemini"):
        return _ocr_image_gemini(image_path, api_key, model)
    if not base_url and model.lower().startswith("claude"):
        return _ocr_image_anthropic(image_path, api_key, model)
    return _ocr_image_openai_compatible(image_path, api_key, model, base_url)


# ---- Deduplication ----

def _similar(a: str, b: str) -> float:
    if abs(len(a) - len(b)) / max(len(a), len(b), 1) > 0.5:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def dedupe_blocks(blocks: list[str]) -> list[str]:
    result = []
    for block in blocks:
        if len(block) < _config.min_dedupe_length:
            result.append(block)
            continue
        if not any(
            len(prev) >= _config.min_dedupe_length and _similar(block, prev) >= _config.sim_threshold
            for prev in result[-5:]
        ):
            result.append(block)
    return result


def _html_blocks_to_plain_text(blocks: list[str]) -> str:
    text = "\n\n".join(blocks)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return (
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .strip()
    )


def _apply_cleanup(blocks: list[str], rules: list[dict]) -> list[str]:
    """Apply regex cleanup rules to HTML <p> blocks."""
    if not rules:
        return blocks
    active = [r for r in rules if r.get("enabled", True)]
    if not active:
        return blocks
    cleaned = []
    for block in blocks:
        inner = re.sub(r"^<p>", "", block, flags=re.IGNORECASE)
        inner = re.sub(r"</p>$", "", inner, flags=re.IGNORECASE)
        for rule in active:
            try:
                inner = re.sub(rule["pattern"], rule.get("replacement", ""), inner)
            except re.error:
                pass
        cleaned.append(f"<p>{inner}</p>")
    return cleaned


def _parse_blocks_from_html(html: str) -> list[str]:
    """Extract <p>...</p> blocks from an HTML string."""
    return re.findall(r"<p>.*?</p>", html, flags=re.DOTALL | re.IGNORECASE)


def _write_debug_chunk_outputs(
    debug_dir: str,
    image_base: str,
    chunks: list[str],
    chunk_results: dict[str, list[str]],
    copy_chunk_images: bool,
) -> None:
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    text_dir = os.path.join(debug_dir, "ocr_text")
    image_dir = os.path.join(debug_dir, "chunk_images")
    os.makedirs(text_dir, exist_ok=True)
    if copy_chunk_images:
        os.makedirs(image_dir, exist_ok=True)

    for index, chunk_path in enumerate(chunks, start=1):
        stem = f"chunk_{index:04d}"
        blocks = chunk_results.get(chunk_path, [])
        chunk_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{image_base} {stem}</title>
</head>
<body>
{"".join(blocks)}
</body>
</html>"""
        with open(os.path.join(text_dir, f"{stem}.html"), "w", encoding="utf-8") as f:
            f.write(chunk_html)
        with open(os.path.join(text_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(_html_blocks_to_plain_text(blocks))
        if copy_chunk_images and os.path.exists(chunk_path):
            shutil.copy2(chunk_path, os.path.join(image_dir, f"{stem}.png"))

    print(f"  Debug chunk OCR -> {debug_dir}")


# ---- Per-image orchestration ----

def _check_token_budget(chunk_path: str, max_tokens: int):
    """Warn if chunk pixel density suggests the token budget may be insufficient."""
    try:
        with Image.open(chunk_path) as img:
            w, h = img.size
    except Exception:
        return
    estimated = int(w * h * 0.003)
    if estimated > max_tokens * 0.8:
        print(f"  Warning: {os.path.basename(chunk_path)} ({w}x{h}px) may need ~{estimated} tokens "
              f"(budget: {max_tokens}). Consider increasing --max-output-tokens or reducing chunk height.")


def _rechunk_truncated(chunk_path: str, source_image: str, y0: int, y1: int) -> list[str]:
    """Split a truncated chunk's Y-range into 2-3 smaller sub-chunks."""
    height = y1 - y0
    sub_count = 3 if height > _config.target_chunk_height else 2
    sub_height = height // sub_count

    sub_chunks = []
    with Image.open(source_image) as img:
        for i in range(sub_count):
            sub_y0 = y0 + i * sub_height
            sub_y1 = y1 if i == sub_count - 1 else y0 + (i + 1) * sub_height
            sub_img = img.crop((0, sub_y0, img.width, sub_y1))
            sub_path = chunk_path.replace(".png", f"_sub{i:02d}.png")
            sub_img.save(sub_path, format="PNG")
            sub_chunks.append(sub_path)
            _chunk_meta[sub_path] = (source_image, sub_y0, sub_y1)
            _chunk_truncated[sub_path] = 0  # fresh retry counter for sub-chunk

    base = os.path.basename(chunk_path)
    print(f"  Re-chunking {base} ({height}px) → {len(sub_chunks)} sub-chunks "
          f"after {_config.truncation_retries} truncation(s)")
    return sub_chunks


def _process_chunk_job(job):
    chunk_path, api_key, credentials_path, model, base_url = job
    _check_token_budget(chunk_path, _config.max_output_tokens)
    return chunk_path, ocr_image(chunk_path, api_key, credentials_path, model, base_url)


def _ocr_chunks_parallel(chunks: list[str], api_key: str, credentials_path: str, model: str, base_url: str):
    results: dict[str, list[str]] = {}
    errors = []
    jobs = [(chunk, api_key, credentials_path, model, base_url) for chunk in chunks]
    max_workers = max(1, int(_config.max_workers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {}
        next_index = 0

        def submit_next():
            nonlocal next_index
            if next_index >= len(jobs) or _shutdown:
                return False
            if next_index > 0 and _config.api_call_delay_seconds > 0:
                time.sleep(_config.api_call_delay_seconds)
            job = jobs[next_index]
            pending[executor.submit(_process_chunk_job, job)] = job[0]
            next_index += 1
            return True

        while len(pending) < max_workers and submit_next():
            pass

        while pending:
            for future in as_completed(list(pending.keys()), timeout=None):
                chunk_path = pending.pop(future)
                try:
                    done_chunk, blocks = future.result()
                    results[done_chunk] = blocks
                    char_count = len(_html_blocks_to_plain_text(blocks))
                    print(f"  OK {os.path.basename(done_chunk)} ({char_count} chars)")
                except Exception as exc:
                    errors.append((chunk_path, exc))
                    print(f"  FAIL {os.path.basename(chunk_path)}: {exc}")

                if _shutdown:
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("  Cancelled remaining chunks due to shutdown.")
                    return results, errors

                while len(pending) < max_workers and submit_next():
                    pass
                break

    # Auto re-chunk: collect sub-chunks for any chunk truncated past the retry threshold
    all_sub_chunks = []
    for chunk_path in chunks:
        retries = _chunk_truncated.get(chunk_path, 0)
        if retries >= _config.truncation_retries and chunk_path in _chunk_meta:
            source_img, y0, y1 = _chunk_meta[chunk_path]
            all_sub_chunks.extend(_rechunk_truncated(chunk_path, source_img, y0, y1))

    if all_sub_chunks:
        sub_results, sub_errors = _ocr_chunks_parallel(
            all_sub_chunks, api_key, credentials_path, model, base_url
        )
        # Replace truncated originals with sub-chunk results
        for chunk_path in chunks:
            if _chunk_truncated.get(chunk_path, 0) >= _config.truncation_retries and chunk_path in results:
                del results[chunk_path]
        results.update(sub_results)
        errors.extend(sub_errors)

    return results, errors


def _load_progress(results_dir: str) -> set[str]:
    path = os.path.join(results_dir, "progress.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return set(json.load(f).get("completed", []))
        except (json.JSONDecodeError, KeyError, OSError):
            return set()
    return set()


def _save_progress(results_dir: str, image_base: str):
    path = os.path.join(results_dir, "progress.json")
    completed = _load_progress(results_dir)
    completed.add(image_base)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed)}, f, ensure_ascii=False, indent=2)


def _blocks_to_json_text(blocks: list[str], image_base: str, model: str) -> str:
    return json.dumps({
        "image": image_base,
        "model": model,
        "blocks": _html_blocks_to_plain_text(blocks),
    }, ensure_ascii=False, indent=2)


def process_image(
    image_path: str,
    output_dir: str,
    api_key: str = "",
    credentials_path: str = "",
    model: str = "",
    base_url: str = "",
    job_name: str = "",
) -> None:
    base = os.path.splitext(os.path.basename(image_path))[0]
    _reset_chunk_state()
    print(f"\n-> {base}")

    chunk_dir = _image_chunks_dir(output_dir, job_name, base) if job_name else os.path.join(output_dir, f".{base}_chunks")
    results_dir = _results_dir(output_dir, job_name) if job_name else os.path.join(output_dir, _config.ocr_results_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    if _config.resume_processing and base in _load_progress(results_dir):
        print(f"  Skipping -- already completed (use --no-resume to reprocess)")
        return

    chunks = None
    used_existing = False
    splitting = False

    if _config.use_existing_chunks and job_name and os.path.isdir(chunk_dir):
        existing = sorted(f for f in os.listdir(chunk_dir) if f.lower().endswith(".png"))
        if existing:
            chunks = [os.path.join(chunk_dir, f) for f in existing]
            print(f"  Using {len(chunks)} existing chunk(s) from {chunk_dir}")
            used_existing = True

    if chunks is None:
        if should_split(image_path):
            splitting = True
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
            os.makedirs(chunk_dir, exist_ok=True)
            with Image.open(image_path) as img:
                iw, ih = img.size
            print(f"  Splitting {iw}x{ih}px image")
            chunk_info = split_image(image_path, chunk_dir)
            # Populate re-chunk metadata
            for fname, y0, y1 in chunk_info:
                _chunk_meta[fname] = (image_path, y0, y1)
            chunks = [info[0] for info in chunk_info]
        else:
            chunks = [image_path]

    if _config.dry_run:
        model_label = model or _config.default_ocr_model
        chunk_sizes = []
        for c in chunks:
            try:
                with Image.open(c) as ci:
                    chunk_sizes.append(f"{ci.size[0]}x{ci.size[1]}")
            except Exception:
                chunk_sizes.append("?")
        print(f"  [DRY RUN] {len(chunks)} chunk(s) → {model_label}")
        for i, (c, s) in enumerate(zip(chunks, chunk_sizes)):
            print(f"    chunk {i+1:04d}  {s}  {os.path.basename(c)}")
        request_count = max(1, int(_config.max_workers))
        estimated = len(chunks) + (len(chunks) - 1) * max(0, int(_config.api_call_delay_seconds)) // request_count
        print(f"  [DRY RUN] ~{len(chunks)} API calls, {_config.max_workers} workers, {_config.api_call_delay_seconds}s delay")
        print(f"  [DRY RUN] Output: {os.path.join(results_dir, base + '.html')}")
        if _config.cleanup and splitting and not used_existing:
            _safe_cleanup(chunk_dir, chunks)
        return

    chunk_results, errors = _ocr_chunks_parallel(chunks, api_key, credentials_path, model, base_url)

    if errors:
        print(f"  [{len(errors)}/{len(chunks)} chunks failed]")
        if len(errors) == len(chunks):
            print(f"  All chunks failed; skipping {base}")
            if splitting and not used_existing:
                _safe_cleanup(chunk_dir, chunks)
            return

    debug_dir = os.path.join(results_dir, "debug_chunks", base)
    _write_debug_chunk_outputs(debug_dir, base, chunks, chunk_results, used_existing or splitting)

    all_blocks = []
    for chunk in chunks:
        all_blocks.extend(chunk_results.get(chunk, []))

    deduped = dedupe_blocks(all_blocks) if _config.enable_dedupe else all_blocks
    deduped = _apply_cleanup(deduped, _config.cleanup_rules)

    fmt = _config.output_format.lower()
    if fmt == "json":
        content = _blocks_to_json_text(deduped, base, model)
        ext = ".json"
    elif fmt == "md":
        content = _html_blocks_to_plain_text(deduped)
        ext = ".md"
    elif fmt == "txt":
        content = _html_blocks_to_plain_text(deduped)
        ext = ".txt"
    else:
        content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{base}</title>
</head>
<body>
{"".join(deduped)}
</body>
</html>"""
        ext = ".html"

    out_path = os.path.join(results_dir, f"{base}{ext}")
    encoding = "utf-8"
    with open(out_path, "w", encoding=encoding) as f:
        f.write(content)

    print(f"  Saved -> {out_path} ({len(deduped)} blocks)")

    if _config.resume_processing:
        _save_progress(results_dir, base)

    if _config.cleanup and splitting and not used_existing:
        _safe_cleanup(chunk_dir, chunks)


def process_image_chunk_only(image_path: str, output_dir: str, job_name: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)
    chunk_image_only(image_path, output_dir, job_name)


def _safe_cleanup(temp_dir: str, chunk_files: list[str]) -> None:
    for path in chunk_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"  Warning: could not remove {path}: {e}")
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except OSError as e:
        print(f"  Warning: could not remove temp dir {temp_dir}: {e}")


def collect_images(input_dir: str) -> list[str]:
    return [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(_config.image_exts)
    ]


# ---- EPUB export ----

def _export_epub(results_dir: str, job_name: str) -> str:
    """Build an EPUB from all .html files in results_dir. Returns path to .epub."""
    from ebooklib import epub

    html_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".html"))
    if not html_files:
        raise ValueError(f"No HTML files found in {results_dir}")

    book = epub.EpubBook()
    book.set_title(job_name)
    book.set_language("en")
    book.add_author("ocrrr")

    chapters = []
    spine = ["nav"]
    for html_file in html_files:
        with open(os.path.join(results_dir, html_file), encoding="utf-8") as f:
            body = f.read()
        # Extract body content
        body = re.sub(r".*<body[^>]*>", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"</body>.*", "", body, flags=re.DOTALL | re.IGNORECASE)
        chapter = epub.EpubHtml(
            title=os.path.splitext(html_file)[0],
            file_name=f"chapters/{html_file}",
            lang="en",
        )
        chapter.content = f"<h1>{chapter.title}</h1>\n{body}"
        book.add_item(chapter)
        chapters.append(chapter)
        spine.append(chapter)

    book.toc = chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine

    epub_path = os.path.join(results_dir, f"{job_name}.epub")
    epub.write_epub(epub_path, book)
    return epub_path


# ---- Entry point ----

def main():
    parser = argparse.ArgumentParser(
        description="ocrrr — batch OCR images using Google Cloud Vision, Gemini, OpenAI, Claude, LM Studio, or Ollama"
    )
    parser.add_argument("input_dir", help="Directory containing source images")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for HTML output (default: folder containing this script)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for Gemini/OpenAI-compatible/Anthropic models, or Google Cloud Vision API key",
    )
    parser.add_argument(
        "--credentials-json",
        default=None,
        help="Google service-account JSON key file (or set GOOGLE_APPLICATION_CREDENTIALS env var)",
    )
    parser.add_argument("--model", default=_config.default_ocr_model, help=f"OCR model name. Use {_config.google_cloud_vision_model!r} for Google Cloud Vision.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL. Blank = auto route.")
    parser.add_argument("--lmstudio-url", default=_config.lmstudio_base_url, help="LM Studio base URL (default: http://localhost:1234/v1).")
    parser.add_argument("--ollama-url", default=_config.ollama_base_url, help="Ollama base URL (default: http://localhost:11434/v1).")
    parser.add_argument("--prompt", default="", help="System prompt override. Blank = use built-in default.")
    parser.add_argument("--user-prompt", default="", help="User prompt override. Blank = use built-in default.")
    parser.add_argument("--format", default=_config.output_format, choices=("html", "txt", "md", "json"), help="Output format (default: html).")
    parser.add_argument("--batch-size", type=int, default=_config.max_workers, help="Parallel OCR request slots.")
    parser.add_argument("--api-call-delay", type=float, default=_config.api_call_delay_seconds, help="Delay before filling the next request slot.")
    parser.add_argument("--temperature", type=float, default=_config.temperature, help="Model temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=_config.max_output_tokens, help="Max generated OCR tokens per chunk/request.")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming for API OCR calls.")
    parser.add_argument("--thinking", action="store_true", help="Send thinking/reasoning parameters when supported by the selected route.")
    parser.add_argument(
        "--gemini-thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=_config.gemini_thinking_level,
        help="Gemini thinking level when --thinking is enabled.",
    )
    parser.add_argument(
        "--openai-thinking-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=_config.openai_thinking_effort,
        help="OpenAI-compatible reasoning effort when --thinking is enabled.",
    )
    parser.add_argument(
        "--request-lock-microseconds",
        type=int,
        default=_config.request_prep_lock_microseconds,
        help="Tiny request-prep lock duration in microseconds; does not lock network calls.",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only split images into chunks; do not call an OCR API.",
    )
    parser.add_argument(
        "--skip-pinyin",
        action="store_true",
        help="Skip romanized pinyin/pronunciation-only OCR lines in the HTML output.",
    )
    parser.add_argument(
        "--skip-romanization",
        action="store_true",
        help="Skip broader Latin-only romanization/pronunciation lines in the HTML output.",
    )
    parser.add_argument(
        "--cut-dedupe",
        action="store_true",
        help="Enable fuzzy dedupe for duplicated OCR text caused by chunk/cut overlap.",
    )
    parser.add_argument(
        "--use-chunks",
        action="store_true",
        help="Use existing chunk PNGs from a previous --chunks-only run instead of re-splitting.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all images even if output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunk plan and estimated API calls without sending any requests.",
    )
    parser.add_argument(
        "--truncation-retries",
        type=int,
        default=_config.truncation_retries,
        help="Retry count before auto re-chunking a truncated chunk (default: 2).",
    )
    parser.add_argument("--replace", action="append", nargs=2, metavar=("PATTERN", "REPLACEMENT"),
                        default=[], help="Regex replace rule (repeatable). Applied after OCR.")
    parser.add_argument("--cleanup-file", default="", help="Load cleanup rules from a JSON file.")
    parser.add_argument("--cleanup-only", action="store_true",
                        help="Skip OCR — apply cleanup rules to existing .html files.")
    parser.add_argument("--backup", action="store_true",
                        help="With --cleanup-only, save originals as *_backup.html.")
    parser.add_argument(
        "--epub",
        action="store_true",
        help="Export EPUB after processing all images.",
    )
    args = parser.parse_args()
    _reset_chunk_state()

    # Populate config from CLI args
    _config.skip_pinyin_lines = bool(args.skip_pinyin)
    _config.skip_romanization_lines = bool(args.skip_romanization)
    _config.enable_dedupe = bool(args.cut_dedupe)
    _config.use_existing_chunks = bool(args.use_chunks)
    _config.resume_processing = not bool(args.no_resume)
    _config.dry_run = bool(args.dry_run)
    _config.truncation_retries = max(0, int(args.truncation_retries))
    _config.export_epub = bool(args.epub)
    _config.cleanup_only = bool(args.cleanup_only)
    _config.output_format = args.format

    # Build cleanup rules from CLI args
    cleanup_rules = []
    if args.cleanup_file:
        try:
            with open(args.cleanup_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                cleanup_rules = data
            elif isinstance(data, dict):
                # {"replacements": [...], ...} format
                for pat, rep in data.get("replacements", []):
                    cleanup_rules.append({"pattern": pat, "replacement": rep, "enabled": True})
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load cleanup file: {e}")
    for pattern, replacement in args.replace:
        cleanup_rules.append({"pattern": pattern, "replacement": replacement, "enabled": True})
    _config.cleanup_rules = cleanup_rules
    _config.ollama_base_url = args.ollama_url
    _config.max_workers = max(1, int(args.batch_size))
    _config.api_call_delay_seconds = max(0.0, float(args.api_call_delay))
    _config.temperature = float(args.temperature)
    _config.max_output_tokens = max(1, int(args.max_output_tokens))
    _config.streaming = not bool(args.no_streaming)
    _config.request_prep_lock_microseconds = max(0, int(args.request_lock_microseconds))
    _config.send_thinking_parameters = bool(args.thinking)
    _config.gemini_thinking_level = args.gemini_thinking_level
    _config.openai_thinking_effort = args.openai_thinking_effort
    _config.lmstudio_base_url = args.lmstudio_url
    if args.prompt:
        _config.vision_ocr_prompt = args.prompt
    if args.user_prompt:
        _config.vision_ocr_user_prompt = args.user_prompt

    api_key = args.api_key or os.environ.get("GOOGLE_VISION_API_KEY")
    credentials_path = args.credentials_json or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if args.model != _config.google_cloud_vision_model:
        credentials_path = ""
    if not api_key and not credentials_path and not args.chunks_only and not args.dry_run:
        sys.exit(
            "Error: no API key provided. "
            "Pass --api-key YOUR_KEY, set GOOGLE_VISION_API_KEY, pass --credentials-json service-account.json, "
            "set GOOGLE_APPLICATION_CREDENTIALS, or use --chunks-only."
        )

    input_dir = args.input_dir
    output_dir = args.output_dir or _config.default_output_dir

    if not os.path.isdir(input_dir):
        sys.exit(f"Error: input_dir {input_dir!r} does not exist or is not a directory.")

    images = collect_images(input_dir)
    if not images:
        sys.exit(f"No images found in {input_dir!r} with extensions {_config.image_exts}.")

    job_name = os.path.basename(input_dir.rstrip("/\\"))
    if _config.dry_run:
        print(f"[DRY RUN MODE] No API calls will be made.")
    print(f"Found {len(images)} image(s) in {input_dir!r}")

    # Cleanup-only mode: process existing .html files
    if args.cleanup_only:
        results_dir = _results_dir(output_dir, job_name)
        if not os.path.isdir(results_dir):
            sys.exit(f"No results directory found: {results_dir}")
        cleaned_count = 0
        for image_path in images:
            base = os.path.splitext(os.path.basename(image_path))[0]
            html_path = os.path.join(results_dir, f"{base}.html")
            if not os.path.isfile(html_path):
                print(f"  No .html for {base}, skipping")
                continue
            with open(html_path, encoding="utf-8") as f:
                html = f.read()
            blocks = _parse_blocks_from_html(html)
            cleaned = _apply_cleanup(blocks, cleanup_rules)
            if args.backup:
                backup_path = html_path.replace(".html", "_backup.html")
                shutil.copy2(html_path, backup_path)
                print(f"  Backup -> {backup_path}")
            new_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{base}</title>
</head>
<body>
{"".join(cleaned)}
</body>
</html>"""
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(new_html)
            print(f"  Cleaned -> {html_path} ({len(cleaned)} blocks)")
            cleaned_count += 1
        print(f"\nCleaned {cleaned_count} file(s).")
        return

    for image_path in images:
        if _shutdown:
            print("\nShutdown requested; stopping before next image.")
            break
        if args.chunks_only:
            process_image_chunk_only(image_path, output_dir, job_name)
        else:
            process_image(image_path, output_dir, api_key, credentials_path, args.model, args.base_url, job_name=job_name)

    if _config.export_epub and not args.chunks_only and not _shutdown:
        results_dir = _results_dir(output_dir, job_name)
        if os.path.isdir(results_dir):
            html_files = [f for f in os.listdir(results_dir) if f.endswith(".html")]
            if html_files:
                epub_path = _export_epub(results_dir, job_name)
                print(f"\nExported EPUB -> {epub_path} ({len(html_files)} chapters)")

    print("\nExited early due to interrupt." if _shutdown else "\nDone.")


if __name__ == "__main__":
    main()
