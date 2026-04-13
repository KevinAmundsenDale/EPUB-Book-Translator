import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

# =========================
# Configuration
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MAX_RETRIES = 4   
OLLAMA_RETRY_DELAY = 5  

TRANSLATE_MODEL = "translategemma:12b"
POLISH_MODEL = "qwen3:14b"

ORIGINAL_LANGUAGE = "English"
TRANSLATE_TO_LANGUAGE = "Norwegian Bokmål"
BOOK_GENRE = "Fantasy"

INPUT_EPUB = "input.epub"
OUTPUT_EPUB = "output.epub"
GLOSSARY_FILE = "glossary.txt"

TEMP_DIR = "_epub_work"
PROGRESS_FILE_NAME = ".progress.json"


# Paragraph batching
MAX_PARAGRAPHS_PER_BATCH = 10
MAX_CHARS_PER_BATCH = 6500


TRANSLATE_OPTIONS = {
    "temperature": 0.15,
}
POLISH_OPTIONS = {
    "temperature": 0.35,
}

MANUAL_RESUME_FROM_FILE = None
# Example:
# MANUAL_RESUME_FROM_FILE = "OEBPS/Text/chapter12.xhtml"

# If True and _epub_work exists without progress-file, keep folder
# and create new progress from there. If False, epub is unpacked again.
REUSE_EXISTING_TEMP_IF_NO_PROGRESS = True


BLOCK_TAGS = {
    "p", "li", "dd", "dt", "figcaption", "caption",
    "blockquote", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"
}

SKIP_FILE_PATTERNS = (
    "toc", "nav", "contents", "titlepage", "copyright", "imprint"
)


# =========================
# Helpers
# =========================




def load_glossary(path: str) -> str:

    if os.path.exists(path):
        return Path(path).read_text(encoding="utf-8").strip()
    
    return ""


def is_probably_text_file(name: str) -> bool:

    lower = name.lower()

    return lower.endswith(".xhtml") or lower.endswith(".html") or lower.endswith(".htm")


def should_skip_file(name: str) -> bool:

    lower = name.lower()

    return any(pat in lower for pat in SKIP_FILE_PATTERNS)


def normalize_whitespace(text: str) -> str:

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def looks_translatable(text: str) -> bool:

    text = text.strip()

    if not text:
        return False
    
    if len(re.sub(r"\W+", "", text)) < 2:
        return False
    
    return bool(re.search(r"[A-Za-z]", text))


def split_batches(paragraphs, max_items=MAX_PARAGRAPHS_PER_BATCH, max_chars=MAX_CHARS_PER_BATCH):

    batches = []
    current = []
    current_len = 0

    for p in paragraphs:

        plen = len(p)

        if current and (len(current) >= max_items or current_len + plen > max_chars):

            batches.append(current)
            current = []
            current_len = 0

        current.append(p)
        current_len += plen

    if current:
        batches.append(current)

    return batches


def extract_translatable_blocks(soup: BeautifulSoup):

    blocks = []

    for tag in soup.find_all(BLOCK_TAGS):

        if tag.find_parent(["script", "style", "svg", "math"]):
            continue

        text = tag.get_text(" ", strip=True)

        if not looks_translatable(text):
            continue

        blocks.append(tag)

    return blocks


def replace_block_text_preserving_inline_markup(tag, translated_text: str):

    tag.clear()
    tag.append(NavigableString(translated_text))


def update_opf_language(xml_text: str) -> str:

    xml_text = re.sub(
        r"(<dc:language[^>]*>)(.*?)(</dc:language>)",
        r"\1nb\3",
        xml_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return xml_text


def make_schema(n: int) -> dict:

    return {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["paragraphs"],
        "additionalProperties": False,
    }


def print_json_error_context(raw: str, err: Exception, file_name: str | None = None, batch_index: int | None = None, source_paragraphs: list[str] | None = None):

    print("\n[JSON PARSE ERROR]")

    if file_name is not None:
        print(f"File: {file_name}")

    if batch_index is not None:
        print(f"Batch: {batch_index}")

    print(f"Error: {err}")

    debug_path = Path("_json_error_dump.txt")
    debug_path.write_text(raw, encoding="utf-8", errors="replace")

    print(f"Full raw response written to: {debug_path}")

    pos = getattr(err, "pos", None)

    if pos is not None and 0 <= pos < len(raw):

        symbol = raw[pos]
        start = max(0, pos - 80)
        end = min(len(raw), pos + 80)

        snippet = raw[start:end]
        pointer = " " * (pos - start) + "^"

        print(f"Position: {pos}")
        print(f"Symbol: {repr(symbol)}")
        print("Context around error:")
        print(snippet)
        print(pointer)

    else:
        print("Could not determine exact error position in model response.")

    if source_paragraphs:

        print("\nOriginal batch sent to model:")

        for i, p in enumerate(source_paragraphs, start=1):
            print(f"[{i}] {p}")

    print("-" * 80)


def parse_json_response(raw: str, expected_n: int, file_name: str | None = None, batch_index: int | None = None, source_paragraphs: list[str] | None = None) -> list[str]:

    raw = raw.strip()

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as e1:

        cleaned = re.sub(r"^```json\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as e2:
            print_json_error_context(
                raw=cleaned,
                err=e2,
                file_name=file_name,
                batch_index=batch_index,
                source_paragraphs=source_paragraphs,
            )
            raise

    paragraphs = data.get("paragraphs")

    if not isinstance(paragraphs, list):
        raise ValueError("JSON response missing 'paragraphs' list")
    
    if len(paragraphs) != expected_n:
        raise ValueError(f"Expected {expected_n} paragraphs, got {len(paragraphs)}")

    return [normalize_whitespace(str(x)) for x in paragraphs]




# =========================
# Ollama translation
# =========================




def ollama_generate(model: str, prompt: str, schema: dict | None = None, options: dict | None = None) -> str:

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if schema is not None:
        payload["format"] = schema

    if options:
        payload["options"] = options

    last_err = None

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):

        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=900)
            r.raise_for_status()
            data = r.json()
            return data["response"]

        except RequestException as e:
            last_err = e
            print(f"[WARN] Ollama connection failed (attempt {attempt}/{OLLAMA_MAX_RETRIES}): {e}")
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)

    raise last_err


def translate_batch(paragraphs: list[str], glossary: str, file_name: str | None = None, batch_index: int | None = None) -> list[str]:

    schema = make_schema(len(paragraphs))
    schema_text = json.dumps(schema, ensure_ascii=False)
    joined = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))

    prompt = f"""
You are a professional literary translator from {ORIGINAL_LANGUAGE} to {TRANSLATE_TO_LANGUAGE}.

Translate each paragraph faithfully into natural {TRANSLATE_TO_LANGUAGE}.
Preserve meaning, tone, atmosphere, and character voice.
This is {BOOK_GENRE} fiction.

Rules:
- Return ONLY valid JSON matching the schema.
- Keep the same number of paragraphs.
- Do not merge paragraphs.
- Do not split paragraphs.
- Do not summarize or omit content.
- Keep names and protected terms unchanged unless the glossary says otherwise.
- Keep emphasis and dialogue meaning intact.

Glossary / terminology rules:
{glossary if glossary else "(none)"}
Use natural {TRANSLATE_TO_LANGUAGE} prose.
Do not modernize the tone.

JSON schema:
{schema_text}

Paragraphs to translate:
{joined}
""".strip()

    raw = ollama_generate(
        model=TRANSLATE_MODEL,
        prompt=prompt,
        schema=schema,
        options=TRANSLATE_OPTIONS,
    )
    return parse_json_response(
    raw,
    len(paragraphs),
    file_name=file_name,
    batch_index=batch_index,
    source_paragraphs=paragraphs,
    )


def polish_batch(paragraphs: list[str], glossary: str, file_name: str | None = None, batch_index: int | None = None) -> list[str]:
    schema = make_schema(len(paragraphs))
    schema_text = json.dumps(schema, ensure_ascii=False)
    joined = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))

    prompt = f"""
You are a {TRANSLATE_TO_LANGUAGE} literary editor.

Rewrite each paragraph so it sounds natural, fluent, and idiomatic in {TRANSLATE_TO_LANGUAGE} prose.
Preserve meaning exactly.
Do not add or remove content.
Do not change names, places, or protected {BOOK_GENRE} terms.
Keep the same number of paragraphs.

Rules:
- Return ONLY valid JSON matching the schema.
- Do not merge paragraphs.
- Do not split paragraphs.
- Maintain a literary {BOOK_GENRE} tone.
- Avoid overly literal wording.

Glossary / protected terminology:
{glossary if glossary else "(none)"}
Use natural {TRANSLATE_TO_LANGUAGE} prose.
Do not modernize the tone.

JSON schema:
{schema_text}

Paragraphs to polish:
{joined}
""".strip()

    raw = ollama_generate(
        model=POLISH_MODEL,
        prompt=prompt,
        schema=schema,
        options=POLISH_OPTIONS,
    )
    
    return parse_json_response(
    raw,
    len(paragraphs),
    file_name=file_name,
    batch_index=batch_index,
    source_paragraphs=paragraphs,
    )




# =========================
# Progress handling
# =========================




def progress_path(temp_dir: Path) -> Path:
    return temp_dir / PROGRESS_FILE_NAME


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_progress(temp_dir: Path) -> dict | None:

    path = progress_path(temp_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(temp_dir: Path, progress: dict):

    path = progress_path(temp_dir)
    progress["updated_at"] = now_ts()
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_manual_resume_override(progress: dict, text_files: list[Path], temp_dir: Path):

    if not MANUAL_RESUME_FROM_FILE:
        return progress

    rel_files = [str(p.relative_to(temp_dir)).replace("\\", "/") for p in text_files]
    rel_files.sort()

    matches = [f for f in rel_files if f.endswith(MANUAL_RESUME_FROM_FILE) or f == MANUAL_RESUME_FROM_FILE]

    if not matches:
        raise ValueError(f"MANUAL_RESUME_FROM_FILE='{MANUAL_RESUME_FROM_FILE}' ble ikke funnet i EPUB-strukturen.")

    target = matches[0]
    idx = rel_files.index(target)

    done_files = set(progress.get("done_files", []))

    if target in done_files:
        done_files.remove(target)

    progress["done_files"] = sorted(done_files)
    progress["current_file"] = target
    progress["status"] = "running"
    progress["manual_resume_target"] = target

    return progress


def build_initial_progress(text_files: list[Path], temp_dir: Path) -> dict:

    rel_files = [str(p.relative_to(temp_dir)).replace("\\", "/") for p in text_files]
    rel_files.sort()

    done_files = []

    if MANUAL_RESUME_FROM_FILE:

        if MANUAL_RESUME_FROM_FILE not in rel_files:

            raise ValueError(
                f"MANUAL_RESUME_FROM_FILE='{MANUAL_RESUME_FROM_FILE}' ble ikke funnet i EPUB-strukturen."
            )
        
        idx = rel_files.index(MANUAL_RESUME_FROM_FILE)
        done_files = rel_files[:idx + 1]

    return {
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "input_epub": INPUT_EPUB,
        "output_epub": OUTPUT_EPUB,
        "translate_model": TRANSLATE_MODEL,
        "polish_model": POLISH_MODEL,
        "done_files": done_files,
        "failed_files": [],
        "skipped_files": [],
        "current_file": None,
        "last_completed_file": done_files[-1] if done_files else None,
        "status": "running"
    }


def ensure_temp_and_progress(input_epub: str, temp_dir: Path):

    temp_exists = temp_dir.exists()
    prog = load_progress(temp_dir) if temp_exists else None

    if temp_exists and not REUSE_EXISTING_TEMP_IF_NO_PROGRESS and prog is None:

        shutil.rmtree(temp_dir)
        temp_exists = False

    if not temp_exists:

        temp_dir.mkdir(parents=True)
        with zipfile.ZipFile(input_epub, "r") as zin:
            zin.extractall(temp_dir)

    all_files = [p for p in temp_dir.rglob("*") if p.is_file()]
    text_files = [p for p in all_files if is_probably_text_file(p.name)]

    if prog is None:
        prog = build_initial_progress(text_files, temp_dir)

    # Apply manual override even if progress already exists
    prog = apply_manual_resume_override(prog, text_files, temp_dir)

    save_progress(temp_dir, prog)
    return prog




# =========================
# Translation core
# =========================




def process_html_content(html_text: str, glossary: str, file_name: str) -> str:

    soup = BeautifulSoup(html_text, "lxml-xml")

    if soup is None or soup.find() is None:
        soup = BeautifulSoup(html_text, "lxml")

    blocks = extract_translatable_blocks(soup)

    if not blocks:
        return html_text

    original_texts = [normalize_whitespace(tag.get_text(" ", strip=True)) for tag in blocks]

    translated_results = []
    batches = split_batches(original_texts)

    indexed_batches = list(enumerate(batches, start=1))
    temporary_translated = []

    for batch_idx, batch in tqdm(indexed_batches, desc=f"Translating {Path(file_name).name}", leave=False):

        translated = translate_batch(batch, glossary, file_name=file_name, batch_index=batch_idx)
        temporary_translated.append((batch_idx, translated))
        time.sleep(0.1)

    for batch_idx, translated_batch in tqdm(temporary_translated, desc=f"Polishing {Path(file_name).name}", leave=False):

        polished = polish_batch(translated_batch, glossary, file_name=file_name, batch_index=batch_idx)
        translated_results.extend(polished)
        time.sleep(0.1)

    

    if len(translated_results) != len(blocks):
        raise RuntimeError(f"Paragraph count mismatch in {file_name}")

    for tag, new_text in zip(blocks, translated_results):
        replace_block_text_preserving_inline_markup(tag, new_text)

    return str(soup)


def repack_epub(temp_dir: Path, output_epub: str):

    all_files = [p for p in temp_dir.rglob("*") if p.is_file()]
    mimetype_path = temp_dir / "mimetype"

    with zipfile.ZipFile(output_epub, "w") as zout:

        if mimetype_path.exists():
            zout.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)

        for path in all_files:

            rel = str(path.relative_to(temp_dir)).replace("\\", "/")

            if rel == "mimetype":
                continue

            if rel == PROGRESS_FILE_NAME:
                continue

            zout.write(path, rel, compress_type=zipfile.ZIP_DEFLATED)


def process_epub(input_epub: str, output_epub: str, glossary: str):

    temp_dir = Path(TEMP_DIR)
    progress = ensure_temp_and_progress(input_epub, temp_dir)

    all_files = [p for p in temp_dir.rglob("*") if p.is_file()]
    text_files = [p for p in all_files if is_probably_text_file(p.name)]
    opf_files = [p for p in all_files if p.suffix.lower() == ".opf"]

    text_files.sort(key=lambda p: str(p.relative_to(temp_dir)).replace("\\", "/"))
    done_files = set(progress.get("done_files", []))
    skipped_files = set(progress.get("skipped_files", []))
    failed_files = set(progress.get("failed_files", []))
    manual_target = progress.get("manual_resume_target")

    if manual_target:
        prioritized = []
        rest = []

        for p in text_files:

            rel = str(p.relative_to(temp_dir)).replace("\\", "/")

            if rel == manual_target:
                prioritized.append(p)
            else:
                rest.append(p)

        text_files = prioritized + rest

    try:
        for path in tqdm(text_files, desc="EPUB documents"):

            rel = str(path.relative_to(temp_dir)).replace("\\", "/")
            
            # skips file if it is deemed unnecessary to translate 
            if should_skip_file(rel):

                if rel not in skipped_files:

                    skipped_files.add(rel)
                    progress["skipped_files"] = sorted(skipped_files)
                    save_progress(temp_dir, progress)

                continue

            if rel in done_files:
                continue

            progress["current_file"] = rel
            progress["status"] = "running"
            save_progress(temp_dir, progress)

            try:
                try:
                    original = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    original = path.read_text(encoding="utf-8", errors="ignore")

                processed = process_html_content(original, glossary, rel)
                path.write_text(processed, encoding="utf-8")

                done_files.add(rel)
                progress["done_files"] = sorted(done_files)
                progress["last_completed_file"] = rel

                if progress.get("manual_resume_target") == rel:
                    progress["manual_resume_target"] = None

                progress["current_file"] = None

                if rel in failed_files:

                    failed_files.remove(rel)
                    progress["failed_files"] = sorted(failed_files)

                save_progress(temp_dir, progress)

            except KeyboardInterrupt:

                progress["status"] = "paused"
                progress["current_file"] = rel
                save_progress(temp_dir, progress)
                raise

            except Exception as e:

                print(f"[WARN] Skipping {rel} due to error: {e}")
                failed_files.add(rel)
                progress["failed_files"] = sorted(failed_files)
                progress["current_file"] = None
                save_progress(temp_dir, progress)

        # Update OPF only after text files are done
        for path in opf_files:

            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")

            text = update_opf_language(text)
            path.write_text(text, encoding="utf-8")

        repack_epub(temp_dir, output_epub)

        progress["status"] = "completed"
        progress["current_file"] = None
        save_progress(temp_dir, progress)

    except KeyboardInterrupt:
        print("\nPaused by user. Progress saved.")
        print(f"Resume next time by running the same script again.")
        return


def main():

    glossary = load_glossary(GLOSSARY_FILE)

    if not os.path.exists(INPUT_EPUB):
        raise FileNotFoundError(f"Could not find {INPUT_EPUB}")

    print(f"Input:  {INPUT_EPUB}")
    print(f"Output: {OUTPUT_EPUB}")
    print(f"Translate model: {TRANSLATE_MODEL}")
    print(f"Polish model:    {POLISH_MODEL}")
    print(f"Temp dir:        {TEMP_DIR}")

    process_epub(INPUT_EPUB, OUTPUT_EPUB, glossary)

    temp_dir = Path(TEMP_DIR)
    prog = load_progress(temp_dir)
    
    if prog and prog.get("status") == "completed":
        print(f"\nDone. Wrote translated EPUB to: {OUTPUT_EPUB}")


if __name__ == "__main__":
    main()