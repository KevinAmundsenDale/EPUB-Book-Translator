import json
import os
import re
import shutil
import time
import zipfile
import traceback
from pprint import pprint
from pathlib import Path
from itertools import zip_longest

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm




# =========================
# Configuration
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MAX_RETRIES = 10   
OLLAMA_RETRY_DELAY = 5  

TRANSLATE_MODEL = "translategemma:12b"
POLISH_MODEL = "qwen3:14b"

ORIGINAL_LANGUAGE = "English"
TRANSLATE_TO_LANGUAGE = "Norwegian Bokmål"
BOOK_GENRE = "Fantasy"

INPUT_EPUB = "input.epub"
OUTPUT_EPUB = "output.epub"  #TODO should be named after the original epub but with _{TRANSLATE_TO_LANGUAGE} appended
GLOSSARY_FILE = "glossary.txt"

TEMP_DIR = "_epub_work"
PROGRESS_FILE_NAME = ".progress.json"

TRANSLATE_OPTIONS = {
    "temperature": 0.15,
}
POLISH_OPTIONS = {
    "temperature": 0.35,
}

MANUAL_RESUME_FROM_FILE = None

# If True and _epub_work exists without progress-file, keep folder and create new progress from there. If False, epub is unpacked again.
REUSE_EXISTING_TEMP_IF_NO_PROGRESS = True

MANUAL_CORRECTION_IF_WRONG = True

BLOCK_TAGS = {
    "p", "li", "dd", "dt", "figcaption", "caption",
    "blockquote"
}

SKIP_FILE_PATTERNS = (
    "toc", "nav", "contents", "titlepage", "copyright", "imprint"
)


# =========================
# Helper Functions
# =========================


def load_glossary(path: str) -> str:

    if os.path.exists(path):
        return Path(path).read_text(encoding="utf-8").strip()
    
    return ""


def progress_path(temp_dir: Path) -> Path:
    return temp_dir/PROGRESS_FILE_NAME


def load_progress(temp_dir: Path) -> dict | None:

    path = progress_path(temp_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(temp_dir: Path, progress: dict):

    path = progress_path(temp_dir)
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def text_file_finder(name: str) -> bool:

    lower = name.lower()

    return lower.endswith(".xhtml") or lower.endswith(".html") or lower.endswith(".htm")


def skip_file(name: str) -> bool:

    lower = name.lower()

    return any(pat in lower for pat in SKIP_FILE_PATTERNS)


def looks_translatable(text: str) -> bool:

    text = text.strip()

    if not text:
        return False
    
    if len(re.sub(r"\W+", "", text)) < 2:
        return False
    
    return bool(re.search(r"[A-Za-z]", text))


def update_opf_language(xml_text: str) -> str:

    xml_text = re.sub(
        r"(<dc:language[^>]*>)(.*?)(</dc:language>)",
        r"\1nb\3",
        xml_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return xml_text


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


def extract_html_tags(value) -> list[str]:
    
    HTML_TAG_RE = re.compile(r"</?[^>]+>")

    if isinstance(value, str):
        html = value
    else:
        html = "\n".join(str(item) for item in value)

    return HTML_TAG_RE.findall(html)


# =========================
# Structuring
# =========================


def apply_manual_resume_override(progress: dict, text_files: list[Path], temp_dir: Path):

    if not MANUAL_RESUME_FROM_FILE:
        return progress
    
    rel_files = [str(p.relative_to(temp_dir)).replace("\\", "/") for p in text_files]
    rel_files.sort()
    
    matches = [
        f for f in rel_files
        if f == MANUAL_RESUME_FROM_FILE or f.endswith("/" + MANUAL_RESUME_FROM_FILE)
    ]

    if not matches:
        raise ValueError(f"MANUAL_RESUME_FROM_FILE='{MANUAL_RESUME_FROM_FILE}' not found in this EPUB-structure.")

    target = matches[0]

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

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    text_files = [p for p in all_files if text_file_finder(p.name)]

    if prog is None:
        prog = build_initial_progress(text_files, temp_dir)

    # Apply manual override even if progress already exists
    prog = apply_manual_resume_override(prog, text_files, temp_dir)

    save_progress(temp_dir, prog)
    return prog


def is_mismatched_tags(original: str, translated: str):
    orig_tags = extract_html_tags(str(original))
    trans_tags = extract_html_tags(str(translated))

    if orig_tags != trans_tags:
        return True
    return False


def ensure_correctness(original: list, translated: list) -> list:
    mismatched = []

    for i, (orig_item, trans_item) in enumerate(zip_longest(original, translated, fillvalue=None)):
        if orig_item is None or trans_item is None:
            mismatched.append((i, orig_item, trans_item))
            continue
        
        if is_mismatched_tags(str(orig_item), str(trans_item)):
            mismatched.append((i, str(orig_item), str(trans_item)))

    return mismatched


def rectify_tag_difference(current_correcting: list, mode: str):

    if mode == "translation":
        original_text_language = ORIGINAL_LANGUAGE
        changed_text_language = TRANSLATE_TO_LANGUAGE
    else:
        original_text_language = TRANSLATE_TO_LANGUAGE
        changed_text_language = TRANSLATE_TO_LANGUAGE

    prompt = f"""
You are a professional web developer and editor as well as an expert in these languages: {original_text_language} and {changed_text_language}.

You are given two texts that contains a discrepancy between the html formating between the original {original_text_language} text and the {changed_text_language}. 
You are to correct the {changed_text_language} texts difference in html tags from the original {original_text_language} text. 
Preserve the text, only change the HTML code/tags. Your only job is to correct the formatting such that the translated text has the same structure as the original.

CRITICAL HTML RULES:
- Treat every HTML tag as immutable markup.
- An HTML tag is any text starting with < and ending with >.
- Correct all HTML tags from the original exactly as they appear.
- Keep the exact same HTML tags in the exact same order.

Output rules:
- Output only the translated HTML fragment.
- Do not explain anything.
- Do not add Markdown.
- Do not wrap the answer in triple backticks.

Keep the HTML tags exactly as they appear in the input. Do not move them.
Output only the translated HTML fragment, nothing else.

original paragraph:
{current_correcting[1]}
Paragraph to correct:
{current_correcting[2]}
""".strip()

    model_change_correction = ollama_generate(
    model=POLISH_MODEL,
    prompt=prompt,
    options=POLISH_OPTIONS)

    return model_change_correction

# =========================
# Translation Core
# =========================


def check_and_redo(
    source_texts: list[str],
    changed_texts: list[str],
    file_name: str,
    glossary: str,
    mode: str,
    max_attempts: int = 3,
    ) -> list[str]:
    
    """Retry paragraphs whose HTML-tag sequence changed."""
    if len(source_texts) != len(changed_texts):
        raise RuntimeError(
            f"Paragraph count mismatch before {mode} validation in {file_name}: "
            f"{len(source_texts)} source paragraphs vs {len(changed_texts)} results"
        )

    mode = mode.lower()
    if mode not in {"translation", "polish"}:
        raise ValueError(f"Unknown correction mode: {mode!r}")

    corrected = list(changed_texts)
    mismatches = ensure_correctness(source_texts, corrected)

    for index, source_item, changed_item in tqdm(mismatches, total=len(mismatches), desc=f"Fixing html tag mistakes {Path(file_name).name}", leave=False):
        if source_item is None or changed_item is None:
            raise RuntimeError(
                f"Missing paragraph at index {index} during {mode} validation "
                f"in {file_name}"
            )

        source_html = str(source_item)
        current_html = str(changed_item)

        for attempt in range(1, max_attempts + 1):
            if not is_mismatched_tags(source_html, current_html):
                break

            if mode == "translation":
                current_html = translate_batch(
                    source_html,
                    glossary,
                    file_name=file_name,
                    batch_index=index + 1,
                )
            else:
                current_html = polish_batch(
                    source_html,
                    glossary,
                    file_name=file_name,
                    batch_index=index + 1,
                )

            time.sleep(0.1)

        if is_mismatched_tags(source_html, current_html):
            current_html = rectify_tag_difference(
                (index, source_html, current_html)
            )

        if is_mismatched_tags(source_html, current_html) and MANUAL_CORRECTION_IF_WRONG:

            print("This is the original text: \n \n" f"{source_html}\n")
            print("This is the wrongly formated text: \n \n" f"{current_html}\n")
            print("Correctly write how you want it. Remember to be exact on the HTML. If you want to stop write EXIT, or write KEEP to keep the text \n")
            correction = input()

            if is_mismatched_tags(source_html, correction) and correction != "EXIT" and correction != "KEEP":

                print("\n The HTML was not written correctly. Please try again or write EXIT to stop or write KEEP to keep the new text you wrote \n")

                prev_try = correction

                while is_mismatched_tags(source_html, correction) and correction != "KEEP":

                    if correction == "EXIT":
                        raise RuntimeError(f"Could not restore matching HTML tags for paragraph " f"{index + 1} in {file_name} during {mode}")
                    
                    if correction == "KEEP":
                        corrected[index] = prev_try
                        break 

                    else: 
                        prev_try = correction
                        correction = input()     

            elif correction == "EXIT":

                raise RuntimeError(f"Could not restore matching HTML tags for paragraph " f"{index + 1} in {file_name} during {mode}")
            
            elif correction == "KEEP":

                corrected[index] = current_html

            else:

                corrected[index] = correction

        elif is_mismatched_tags(source_html, current_html) and not MANUAL_CORRECTION_IF_WRONG:

            raise RuntimeError(f"Could not restore matching HTML tags for paragraph " f"{index + 1} in {file_name} during {mode}")

    return corrected


def ollama_generate(model: str, prompt: str, options: dict | None = None, think: bool | None = None) -> str:

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
    }

    if options:
        payload["options"] = options

    if think is not None:
        payload["think"] = think

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


def translate_batch(paragraphs: object, glossary: str, file_name: str | None = None, batch_index: int | None = None) -> str:

    text = str(paragraphs)

    prompt = f"""
You are a professional literary translator from {ORIGINAL_LANGUAGE} to {TRANSLATE_TO_LANGUAGE}.

Translate the visible text from {ORIGINAL_LANGUAGE} to natural {TRANSLATE_TO_LANGUAGE}.
Preserve meaning, tone, atmosphere, and character voice.
Genre: {BOOK_GENRE} fiction.

CRITICAL HTML RULES:
- Treat every HTML tag as immutable markup.
- An HTML tag is any text starting with < and ending with >.
- Copy all HTML tags from the input exactly as they appear.
- Keep the exact same HTML tags in the exact same order.
- Do not create, delete, rename, escape, reorder, split, merge, or move any HTML tag.
- Do not add italics, spans, emphasis, or any other markup.
- Glossary terms do not require italics or markup.
- Only translate visible human-readable text outside HTML tags.
- If text is already inside an HTML tag pair, translate only that text and keep the surrounding tags.
- If a single emphasized English word is inside a tag, place the same existing tags around only the translated equivalent of that word.
- Never add HTML tags around names, glossary terms, protected terms, or important words unless those tags already exist in the input.
- Existing HTML emphasis must be preserved only where it already exists. Do not decide that any new word should be emphasized.

Output rules:
- Output only the translated HTML fragment.
- Do not explain anything.
- Do not add Markdown.
- Do not wrap the answer in triple backticks.
- Do not output in code format.

Glossary / terminology rules:
{glossary if glossary else "(none)"}

Use natural {TRANSLATE_TO_LANGUAGE} prose.
Do not modernize the tone.
Keep the HTML tags exactly as they appear in the input. Do not move them.
Output only the translated HTML fragment.

Paragraph to translate:
{text}
""".strip()

    raw = ollama_generate(
        model=TRANSLATE_MODEL,
        prompt=prompt,
        options=TRANSLATE_OPTIONS,
    )
    return raw


def polish_batch(paragraphs: object, glossary: str, file_name: str | None = None, batch_index: int | None = None) -> str:

    text = str(paragraphs)

    prompt = f"""
You are a professional {TRANSLATE_TO_LANGUAGE} literary editor.

Polish the visible prose so it sounds natural, fluent, and idiomatic.
Preserve the exact meaning, content, tone, names, places, and protected terms.
Do not summarize, add, remove, reinterpret, or modernize anything.
Maintain a literary {BOOK_GENRE} tone.

HTML RULES:
- Treat all text beginning with < and ending with > as immutable HTML.
- Copy every HTML tag exactly.
- Keep the same tags in the same order and around the same local text.
- Do not add, remove, rename, escape, reorder, split, merge, or move tags.
- Do not add new emphasis or markup.
- Polish only visible text.

OUTPUT:
- Output only the polished HTML fragment.
- Do not output explanations, Markdown, or code fences.

Protected terminology:
{glossary if glossary else "(none)"}

Paragraph to polish:
{text}
""".strip()

    raw = ollama_generate(
        model=POLISH_MODEL,
        prompt=prompt,
        options=POLISH_OPTIONS,
        think=False,
    )
    return raw


def parse_replacement_block(new_html: str, expected_tag: str) -> Tag:
    """Parse one translated HTML block and return its outer tag."""
    fragment = BeautifulSoup(new_html, "html.parser")
    replacement = fragment.find(expected_tag)

    if replacement is None:
        raise RuntimeError(
            f"Model output did not contain the expected <{expected_tag}> block"
        )

    return replacement


# =========================
# Core
# =========================


def process_html_content(html_text: str, glossary: str, file_name: str) -> str:
    soup = BeautifulSoup(html_text, "lxml-xml")

    if soup.find() is None:
        soup = BeautifulSoup(html_text, "lxml")

    blocks = extract_translatable_blocks(soup)

    if not blocks:
        return html_text

    original_texts = [str(block) for block in blocks]

    temporary_translated: list[str] = []

    for batch_idx, original_html in tqdm(enumerate(original_texts, start=1), total=len(original_texts), desc=f"Translating {Path(file_name).name}", leave=False):
        translated = translate_batch(
            original_html,
            glossary,
            file_name=file_name,
            batch_index=batch_idx,
        )
        temporary_translated.append(translated)
        time.sleep(0.1)

    temporary_translated = check_and_redo(
        original_texts,
        temporary_translated,
        file_name,
        glossary,
        "translation",
    )

    polished_results: list[str] = []

    for batch_idx, translated_html in tqdm(enumerate(temporary_translated, start=1), total=len(temporary_translated), desc=f"Polishing {Path(file_name).name}", leave=False):
        polished = polish_batch(
            translated_html,
            glossary,
            file_name=file_name,
            batch_index=batch_idx,
        )
        polished_results.append(polished)
        time.sleep(0.1)

    polished_results = check_and_redo(
        temporary_translated,
        polished_results,
        file_name,
        glossary,
        "polish",
    )

    if len(polished_results) != len(blocks):
        raise RuntimeError(
            f"Paragraph count mismatch in {file_name}: "
            f"{len(blocks)} blocks vs {len(polished_results)} results"
        )

    for block, new_html in zip(blocks, polished_results):
        replacement = parse_replacement_block(new_html, block.name)
        block.replace_with(replacement)

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
    text_files = [p for p in all_files if text_file_finder(p.name)]
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
            
            if skip_file(rel):

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
                traceback.print_exc()
                failed_files.add(rel)
                progress["failed_files"] = sorted(failed_files)
                progress["current_file"] = None
                save_progress(temp_dir, progress)

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


# =========================
# Main Function
# =========================


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
