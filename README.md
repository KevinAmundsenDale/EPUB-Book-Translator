# EPUB Book Translator

This is an Epub book translator. It translates an epub file from one language to another using locally installed AI models. It uses two AI models for translation; a translator and a polisher/editor. The models used are currently set to `translategemma 12b` and `qwen3 14b`.

## Specifications

For common things that one might want to change I have included at the top of the code some things that can be modified for your specific use case and needs. These currently include:

| Specifications / Configurations                   |  Variable name in code   |     Default value     |                                                                                                                                                                                                                  Description |
| :------------------------------------------------ | :----------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Original language                                 |    ORIGINAL_LANGUAGE     |        English        |                                                                                                                                                                                       The original language of the epub file |
| Preferred translated language                     |  TRANSLATE_TO_LANGUAGE   |   Norwegian Bokmål    |                                                                                                                                                                                 The language you want the book translated to |
| Book genre                                        |        BOOK_GENRE        |        Fantasy        |                                                                                                                                         Genre of your book, used for the AI model to further understand context and glossary |
| Glossary file                                     |      GLOSSARY_FILE       |     glossary.txt      |                          contains names that shouldn't be translated and words that should be translated specifically and consistently. I have attached some examples from Star Wars in the provided glossary.txt as example |
| Name of original epub                             |        INPUT_EPUB        |      input.epub       |                                                                                                                                                                      This is the name of the epub that you want to translate |
| Name of translated epub                           |       OUTPUT_EPUB        |      output.epub      |                                                                                                                                                   After the translation this is what the translated epub file will be called |
| Translator model                                  |     TRANSLATE_MODEL      |  translategemma:12b   |                                                                                                                                                     The model that is used for translating from language "A" to Language "B" |
| Polisher model                                    |       POLISH_MODEL       |       qwen3:14b       |                                                                                                                     The model used after each translated batch. It acts as an editor to improve flow, readability, and tone. |
| Max number of paragraphs translated in one prompt | MAX_PARAGRAPHS_PER_BATCH |          10           |                             The number of paragraphs that the Translator model and Polisher model takes in. Increasing this will provide better context for the models at the cost of speed and vice versa with reducing it. |
| Max number of characters translated in one prompt |   MAX_CHARS_PER_BATCH    |         6500          |                                                                                                       This serves as a safeguard for in case the size of the paragraphs is larger than what we want for our AI models to use |
| Translation model temperature                     |    TRANSLATE_OPTIONS     | {"temperature": 0.15} |                                                                      The temperature of the model that determines randomness and creativity of the translation. This is set low but higher than the model's original default |
| Polisher model temperature                        |      POLISH_OPTIONS      | {"temperature": 0.35} | The temperature of the model that determines randomness and creativity of the polishing. This number is set really low since we still want it to be a translation and not a rewrite, trying to keep the authors voice intact |
| Translate file "\*" first                         | MANUAL_RESUME_FROM_FILE  |         None          |                                                                                                                           In case a file was skipped or you particularly want a certain chapter translated before the others |

## Set up guide

To use this program, you need to have Ollama installed, along with the required AI models and Python libraries. There are also a couple of packages that are required. To start we will first install Ollama and the AI models.

### Open Terminal/Powershell

Windows:

```bash
irm https://ollama.com/install.ps1 | iex
```

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

macOS:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### download the Ollama models

Now download the models used by the script. The default setup uses `translategemma:12b` for translation and `qwen3:14b` for polishing.

Smaller models are usually faster but may give worse results. Larger models may improve quality, but they are slower and require more RAM/VRAM.

```bash
ollama pull translategemma:12b
```

```bash
ollama pull qwen3:14b
```

### download the libraries used

```bash
python -m pip install requests beautifulsoup4 lxml tqdm
```

## Usage

### Before running the script:

1. Make sure Ollama is running.
2. Place the EPUB file in the same directory as the script.
3. Rename the file to `input.epub`, or change `INPUT_EPUB` in the code.
4. Set the output file name in `OUTPUT_EPUB`.
5. Update `glossary.txt` with names, locations, terms, and words that should be translated consistently.
6. Adjust batch sizes, models, and other settings as needed.

These are the things that are most important to change:

```python
ORIGINAL_LANGUAGE = "English"
TRANSLATE_TO_LANGUAGE = "Norwegian Bokmål"
BOOK_GENRE = "Fantasy"

INPUT_EPUB = "input.epub"
OUTPUT_EPUB = "output.epub"
```

### Then run the script with:

```bash
python epub_translate.py
```

### Pausing the workflow

If you need to pause the process, press `Ctrl + C` in the terminal.  
The script will save progress, but the file currently being translated may need to be restarted.  
When run again, it will continue from the last completed HTML/XHTML file.

## Features

- Translates EPUB files locally using Ollama
- Uses two AI models for its wokflow: translation + polishing
- Saves progress to allow pause/resume
- Retries failed Ollama requests automatically
- Rebuilds the translated EPUB automatically

## Important notes

- This script is designed for EPUB files, not PDF files.
- Translation quality depends heavily on the models used.
- The workflow can be slow on weaker hardware.
- Some inline formatting inside paragraphs may be lost. Improving this is a priority in future development.
- The first run can take a long time depending on model size, hardware, and batch settings.
- You are responsible for optimizing the models for your computer/hardware a few things that can be looked into is CUDA if you have NVIDIA GPU and FlashAttention.
