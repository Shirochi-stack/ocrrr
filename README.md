# ocrrr

Batch OCR images to structured text using cloud and local vision models. Optimized for novel consoomers.

## Features

- **6 providers** — Google Cloud Vision, Gemini, OpenAI-compatible, Anthropic Claude, LM Studio, Ollama
- **Smart chunking** — line-boundary-aware image splitting with fallback overlap for tall images
- **Resume / progress persistence** — `progress.json` per job, skip completed images on restart
- **Dry run** — preview chunk plan and API call count without spending credits
- **EPUB export** — combine all chapter HTMLs into a single ebook
- **Text cleanup toolkit** — regex find/replace rules, applied inline during OCR or retroactively
- **Model comparison** — run two models side-by-side on the same image
- **Export formats** — HTML, plain text, Markdown, JSON
- **Full Qt6 GUI** — drag-and-drop, editable prompts, config profiles, custom models, progress viewer
- **CLI** — 20+ flags for automation and scripting
- **LM Studio / Ollama** — local model support with auto-routing, no API key needed

## Supported Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| Google Cloud Vision | `google-cloud-vision` | API key or service account JSON |
| Gemini | `gemini-*` (50+ models) | API key |
| OpenAI-compatible | `gpt-*`, `o3`, custom | API key + optional base URL |
| Anthropic Claude | `claude-*` (10+ models) | API key |
| LM Studio | `lmstudio/*` (5+ models) | None (local) |
| Ollama | `ollama/*` (6+ models) | None (local) |

## Installation

```bash
git clone https://codeberg.org/laotzi/ocrrr.git
cd ocrrr
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Requires Python 3.10+.

## Quick Start

### CLI

```bash
# Basic OCR with Gemini
python ocr_pipeline.py ./screenshots --api-key YOUR_KEY --model gemini-2.0-flash

# Dry run — see what will happen
python ocr_pipeline.py ./screenshots --dry-run

# Chunks only (split without OCR)
python ocr_pipeline.py ./screenshots --chunks-only

# OCR using existing chunks (skip re-splitting)
python ocr_pipeline.py ./screenshots --api-key KEY --use-chunks

# EPUB export after processing
python ocr_pipeline.py ./screenshots --api-key KEY --epub

# Text cleanup — remove page numbers
python ocr_pipeline.py ./screenshots --api-key KEY --replace "第\d+页" ""

# Retroactive cleanup on existing HTML files
python ocr_pipeline.py ./screenshots --cleanup-only --replace "第\d+页" "" --backup
```

### GUI

```bash
python gui.py
# Windows: double-click run_gui.bat
```

### Local models (LM Studio / Ollama)

```bash
# No API key needed
python ocr_pipeline.py ./screenshots --model lmstudio/default
python ocr_pipeline.py ./screenshots --model ollama/llama3.2-vision
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemini-3.1-flash-lite-preview` | OCR model name |
| `--api-key` | — | API key for cloud providers |
| `--base-url` | — | Custom OpenAI-compatible endpoint |
| `--batch-size` | `4` | Parallel API request slots |
| `--api-call-delay` | `2.0` | Seconds between request submissions |
| `--temperature` | `0.0` | Model temperature |
| `--max-output-tokens` | `8192` | Token cap per chunk |
| `--no-streaming` | — | Disable streaming responses |
| `--format` | `html` | Output: `html`, `txt`, `md`, or `json` |
| `--prompt` | *(built-in)* | System prompt override |
| `--user-prompt` | *(built-in)* | User prompt override |
| `--chunks-only` | — | Split without OCR |
| `--use-chunks` | — | Use existing chunk PNGs |
| `--dry-run` | — | Preview without API calls |
| `--no-resume` | — | Reprocess all images |
| `--epub` | — | Export EPUB after processing |
| `--replace` | — | Regex replace rule (repeatable) |
| `--cleanup-only` | — | Apply rules to existing HTML |
| `--backup` | — | Save originals during cleanup |
| `--skip-pinyin` | — | Filter pinyin-only lines |
| `--skip-romanization` | — | Filter romanization lines |
| `--cut-dedupe` | — | Dedupe overlapping chunk text |
| `--thinking` | — | Send thinking/reasoning params |
| `--truncation-retries` | `2` | Retries before auto re-chunk |

## Output Structure

```
<output_dir>/
├── OCR Chunks/
│   └── <job_name>/
│       └── <image>/
│           └── chunk_0001.png
├── OCR Results/
│   └── <job_name>/
│       ├── <image>.html
│       ├── <image>.epub
│       ├── progress.json
│       └── debug_chunks/
│           └── <image>/
│               ├── ocr_text/
│               └── chunk_images/
```

## GUI Features

- Drag-and-drop images and folders
- Editable system/user prompts with reset buttons
- Config profiles for quick preset switching
- Custom model management
- Encrypted API key storage (Fernet)
- Progress viewer per job
- Chunk preview overlay with adjustable height
- Results browser with search
- Model comparison (side-by-side)
- Text cleanup rules editor with save/load

## License

MIT
