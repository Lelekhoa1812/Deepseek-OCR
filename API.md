## DeepSeek-OCR WebUI – HTTP API Usage

This Space exposes a programmatic API that mirrors the WebUI flow. You can call it using cURL or Python.

Base URL: `https://binkhoale1812-deepseekocr.hf.space`

### Python (recommended)

```python
from gradio_client import Client

client = Client("BinKhoaLe1812/DeepSeekOCR", hf_token="<HF_TOKEN>")

result = client.predict(
    image=None,                              # or a local image path/URL
    file_path="/absolute/path/to/doc.pdf",   # local PDF or image path
    mode_label="Base",                       # one of: Gundam, Tiny, Small, Base, Large
    task_label="Markdown",                   # one of: Markdown, Tables, Locate, Describe, Custom
    custom_prompt="",                        # required for Custom/Locate
    dpi_val=300,                             # PDF DPI
    page_range_text="",                      # e.g. "1-3,5"; empty = all pages
    embed=True,                              # embed detected figures into Markdown
    hiacc=False,                             # high-accuracy second pass
    sep_pages=True,                          # insert --- between pages
    api_name="/run",
)

# Save Markdown to a file
markdown = result[1]
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown or "")
print("Wrote output.md")
```

### Raw HTTP via cURL

The Gradio REST API expects a two-step process for files:
1) Upload the file to `/upload` to get a server-side path
2) Call the function route with a JSON body including that path

Replace `<HF_TOKEN>` with your token if the Space requires auth.

Step 1 – Upload the file:
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -F "files[]=@/absolute/path/to/doc.pdf" \
  https://binkhoale1812-deepseekocr.hf.space/upload
```

This returns JSON like:
```json
{
  "files": [
    {"name": "/tmp/gradio/1234/filename.pdf"}
  ]
}
```

Capture the `name` field as `SERVER_PATH`.

Step 2 – Invoke the API (`api_name` is `/run`, so route is `/run/run`):
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
        "data": [
          null,
          {"path": "SERVER_PATH"},
          "Base",
          "Markdown",
          "",
          300,
          "",
          true,
          false,
          true
        ]
      }' \
  https://binkhoale1812-deepseekocr.hf.space/run/run \
| python3 - <<'PY'
import sys, json
resp=json.loads(sys.stdin.read())
md = resp.get("data", [None, None])[1]
with open("output.md", "w", encoding="utf-8") as f:
    f.write(md or "")
print("Wrote output.md")
PY
```

The response is a JSON envelope whose `data` contains 8 outputs. The second element (`data[1]`) is the Markdown string. Both examples above save it to `output.md`. If you enabled `sep_pages`, pages are separated by `---`. The ZIP path for per-page Markdown is in `data[7]`.

### Notes
- The server performs per-page batching with retries under the hood to respect 120s GPU windows.
- For large PDFs (up to ~200 pages), prefer `mode_label="Base"` or `"Gundam"` with `dpi_val=300`.
- Use `page_range_text` to subset processing, e.g. `"1-10, 15, 20-25"`.

