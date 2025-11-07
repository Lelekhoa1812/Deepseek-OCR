import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import spaces
import os
import sys
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz
import re
import warnings
import numpy as np
import base64
from io import StringIO, BytesIO
import subprocess
import importlib
import time
import zipfile
import atexit
import functools
from queue import Queue
from threading import Event, Thread
from threading import Lock
from itertools import cycle
import logging

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token.*")
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
warnings.filterwarnings("ignore", message=".*get_max_cache.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*position_embeddings.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

# PaddleOCR-VL imports
try:
    from paddleocr import PaddleOCRVL
    PADDLEOCRVL_AVAILABLE = True
except ImportError:
    PADDLEOCRVL_AVAILABLE = False
    warnings.warn("PaddleOCR-VL not available. Install with: pip install 'paddleocr[doc-parser]'")

# Gemini imports (optional)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    types = None
    warnings.warn("Gemini SDK not available. Install with: pip install google-generativeai")

# Setup logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)

# Gather Gemini API keys from environment and prepare round-robin iterator
GEMINI_KEYS = [
    os.getenv("GEMINI_API_1"),
    os.getenv("GEMINI_API_2"),
    os.getenv("GEMINI_API_3"),
    os.getenv("GEMINI_API_4"),
    os.getenv("GEMINI_API_5"),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]
_gemini_cycle = cycle(GEMINI_KEYS) if GEMINI_KEYS else None
_gemini_lock = Lock()

def _get_next_gemini_key():
    if not GEMINI_AVAILABLE or not _gemini_cycle:
        return None
    with _gemini_lock:
        return next(_gemini_cycle)

# Allow overriding model via env, default to a stable flash model with fallback
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

class GeminiClient:
    """Gemini API client for generating responses"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    def generate_content(self, prompt: str, image_data: bytes = None, mime_type: str = "image/jpeg", model: str = "gemini-2.5-flash", temperature: float = 0.7) -> str:
        """Generate content using Gemini API
        
        Args:
            prompt: Text prompt
            image_data: Optional image bytes
            mime_type: MIME type of the image (default: "image/jpeg")
            model: Model name to use
            temperature: Temperature for generation
        """
        try:
            # Build parts list
            parts = [types.Part(text=prompt)]
            if image_data:
                # Use InlineData with Blob for image data
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_data
                    )
                ))
            
            # Create Content object with role and parts
            content = types.Content(role="user", parts=parts)
            
            # Generate content
            response = self.client.models.generate_content(
                model=model,
                contents=[content],
                temperature=temperature
            )
            return response.text
        except Exception as e:
            logger.error(f"[LLM] ❌ Error calling Gemini API: {e}")
            return "Error generating response from Gemini."

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Set padding side early
tokenizer.padding_side = 'right'
 
def ensure_flash_attn_if_cuda():
    # Only attempt install when CUDA is available
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module('flash_attn')
        return True
    except Exception:
        pass
    try:
        # Install without build isolation so setup can import torch
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-cache-dir', 'flash-attn==2.7.3'
        ])
        importlib.invalidate_caches()
        importlib.import_module('flash_attn')
        return True
    except Exception:
        return False
flash_ok = ensure_flash_attn_if_cuda()
try:
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation='flash_attention_2' if flash_ok else None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_safetensors=True,
    )
    if torch.cuda.is_available():
        model = model.eval().cuda()
    else:
        raise RuntimeError("CUDA not available; cannot use flash attention")
except Exception as e:
    warnings.warn(f"Flash attention/CUDA unavailable ({e}); falling back to default attention.")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

# Configure pad token after model is loaded
try:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
    # Ensure model config has pad_token_id set
    if hasattr(model, 'config'):
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        # Also set pad_token_id in generation config if it exists
        if hasattr(model.config, 'generation_config') and hasattr(model.config.generation_config, 'pad_token_id'):
            if model.config.generation_config.pad_token_id is None:
                model.config.generation_config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    pass  # Suppress warnings for pad token configuration

MODEL_CONFIGS = {
    "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False}
}

# UI labels mapped to internal keys (use plain labels to match dropdown values)
MODE_LABEL_TO_KEY = {
    "Gundam": "Gundam",
    "Tiny": "Tiny",
    "Small": "Small",
    "Base": "Base",
    "Large": "Large",
}
KEY_TO_MODE_LABEL = {v: k for k, v in MODE_LABEL_TO_KEY.items()}

# PaddleOCR-VL Configuration
# PaddleOCR-VL is a document parsing model that doesn't need language-specific configs
# It uses a single pipeline for all languages

# Defer PaddleOCR-VL pipeline init until first use to avoid startup errors on some builds
paddleocrvl_pipeline = None
PADDLEOCRVL_ERROR_MESSAGE = None

TASK_PROMPTS = {
    "Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to GitHub-flavored Markdown. Preserve headings, lists, links, code blocks, and tables.", "has_grounding": True},
    "Tables": {"prompt": "<image>\n<|grounding|>Extract ALL tables only as GitHub Markdown tables. Preserve merged cells as best as possible. Do not include non-table content.", "has_grounding": True},
    "Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "Custom": {"prompt": "", "has_grounding": False}
}

TASK_LABEL_TO_KEY = {
    "Markdown": "Markdown",
    "Tables": "Tables",
    "Locate": "Locate",
    "Describe": "Describe",
    "Custom": "Custom",
}
KEY_TO_TASK_LABEL = {v: k for k, v in TASK_LABEL_TO_KEY.items()}

# -----------------
# Simple in-memory LRU cache for per-page results
# -----------------
PAGE_CACHE = {}
PAGE_CACHE_ORDER = []
PAGE_CACHE_CAPACITY = int(os.getenv("PAGE_CACHE_CAPACITY", "512"))
PAGE_CACHE_LOCK = Lock()

def _page_cache_get(key):
    with PAGE_CACHE_LOCK:
        val = PAGE_CACHE.get(key)
        if val is not None:
            # Move to end (most recent)
            try:
                PAGE_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            PAGE_CACHE_ORDER.append(key)
        return val

def _page_cache_set(key, value):
    with PAGE_CACHE_LOCK:
        if key in PAGE_CACHE:
            PAGE_CACHE[key] = value
            try:
                PAGE_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            PAGE_CACHE_ORDER.append(key)
            return
        # Evict if needed
        while len(PAGE_CACHE_ORDER) >= PAGE_CACHE_CAPACITY:
            old_key = PAGE_CACHE_ORDER.pop(0)
            PAGE_CACHE.pop(old_key, None)
        PAGE_CACHE[key] = value
        PAGE_CACHE_ORDER.append(key)

def extract_grounding_references(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)

def draw_bounding_boxes(image, refs, extract_images=False):
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    crops = []
    
    for ref in refs:
        label = ref[1]
        coords = eval(ref[2])
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        color_a = color + (60,)
        
        for box in coords:
            x1, y1, x2, y2 = int(box[0]/999*img_w), int(box[1]/999*img_h), int(box[2]/999*img_w), int(box[3]/999*img_h)
            
            if extract_images and label == 'image':
                crops.append(image.crop((x1, y1, x2, y2)))
            
            width = 5 if label == 'title' else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)
            
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops

def clean_output(text, include_images=False, remove_labels=False):
    if not text:
        return ""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0
    
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            if remove_labels:
                text = text.replace(match[0], '', 1)
            else:
                text = text.replace(match[0], match[1], 1)
    
    return text.strip()

def embed_images(markdown, crops):
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(f'**[Figure {i + 1}]**', f'\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n', 1)
    return markdown

def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=95)
    return buf.getvalue()

def _build_gemini_system_prompt():
    return (
        "You are an expert document parser. Convert the given document image to GitHub-flavored Markdown. "
        "Preserve headings, lists, links, code blocks, and tables with correct alignment and borders. "
        "Keep the reading order, avoid hallucinations, and output Markdown only."
    )

def process_image_gemini(image: Image.Image):
    if image is None:
        return " Error Upload image", "", "", None, []
    if not GEMINI_AVAILABLE or not GEMINI_KEYS:
        return " Gemini not available or no API keys set in .env", "", "", None, []
    key = _get_next_gemini_key()
    if not key:
        return " Gemini API keys not configured", "", "", None, []

    try:
        client = GeminiClient(api_key=key)
        img_bytes = _image_to_jpeg_bytes(image)
        system_prompt = _build_gemini_system_prompt()
        
        try:
            md = client.generate_content(
                prompt=system_prompt,
                image_data=img_bytes,
                mime_type="image/jpeg",
                model=GEMINI_MODEL
            )
        except Exception:
            # Fallback for users who configured an unsupported model name
            md = client.generate_content(
                prompt=system_prompt,
                image_data=img_bytes,
                mime_type="image/jpeg",
                model="gemini-2.5-flash"
            )
        
        md = (md or "").strip()
        if not md or md == "Error generating response from Gemini.":
            return "No text" if md != "Error generating response from Gemini." else md, "", "", None, []
        return md, md, md, None, []
    except Exception as e:
        return f"Error: {str(e)}", "", "", None, []

@spaces.GPU(duration=120)
def process_image(image, mode_label, task_label, custom_prompt, embed_figures=False, high_accuracy=False):
    if image is None:
        return " Error Upload image", "", "", None, []
    if task_label in ["Custom", "Locate"] and not custom_prompt.strip():
        return "Enter prompt", "", "", None, []
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    # Normalize labels to internal keys
    mode_key = MODE_LABEL_TO_KEY.get(mode_label, mode_label)
    task_key = TASK_LABEL_TO_KEY.get(task_label, task_label)
    config = MODEL_CONFIGS[mode_key]
    
    if task_label == "Custom":
        prompt = f"<image>\n{custom_prompt.strip()}"
        has_grounding = '<|grounding|>' in custom_prompt
    elif task_label == "Locate":
        prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt = TASK_PROMPTS[task_key]["prompt"]
        has_grounding = TASK_PROMPTS[task_key]["has_grounding"]
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    out_dir = tempfile.mkdtemp()
    
    stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        model.infer(tokenizer=tokenizer, prompt=prompt, image_file=tmp.name, output_path=out_dir,
                base_size=config["base_size"], image_size=config["image_size"], crop_mode=config["crop_mode"])
    except Exception as e:
        sys.stdout = stdout
        os.unlink(tmp.name)
        shutil.rmtree(out_dir, ignore_errors=True)
        warnings.warn(f"Inference error: {e}")
        return f"Inference error: {str(e)}", "", "", None, []
    
    # Get result from stdout
    stdout_output = sys.stdout.getvalue()
    sys.stdout = stdout
    
    # Filter stdout output
    result = '\n'.join([l for l in stdout_output.split('\n') 
                        if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size', 'torch.cuda', 'loading', 'INFO:', 'WARNING:', 'ERROR:'])]).strip()
    
    # Also check output directory for markdown/text files
    if os.path.exists(out_dir):
        # Look for markdown files first
        md_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.md')])
        txt_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.txt')])
        
        # Read markdown files if available
        if md_files:
            file_contents = []
            for md_file in md_files:
                md_path = os.path.join(out_dir, md_file)
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            file_contents.append(content)
                except Exception as e:
                    warnings.warn(f"Failed to read {md_path}: {e}")
            if file_contents:
                result = '\n\n'.join(file_contents) if not result else result + '\n\n' + '\n\n'.join(file_contents)
        
        # Fallback to text files if no markdown
        elif txt_files and not result:
            file_contents = []
            for txt_file in txt_files:
                txt_path = os.path.join(out_dir, txt_file)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            file_contents.append(content)
                except Exception:
                    pass
            if file_contents:
                result = '\n\n'.join(file_contents)
    
    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    
    if not result:
        return "No text", "", "", None, []
    
    cleaned = clean_output(result, False, False)
    markdown = clean_output(result, True, True)
    
    img_out = None
    crops = []
    
    if has_grounding and '<|ref|>' in result:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)
    
    if embed_figures:
        markdown = embed_images(markdown, crops)

    # Optional second pass for high accuracy (focus on tables refinement)
    if high_accuracy and task_key in ["Markdown", "Tables"]:
        refine_prompt = "<image>\nRefine the previous extraction with emphasis on accurate table structure and alignment. Output GitHub Markdown only."
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(tmp2.name, 'JPEG', quality=95)
        tmp2.close()
        out_dir2 = tempfile.mkdtemp()
        stdout2 = sys.stdout
        sys.stdout = StringIO()
        try:
            model.infer(tokenizer=tokenizer, prompt=refine_prompt, image_file=tmp2.name, output_path=out_dir2,
                    base_size=config["base_size"], image_size=config["image_size"], crop_mode=config["crop_mode"])
        except Exception:
            pass
        
        stdout_output2 = sys.stdout.getvalue()
        sys.stdout = stdout2
        
        refine_result = '\n'.join([l for l in stdout_output2.split('\n')
                            if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size', 'torch.cuda', 'loading', 'INFO:', 'WARNING:', 'ERROR:'])]).strip()
        
        # Check output directory for refine results
        if os.path.exists(out_dir2):
            md_files2 = sorted([f for f in os.listdir(out_dir2) if f.endswith('.md')])
            if md_files2:
                for md_file in md_files2:
                    md_path2 = os.path.join(out_dir2, md_file)
                    try:
                        with open(md_path2, 'r', encoding='utf-8') as f:
                            content2 = f.read().strip()
                            if content2:
                                refine_result = content2 if not refine_result else refine_result + '\n\n' + content2
                    except Exception:
                        pass
        
        os.unlink(tmp2.name)
        shutil.rmtree(out_dir2, ignore_errors=True)
        if refine_result:
            refined_md = clean_output(refine_result, embed_figures, True)
            # Prefer refined markdown if longer (heuristic)
            if len(refined_md) > len(markdown):
                markdown = refined_md
    
    return cleaned, markdown, result, img_out, crops

def process_image_paddleocrvl(image, prompt=None):
    """Process image using PaddleOCR-VL and return results in Markdown format.
    
    Args:
        image: PIL Image to process
        prompt: Optional custom prompt. If None, uses default document parsing prompt.
    """
    if image is None:
        return " Error Upload image", "", "", None, []
    
    # Lazy init to avoid import-time errors on some environments
    global paddleocrvl_pipeline, PADDLEOCRVL_AVAILABLE, PADDLEOCRVL_ERROR_MESSAGE
    if not PADDLEOCRVL_AVAILABLE:
        msg = PADDLEOCRVL_ERROR_MESSAGE or "PaddleOCR-VL not available. Install with: pip install 'paddleocr[doc-parser]'"
        return f" {msg}", "", "", None, []
    if paddleocrvl_pipeline is None:
        try:
            paddleocrvl_pipeline = PaddleOCRVL()
        except Exception as e:
            PADDLEOCRVL_AVAILABLE = False
            PADDLEOCRVL_ERROR_MESSAGE = f"PaddleOCR-VL init failed: {e}"
            return f" {PADDLEOCRVL_ERROR_MESSAGE}", "", "", None, []
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    # Save image to temporary file for PaddleOCR
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    
    try:
        # Try different approaches based on PaddleOCR-VL API
        # First, try the simple predict method
        try:
            if prompt:
                # If custom prompt is provided, try to use it with structured messages
                # Format: PaddleOCR-VL might accept messages with image and text
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": tmp.name},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                # Try if the pipeline has a chat or generate method
                if hasattr(paddleocrvl_pipeline, 'chat'):
                    output = paddleocrvl_pipeline.chat(messages)
                elif hasattr(paddleocrvl_pipeline, 'generate'):
                    output = paddleocrvl_pipeline.generate(messages)
                else:
                    # Fallback to predict with just image
                    output = paddleocrvl_pipeline.predict(tmp.name)
            else:
                # Default: use predict with just image (document parsing mode)
                output = paddleocrvl_pipeline.predict(tmp.name)
        except (TypeError, AttributeError):
            # If structured messages don't work, fallback to simple predict
            output = paddleocrvl_pipeline.predict(tmp.name)
        
        if not output or len(output) == 0:
            os.unlink(tmp.name)
            return "No text detected", "", "", None, []
        
        # PaddleOCR-VL returns results that can be saved to markdown
        # Save to a temporary directory
        out_dir = tempfile.mkdtemp()
        markdown_results = []
        text_results = []
        raw_results = []
        
        # Handle different output types
        if isinstance(output, str):
            # Direct string output
            markdown_results.append(output)
            text_results.append(output)
            raw_results.append("PaddleOCR-VL direct output")
        elif isinstance(output, list):
            # List of results
            for res in output:
                try:
                    if isinstance(res, str):
                        # String result
                        markdown_results.append(res)
                        text_results.append(res)
                        raw_results.append("PaddleOCR-VL result")
                    elif hasattr(res, 'save_to_markdown'):
                        # Object with save_to_markdown method
                        res.save_to_markdown(save_path=out_dir)
                        # Find the markdown file
                        md_files = [f for f in os.listdir(out_dir) if f.endswith('.md')]
                        if md_files:
                            md_path = os.path.join(out_dir, md_files[0])
                            with open(md_path, 'r', encoding='utf-8') as f:
                                markdown_content = f.read()
                                markdown_results.append(markdown_content)
                                text_results.append(markdown_content)
                                raw_results.append(f"PaddleOCR-VL result: {str(res)}")
                            # Remove the file to avoid conflicts
                            os.remove(md_path)
                    else:
                        # Try to convert to string
                        markdown_results.append(str(res))
                        text_results.append(str(res))
                        raw_results.append(f"PaddleOCR-VL result: {str(res)}")
                except Exception as e:
                    warnings.warn(f"Failed to process PaddleOCR-VL result: {e}")
        else:
            # Try to convert to string
            markdown_results.append(str(output))
            text_results.append(str(output))
            raw_results.append(f"PaddleOCR-VL result: {str(output)}")
        
        # Clean up temp directory
        shutil.rmtree(out_dir, ignore_errors=True)
        os.unlink(tmp.name)
        
        if not markdown_results:
            return "No text detected", "", "", None, []
        
        # Combine results
        markdown = "\n\n".join(markdown_results)
        text = "\n\n".join(text_results)
        raw = "\n\n".join(raw_results)
        
        # For bounding boxes, we can try to extract from the result if available
        img_out = None
        try:
            # Draw bounding boxes if available in the result
            img_draw = image.copy()
            # PaddleOCR-VL may have layout information we can use
            # This is a simplified version - adjust based on actual API
            img_out = img_draw
        except Exception as e:
            warnings.warn(f"Failed to draw bounding boxes: {e}")
        
        return text, markdown, raw, img_out, []
    
    except Exception as e:
        os.unlink(tmp.name)
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        warnings.warn(error_msg)
        return f"Error: {str(e)}", "", "", None, []

@spaces.GPU(duration=120)
def process_pdf(path, mode_label, task_label, custom_prompt, dpi=300, page_indices=None, embed_figures=False, high_accuracy=False, insert_separators=True, max_retries=3, retry_backoff_seconds=3):
    doc = fitz.open(path)
    texts, markdowns, raws, all_crops = [], [], [], []
    if page_indices is None:
        page_indices = list(range(len(doc)))
    
    for i in page_indices:
        # Cache key for DeepSeekOCR per page
        cache_key = ("DeepSeekOCR", path, int(dpi), int(i), mode_label, task_label, bool(embed_figures), bool(high_accuracy))
        cached = _page_cache_get(cache_key)
        if cached:
            text, md, raw, crops = cached
            if text and text.strip() and text != "No text":
                texts.append(f"### Page {i + 1}\n\n{text}")
                markdowns.append(f"### Page {i + 1}\n\n{md}")
                raws.append(f"=== Page {i + 1} ===\n{raw}")
                all_crops.extend(crops or [])
            continue
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        
        # Retry loop to handle GPU timeouts/busy states gracefully
        attempt = 0
        while True:
            try:
                text, md, raw, _, crops = process_image(img, mode_label, task_label, custom_prompt, embed_figures=embed_figures, high_accuracy=high_accuracy)
                # If we got a result (even if it's "No text"), break the retry loop
                if text is not None:
                    break
                # If we got None or empty, retry
                attempt += 1
                if attempt >= max_retries:
                    text, md, raw, crops = "", f"<!-- Failed to process page {i+1} after retries -->", "", []
                    break
                time.sleep(retry_backoff_seconds * attempt)
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    error_msg = f"Error processing page {i+1}: {str(e)}"
                    text, md, raw, crops = error_msg, f"<!-- {error_msg} -->", error_msg, []
                    warnings.warn(error_msg)
                    break
                time.sleep(retry_backoff_seconds * attempt)
        
        # Check for valid text results (not empty, not error messages, not "No text")
        if text and text.strip() and text != "No text" and not text.startswith("Error") and not text.startswith(" "):
            texts.append(f"### Page {i + 1}\n\n{text}")
            markdowns.append(f"### Page {i + 1}\n\n{md}")
            raws.append(f"=== Page {i + 1} ===\n{raw}")
            all_crops.extend(crops)
            _page_cache_set(cache_key, (text, md, raw, crops))
        elif text and (text.startswith("Error") or text.startswith("Inference error")):
            # Include error messages in output for debugging
            texts.append(f"### Page {i + 1}\n\n{text}")
            markdowns.append(f"### Page {i + 1}\n\n<!-- {text} -->")
            raws.append(f"=== Page {i + 1} ===\n{text}")
    
    doc.close()
    
    sep = "\n\n---\n\n" if insert_separators else "\n\n"
    return (sep.join(texts) if texts else "",
            sep.join(markdowns) if markdowns else "",
            "\n\n".join(raws), None, all_crops)

def process_pdf_all(path, mode_label, task_label, custom_prompt, dpi=300, page_range_text="", embed_figures=False, high_accuracy=False, insert_separators=True, batch_size=3, max_retries=5, retry_backoff_seconds=5):
    doc = fitz.open(path)
    total_pages = len(doc)
    doc.close()
    
    # Parse page range like "1-3,5"
    def parse_ranges(s, total):
        if not s.strip():
            return list(range(total))
        pages = set()
        parts = [p.strip() for p in s.split(',') if p.strip()]
        for part in parts:
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    a, b = int(a) - 1, int(b) - 1
                except:
                    continue
                for x in range(max(0, a), min(total - 1, b) + 1):
                    pages.add(x)
            else:
                try:
                    idx = int(part) - 1
                    if 0 <= idx < total:
                        pages.add(idx)
                except:
                    continue
        return sorted(pages)

    target_pages = parse_ranges(page_range_text, total_pages)

    texts_all, mds_all, raws_all, crops_all = [], [], [], []
    for start in range(0, len(target_pages), batch_size):
        batch = target_pages[start:start+batch_size]
        # Orchestrate retries outside GPU scope (retries at chunk level)
        attempt = 0
        while True:
            try:
                tx, mdx, rawx, _, cropsx = process_pdf(path, mode_label, task_label, custom_prompt, dpi=dpi, page_indices=batch, embed_figures=embed_figures, high_accuracy=high_accuracy, insert_separators=insert_separators)
                break
            except Exception:
                attempt += 1
                if attempt >= max_retries:
                    tx, mdx, rawx, cropsx = "", "\n\n".join([f"<!-- Failed batch {start//batch_size+1} -->"]), "", []
                    break
                time.sleep(retry_backoff_seconds * attempt)
        if tx:
            texts_all.append(tx)
        if mdx:
            mds_all.append(mdx)
        if rawx:
            raws_all.append(rawx)
        crops_all.extend(cropsx)

    sep = "\n\n---\n\n" if insert_separators else "\n\n"
    return (sep.join(texts_all) if texts_all else "No text in PDF",
            sep.join(mds_all) if mds_all else "No text in PDF",
            "\n\n".join(raws_all), None, crops_all)

def process_pdf_all_gemini(path, dpi=300, page_range_text="", insert_separators=True, batch_size=3, max_retries=5, retry_backoff_seconds=5):
    if not GEMINI_AVAILABLE or not GEMINI_KEYS:
        return "Gemini not available or no keys", "", "", None, []
    doc = fitz.open(path)
    total_pages = len(doc)
    doc.close()

    def parse_ranges(s, total):
        if not s.strip():
            return list(range(total))
        pages = set()
        parts = [p.strip() for p in s.split(',') if p.strip()]
        for part in parts:
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    a, b = int(a) - 1, int(b) - 1
                except:
                    continue
                for x in range(max(0, a), min(total - 1, b) + 1):
                    pages.add(x)
            else:
                try:
                    idx = int(part) - 1
                    if 0 <= idx < total:
                        pages.add(idx)
                except:
                    continue
        return sorted(pages)

    target_pages = parse_ranges(page_range_text, total_pages)

    texts_all, mds_all, raws_all, crops_all = [], [], [], []
    for start in range(0, len(target_pages), batch_size):
        batch = target_pages[start:start+batch_size]
        attempt = 0
        while True:
            try:
                # Process each page to image and pass to Gemini
                docx = fitz.open(path)
                for i in batch:
                    cache_key = ("Gemini", path, int(dpi), int(i), GEMINI_MODEL)
                    cached = _page_cache_get(cache_key)
                    if cached:
                        text, md, raw, crops = cached
                        if text and text.strip() and text != "No text":
                            texts_all.append(f"### Page {i + 1}\n\n{text}")
                            mds_all.append(f"### Page {i + 1}\n\n{md}")
                            raws_all.append(f"=== Page {i + 1} ===\n{raw}")
                            crops_all.extend(crops or [])
                        continue
                    page = docx.load_page(i)
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    text, md, raw, _, crops = process_image_gemini(img)
                    if text and text != "No text" and not text.startswith("Error"):
                        texts_all.append(f"### Page {i + 1}\n\n{text}")
                        mds_all.append(f"### Page {i + 1}\n\n{md}")
                        raws_all.append(f"=== Page {i + 1} ===\n{raw}")
                        crops_all.extend(crops or [])
                        _page_cache_set(cache_key, (text, md, raw, crops))
                    elif text and text.startswith("Error"):
                        texts_all.append(f"### Page {i + 1}\n\n{text}")
                        mds_all.append(f"### Page {i + 1}\n\n<!-- {text} -->")
                        raws_all.append(f"=== Page {i + 1} ===\n{text}")
                docx.close()
                break
            except Exception:
                attempt += 1
                if attempt >= max_retries:
                    mds_all.append(f"<!-- Failed batch {start//batch_size+1} -->")
                    break
                time.sleep(retry_backoff_seconds * attempt)

    sep = "\n\n---\n\n" if insert_separators else "\n\n"
    return (sep.join(texts_all) if texts_all else "No text in PDF",
            sep.join(mds_all) if mds_all else "No text in PDF",
            "\n\n".join(raws_all), None, crops_all)

def process_pdf_paddleocrvl(path, dpi=300, page_indices=None, insert_separators=True, max_retries=3, retry_backoff_seconds=3):
    """Process PDF using PaddleOCR-VL."""
    # Early exit if engine is unavailable
    if not PADDLEOCRVL_AVAILABLE or paddleocrvl_pipeline is None:
        msg = PADDLEOCRVL_ERROR_MESSAGE or "PaddleOCR-VL not available. Install with: pip install 'paddleocr[doc-parser]'"
        return msg, f"<!-- {msg} -->", msg, None, []
    doc = fitz.open(path)
    texts, markdowns, raws, all_crops = [], [], [], []
    if page_indices is None:
        page_indices = list(range(len(doc)))
    
    for i in page_indices:
        cache_key = ("PaddleOCR-VL", path, int(dpi), int(i))
        cached = _page_cache_get(cache_key)
        if cached:
            text, md, raw, crops = cached
            if text and text.strip() and text != "No text detected":
                texts.append(f"### Page {i + 1}\n\n{text}")
                markdowns.append(f"### Page {i + 1}\n\n{md}")
                raws.append(f"=== Page {i + 1} ===\n{raw}")
                all_crops.extend(crops or [])
            continue
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        
        attempt = 0
        while True:
            try:
                text, md, raw, _, crops = process_image_paddleocrvl(img)
                break
            except Exception:
                attempt += 1
                if attempt >= max_retries:
                    text, md, raw, crops = "", f"<!-- Failed to process page {i+1} after retries -->", "", []
                    break
                time.sleep(retry_backoff_seconds * attempt)
        
        if text and text != "No text detected":
            texts.append(f"### Page {i + 1}\n\n{text}")
            markdowns.append(f"### Page {i + 1}\n\n{md}")
            raws.append(f"=== Page {i + 1} ===\n{raw}")
            all_crops.extend(crops)
            _page_cache_set(cache_key, (text, md, raw, crops))
    
    doc.close()
    
    sep = "\n\n---\n\n" if insert_separators else "\n\n"
    return (sep.join(texts) if texts else "",
            sep.join(markdowns) if markdowns else "",
            "\n\n".join(raws), None, all_crops)

def process_pdf_all_paddleocrvl(path, dpi=300, page_range_text="", insert_separators=True, batch_size=3, max_retries=5, retry_backoff_seconds=5):
    """Process all pages of PDF using PaddleOCR-VL."""
    # Early exit if engine is unavailable
    if not PADDLEOCRVL_AVAILABLE or paddleocrvl_pipeline is None:
        msg = PADDLEOCRVL_ERROR_MESSAGE or "PaddleOCR-VL not available. Install with: pip install 'paddleocr[doc-parser]'"
        return msg, f"<!-- {msg} -->", msg, None, []
    doc = fitz.open(path)
    total_pages = len(doc)
    doc.close()
    
    def parse_ranges(s, total):
        if not s.strip():
            return list(range(total))
        pages = set()
        parts = [p.strip() for p in s.split(',') if p.strip()]
        for part in parts:
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    a, b = int(a) - 1, int(b) - 1
                except:
                    continue
                for x in range(max(0, a), min(total - 1, b) + 1):
                    pages.add(x)
            else:
                try:
                    idx = int(part) - 1
                    if 0 <= idx < total:
                        pages.add(idx)
                except:
                    continue
        return sorted(pages)

    target_pages = parse_ranges(page_range_text, total_pages)

    texts_all, mds_all, raws_all, crops_all = [], [], [], []
    for start in range(0, len(target_pages), batch_size):
        batch = target_pages[start:start+batch_size]
        attempt = 0
        while True:
            try:
                tx, mdx, rawx, _, cropsx = process_pdf_paddleocrvl(path, dpi=dpi, page_indices=batch, insert_separators=insert_separators)
                break
            except Exception:
                attempt += 1
                if attempt >= max_retries:
                    tx, mdx, rawx, cropsx = "", "\n\n".join([f"<!-- Failed batch {start//batch_size+1} -->"]), "", []
                    break
                time.sleep(retry_backoff_seconds * attempt)
        if tx:
            texts_all.append(tx)
        if mdx:
            mds_all.append(mdx)
        if rawx:
            raws_all.append(rawx)
        crops_all.extend(cropsx)

    sep = "\n\n---\n\n" if insert_separators else "\n\n"
    return (sep.join(texts_all) if texts_all else "No text in PDF",
            sep.join(mds_all) if mds_all else "No text in PDF",
            "\n\n".join(raws_all), None, crops_all)

def process_file(path, mode_label, task_label, custom_prompt, dpi=300, page_range_text="", embed_figures=False, high_accuracy=False, insert_separators=True, ocr_engine="DeepSeekOCR"):
    if not path:
        return "Error Upload file", "", "", None, []
    
    if ocr_engine == "PaddleOCR-VL" or ocr_engine == "PaddleOCR":
        if path.lower().endswith('.pdf'):
            return process_pdf_all_paddleocrvl(path, dpi=dpi, page_range_text=page_range_text, insert_separators=insert_separators)
        else:
            return process_image_paddleocrvl(Image.open(path))
    else:
        if path.lower().endswith('.pdf'):
            return process_pdf_all(path, mode_label, task_label, custom_prompt, dpi=dpi, page_range_text=page_range_text, embed_figures=embed_figures, high_accuracy=high_accuracy, insert_separators=insert_separators)
        else:
            return process_image(Image.open(path), mode_label, task_label, custom_prompt, embed_figures=embed_figures, high_accuracy=high_accuracy)

def toggle_prompt(task_label):
    if task_label == "Custom":
        return gr.update(visible=True, label="Custom Prompt", placeholder="Add <|grounding|> for boxes")
    elif task_label == "Locate":
        return gr.update(visible=True, label="Text to Locate", placeholder="Enter text")
    return gr.update(visible=False)

def load_image(file_path):
    if not file_path:
        return None
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    else:
        return Image.open(file_path)

def get_pdf_page_count(file_path):
    try:
        doc = fitz.open(file_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 1

def render_pdf_page(file_path, page_number, dpi_value):
    try:
        doc = fitz.open(file_path)
        idx = max(1, min(page_number, len(doc))) - 1
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_value/72, dpi_value/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    except Exception:
        return None

def build_blocks(theme):
    with gr.Blocks(theme=theme, title="OCR-VLs") as demo:
        gr.Markdown("""
        # OCR-VLs WebUI
        **Convert documents to markdown, extract raw text, and locate specific content with bounding boxes.**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Uploader container
                file_in = gr.File(label="Upload Image or PDF", file_types=["image", ".pdf"], type="filepath")
                input_img = gr.Image(label="Input Image", type="pil", height=300)
                # PDF preview page selector container (visible only for PDFs)
                page_seps = gr.Checkbox(value=True, label="Insert page separators (---)")
                page_slider = gr.Slider(1, 1, value=1, step=1, label="Preview page", visible=False)
                # OCR Engine selector
                ocr_engine = gr.Radio(
                    choices=[c for c in ["DeepSeekOCR", "PaddleOCR-VL", "Gemini Flash 2.5"] if (c != "PaddleOCR-VL" or PADDLEOCRVL_AVAILABLE) and (c != "Gemini Flash 2.5" or GEMINI_AVAILABLE)],
                    value="DeepSeekOCR",
                    label="OCR Engine",
                    info="Choose between DeepSeekOCR, PaddleOCR-VL, or Gemini Flash 2.5"
                )
                # Processing options container (for DeepSeekOCR)
                mode = gr.Dropdown(list(MODE_LABEL_TO_KEY.keys()), value="Gundam", label="Mode (DeepSeekOCR)")
                task = gr.Dropdown(list(TASK_LABEL_TO_KEY.keys()), value="Markdown", label="Task (DeepSeekOCR)")
                prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
                with gr.Row():
                    embed_fig = gr.Checkbox(value=True, label="Embed figures into Markdown")
                    high_acc = gr.Checkbox(value=False, label="High accuracy (slower)")
                with gr.Row():
                    dpi = gr.Slider(150, 600, value=300, step=50, label="PDF DPI")
                    page_range = gr.Textbox(label="Page range (e.g. 1-3,5)", placeholder="All pages")            
                btn = gr.Button("Extract", variant="primary", size="lg")
            # Second row container
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Text"):
                        text_out = gr.Textbox(lines=20, show_copy_button=True, show_label=False)
                        dl_txt = gr.DownloadButton(label="Download Text", value=None)
                    with gr.Tab("Markdown"):
                        md_out = gr.Markdown("")
                        with gr.Row():
                            dl_md = gr.DownloadButton(label="Download Markdown", value=None)
                            dl_md_zip = gr.DownloadButton(label="Download Markdown (split pages)", value=None)
                    with gr.Tab("Boxes"):
                        img_out = gr.Image(type="pil", height=500, show_label=False)
                    with gr.Tab("Cropped Images"):
                        gallery = gr.Gallery(show_label=False, columns=3, height=400)
                    with gr.Tab("Raw"):
                        raw_out = gr.Textbox(lines=20, show_copy_button=True, show_label=False)
        
        
        with gr.Accordion("ℹ️ Info", open=False):
            gr.Markdown("""
            ### OCR Engines
            - **DeepSeekOCR**: AI-powered OCR with advanced document understanding and markdown conversion
            - **PaddleOCR-VL**: Document parsing model that converts documents to markdown format (install with: `pip install 'paddleocr[doc-parser]'`)
            - **Gemini Flash 2.5**: Google Gemini model for fast, high-quality Markdown conversion (set GEMINI_API_1..5 in .env)
            
            ### DeepSeekOCR Modes
            - Gundam: 1024 base + 640 tiles with cropping - Best balance
            - Tiny: 512×512, no crop - Fastest
            - Small: 640×640, no crop - Quick
            - Base: 1024×1024, no crop - Standard
            - Large: 1280×1280, no crop - Highest quality
            
            ### DeepSeekOCR Tasks
            - Markdown: Convert document to structured markdown (grounding)
            - Tables: Extract tables only as Markdown (grounding)
            - Locate: Find specific text in image (grounding)
            - Describe: General image description
            - Custom: Your own prompt (add `<|grounding|>` for boxes)
            
            ### PaddleOCR-VL
            - Document parsing model that automatically converts documents to markdown
            - Supports both images and PDFs

            ### Gemini Flash 2.5
            - Google Gemini model for fast, high-quality Markdown conversion
            """)
        
        # Enhanced preview logic for PDFs: show the selected page and slider
        def init_preview(file_path, dpi_value):
            fp = None
            if isinstance(file_path, str):
                fp = file_path
            elif isinstance(file_path, dict):
                fp = file_path.get('name') or file_path.get('path')
            if not fp:
                return None, gr.update(visible=False)
            if fp.lower().endswith('.pdf'):
                total = get_pdf_page_count(fp)
                img = render_pdf_page(fp, 1, int(dpi_value))
                return img, gr.update(visible=True, minimum=1, maximum=max(1, total), value=1)
            # Non-PDF
            try:
                return Image.open(fp), gr.update(visible=False)
            except Exception:
                return None, gr.update(visible=False)

        def update_preview_page(file_path, page_num, dpi_value):
            fp = None
            if isinstance(file_path, str):
                fp = file_path
            elif isinstance(file_path, dict):
                fp = file_path.get('name') or file_path.get('path')
            if fp and fp.lower().endswith('.pdf'):
                return render_pdf_page(fp, int(page_num), int(dpi_value))
            return input_img.value

        file_in.change(init_preview, [file_in, dpi], [input_img, page_slider])
        page_slider.change(update_preview_page, [file_in, page_slider, dpi], [input_img])
        dpi.release(update_preview_page, [file_in, page_slider, dpi], [input_img])
        task.change(toggle_prompt, [task], [prompt])
        
        def toggle_ocr_engine(engine):
            """Show/hide controls based on selected OCR engine."""
            if engine in ["PaddleOCR-VL", "Gemini Flash 2.5"]:
                return (
                    gr.update(visible=False),  # mode
                    gr.update(visible=False),  # task
                    gr.update(visible=False),  # prompt
                    gr.update(visible=False),  # embed_fig
                    gr.update(visible=False)  # high_acc
                )
            else:
                return (
                    gr.update(visible=True),   # mode
                    gr.update(visible=True),  # task
                    gr.update(visible=False), # prompt (will be toggled by task)
                    gr.update(visible=True),  # embed_fig
                    gr.update(visible=True)   # high_acc
                )
        
        ocr_engine.change(
            toggle_ocr_engine,
            [ocr_engine],
            [mode, task, prompt, embed_fig, high_acc]
        )
        
        def run(image, file_path, ocr_engine_val, mode_label, task_label, custom_prompt, dpi_val, page_range_text, embed, hiacc, sep_pages):
            # Normalize file path value from Gradio (can be str or dict)
            fp = None
            if isinstance(file_path, str):
                fp = file_path
            elif isinstance(file_path, dict):
                fp = file_path.get('name') or file_path.get('path')
            
            # Route to appropriate OCR engine
            if ocr_engine_val == "PaddleOCR-VL":
                # PaddleOCR-VL processing
                if fp and isinstance(fp, str) and fp.lower().endswith('.pdf'):
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="PaddleOCR-VL")
                elif image is not None:
                    text, md, raw, img, crops = process_image_paddleocrvl(image)
                elif fp:
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="PaddleOCR-VL")
                else:
                    return "Error uploading file or image", "", "", None, [], None, None, None
            elif ocr_engine_val == "Gemini Flash 2.5":
                # Gemini processing
                if fp and isinstance(fp, str) and fp.lower().endswith('.pdf'):
                    text, md, raw, img, crops = process_pdf_all_gemini(fp, dpi=int(dpi_val), page_range_text=page_range_text, insert_separators=sep_pages)
                elif image is not None:
                    text, md, raw, img, crops = process_image_gemini(image)
                elif fp:
                    text, md, raw, img, crops = process_pdf_all_gemini(fp, dpi=int(dpi_val), page_range_text=page_range_text, insert_separators=sep_pages)
                else:
                    return "Error uploading file or image", "", "", None, [], None, None, None
            else:
                # DeepSeekOCR processing
                if fp and isinstance(fp, str) and fp.lower().endswith('.pdf'):
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="DeepSeekOCR")
                elif image is not None:
                    text, md, raw, img, crops = process_image(image, mode_label, task_label, custom_prompt, embed_figures=embed, high_accuracy=hiacc)
                elif fp:
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="DeepSeekOCR")
                else:
                    return "Error uploading file or image", "", "", None, [], None, None, None

            # Create temp files for download
            md_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
            txt_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            with open(md_tmp.name, 'w', encoding='utf-8') as f:
                f.write(md or "")
            with open(txt_tmp.name, 'w', encoding='utf-8') as f:
                f.write(text or "")
            # Optional ZIP split by '---' separators
            zip_path = None
            try:
                if md:
                    # Split on standalone '---' separator variants
                    parts = re.split(r"\n\s*---\s*\n", md)
                    if len(parts) > 1:
                        zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                        with zipfile.ZipFile(zip_tmp.name, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for idx, part in enumerate(parts, start=1):
                                fname = f"page_{idx:03d}.md"
                                zf.writestr(fname, part.strip() + "\n")
                        zip_path = zip_tmp.name
            except Exception:
                zip_path = None
            return text, md, raw, img, crops, md_tmp.name, txt_tmp.name, zip_path
        
        btn.click(run, [input_img, file_in, ocr_engine, mode, task, prompt, dpi, page_range, embed_fig, high_acc, page_seps],
                  [text_out, md_out, raw_out, img_out, gallery, dl_md, dl_txt, dl_md_zip])
        
        return demo

# Build the demo
demo = build_blocks(gr.themes.Soft())

if __name__ == "__main__":
    demo.queue(max_size=20).launch()