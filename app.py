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
import logging

# Configure logging to be visible in console and Gradio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr so it's visible in Gradio
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
 
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
# Suppress specific transformers warnings that are expected
warnings.filterwarnings('ignore', category=UserWarning, module='transformers.generation.configuration_utils')
warnings.filterwarnings('ignore', message='.*pad_token_id.*eos_token_id.*')
warnings.filterwarnings('ignore', message='.*attention mask.*pad token id.*')
warnings.filterwarnings('ignore', message='.*attention mask.*cannot be inferred.*')
warnings.filterwarnings('ignore', message='.*seen_tokens.*deprecated.*')
warnings.filterwarnings('ignore', message='.*get_max_cache.*deprecated.*')
warnings.filterwarnings('ignore', message='.*position_ids.*position_embeddings.*')
warnings.filterwarnings('ignore', message='.*do_sample.*temperature.*')

def verify_gpu_access():
    """Verify that GPU is actually accessible by creating a test tensor."""
    if not torch.cuda.is_available():
        return False
    try:
        # Try to create a tensor on GPU to verify actual access
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False

def wait_for_gpu_and_move_model(max_wait_seconds=60, retry_interval=1):
    """Wait for GPU to become available and move model to GPU.
    For ZeroGPU, the GPU may take time to attach after entering @spaces.GPU() context.
    Returns True if model is on GPU, False otherwise.
    """
    device = next(model.parameters()).device
    if device.type == 'cuda':
        return True
    
    if not torch.cuda.is_available():
        return False
    
    # Wait for GPU to become accessible with retries
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < max_wait_seconds:
        attempts += 1
        elapsed = time.time() - start_time
        try:
            # Try to verify GPU access
            if verify_gpu_access():
                # GPU is accessible, try to move model
                try:
                    model.cuda()
                    # Verify it actually moved
                    new_device = next(model.parameters()).device
                    if new_device.type == 'cuda':
                        logger.info(f"Model moved to GPU after {attempts} attempts ({elapsed:.1f}s): {new_device}")
                        return True
                except Exception as e:
                    logger.debug(f"Attempt {attempts}: Failed to move model to GPU: {e}")
            else:
                logger.debug(f"Attempt {attempts} ({elapsed:.1f}s): GPU not yet accessible, waiting...")
        except Exception as e:
            logger.debug(f"Attempt {attempts}: GPU check failed: {e}")
        
        # Sleep before next attempt, but check if we have time left
        if time.time() - start_time < max_wait_seconds - retry_interval:
            time.sleep(retry_interval)
        else:
            break  # Not enough time for another attempt
    
    # Final check
    device = next(model.parameters()).device
    if device.type == 'cuda':
        return True
    
    logger.warning(f"GPU not available after {max_wait_seconds}s wait. Model remains on {device}")
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
    if torch.cuda.is_available() and verify_gpu_access():
        model = model.eval().cuda()
        # Verify model actually ended up on GPU
        actual_device = next(model.parameters()).device
        if actual_device.type == 'cuda':
            logger.info(f"Model loaded on GPU: {actual_device}")
        else:
            raise RuntimeError(f"Model failed to load on GPU. Device: {actual_device}")
    else:
        raise RuntimeError("CUDA not available or GPU not accessible; cannot use flash attention")
except Exception as e:
    logger.warning(f"Flash attention/CUDA unavailable ({e}); falling back to default attention.")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
    )
    # Try GPU first, fallback to CPU
    if torch.cuda.is_available() and verify_gpu_access():
        try:
            model = model.to('cuda').eval()
            actual_device = next(model.parameters()).device
            if actual_device.type == 'cuda':
                logger.info(f"Model loaded on GPU: {actual_device}")
            else:
                raise RuntimeError("Failed to move model to GPU")
        except Exception as e2:
            logger.warning(f"Failed to load on GPU ({e2}), falling back to CPU")
            model = model.to('cpu').eval()
            logger.warning("Model loaded on CPU - this may cause performance issues and 'No text in PDF' errors. GPU is required for proper operation.")
    else:
        model = model.to('cpu').eval()
        logger.warning("Model loaded on CPU - this may cause performance issues and 'No text in PDF' errors. GPU is required for proper operation.")

# Configure tokenizer after model is loaded
try:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a pad token if eos_token is also None
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            # Resize model embeddings if needed
            if hasattr(model, 'resize_token_embeddings'):
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass
    tokenizer.padding_side = 'right'
    
    # Try to set pad_token_id in model's generation config if available
    if hasattr(model, 'generation_config'):
        if hasattr(model.generation_config, 'pad_token_id') and model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.generation_config, 'eos_token_id') and model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
except Exception as e:
    # Silently continue if configuration fails
    pass

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

@spaces.GPU(duration=120)
def process_image(image, mode_label, task_label, custom_prompt, embed_figures=False, high_accuracy=False):
    if image is None:
        logger.warning("No image provided")
        return " Error Upload image", "", "", None, []
    if task_label in ["Custom", "Locate"] and not custom_prompt.strip():
        logger.warning("Custom prompt not provided")
        return "Enter prompt", "", "", None, []
    
    # Wait for GPU and move model (ZeroGPU may take time to attach)
    # Since we're in @spaces.GPU() context, assume GPU will become available
    logger.info("Waiting for GPU to become available (ZeroGPU may take time to attach)...")
    gpu_ready = wait_for_gpu_and_move_model(max_wait_seconds=60, retry_interval=1)
    
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()
    logger.info(f"Processing image - Device: {device.type}, CUDA available: {cuda_available}, GPU ready: {gpu_ready}")
    
    # Only error if GPU is still not ready after waiting
    if device.type == 'cpu':
        error_msg = f"Error: GPU not available after waiting. Model is on CPU. ZeroGPU may not be attached or quota limit reached. Current device: {device.type}, CUDA available: {cuda_available}"
        logger.error(error_msg)
        return error_msg, "", "", None, []
    
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
        logger.info(f"Running inference with mode: {mode_key}, task: {task_key}")
        model.infer(tokenizer=tokenizer, prompt=prompt, image_file=tmp.name, output_path=out_dir,
                    base_size=config["base_size"], image_size=config["image_size"], crop_mode=config["crop_mode"])
        
        result = '\n'.join([l for l in sys.stdout.getvalue().split('\n') 
                            if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size'])]).strip()
        logger.info(f"Inference completed. Result length: {len(result)}")
    except Exception as e:
        result = ""
        error_msg = f"Error during inference: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Also print to stderr so it's visible
        print(f"ERROR: {error_msg}", file=sys.stderr)
    finally:
        sys.stdout = stdout
        os.unlink(tmp.name)
        shutil.rmtree(out_dir, ignore_errors=True)
    
    if not result:
        error_msg = f"No text extracted from image. This may occur if GPU is unavailable or inference failed. Device: {device.type}, CUDA available: {torch.cuda.is_available()}"
        logger.error(error_msg)
        return error_msg, "", "", None, []
    
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
            logger.info("Running high accuracy refinement pass")
            model.infer(tokenizer=tokenizer, prompt=refine_prompt, image_file=tmp2.name, output_path=out_dir2,
                        base_size=config["base_size"], image_size=config["image_size"], crop_mode=config["crop_mode"])
            refine_result = '\n'.join([l for l in sys.stdout.getvalue().split('\n')
                                if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size'])]).strip()
        except Exception as e:
            refine_result = ""
            logger.warning(f"High accuracy refinement failed: {str(e)}", exc_info=True)
        finally:
            sys.stdout = stdout2
            os.unlink(tmp2.name)
            shutil.rmtree(out_dir2, ignore_errors=True)
        if refine_result:
            refined_md = clean_output(refine_result, embed_figures, True)
            # Prefer refined markdown if longer (heuristic)
            if len(refined_md) > len(markdown):
                markdown = refined_md
    
    return cleaned, markdown, result, img_out, crops

@spaces.GPU(duration=120)
def process_pdf(path, mode_label, task_label, custom_prompt, dpi=300, page_indices=None, embed_figures=False, high_accuracy=False, insert_separators=True, max_retries=3, retry_backoff_seconds=3):
    logger.info(f"Processing PDF: {path}, pages: {page_indices}")
    doc = fitz.open(path)
    texts, markdowns, raws, all_crops = [], [], [], []
    if page_indices is None:
        page_indices = list(range(len(doc)))
    
    # Wait for GPU and move model (ZeroGPU may take time to attach)
    # Since we're in @spaces.GPU() context, assume GPU will become available
    logger.info("Waiting for GPU to become available (ZeroGPU may take time to attach)...")
    gpu_ready = wait_for_gpu_and_move_model(max_wait_seconds=60, retry_interval=1)
    
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()
    logger.info(f"PDF processing - Device: {device.type}, CUDA available: {cuda_available}, GPU ready: {gpu_ready}")
    
    # Only error if GPU is still not ready after waiting
    if device.type == 'cpu':
        error_msg = f"Error: GPU not available after waiting. Model is on CPU. ZeroGPU may not be attached or quota limit reached. Current device: {device.type}, CUDA available: {cuda_available}"
        logger.error(error_msg)
        doc.close()
        return (error_msg, error_msg, error_msg, None, [])
    
    for i in page_indices:
        logger.info(f"Processing page {i+1}/{len(page_indices)}")
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        
        # Retry loop to handle GPU timeouts/busy states gracefully
        attempt = 0
        while True:
            try:
                text, md, raw, _, crops = process_image(img, mode_label, task_label, custom_prompt, embed_figures=embed_figures, high_accuracy=high_accuracy)
                # Check if we got an error message about GPU
                if text and text.startswith("Error: GPU not available"):
                    logger.error(f"GPU error detected on page {i+1}: {text}")
                    attempt = max_retries  # Force exit retry loop
                    break
                # Check if result is empty (likely CPU inference failure)
                if not text or text.strip() == "" or "No text extracted" in text:
                    logger.warning(f"Empty result for page {i+1}. Text: {text[:100] if text else 'None'}")
                break
            except Exception as e:
                attempt += 1
                logger.error(f"Error processing page {i+1}, attempt {attempt}/{max_retries}: {str(e)}", exc_info=True)
                if attempt >= max_retries:
                    error_detail = str(e) if str(e) else "Unknown error"
                    text, md, raw, crops = "", f"<!-- Failed to process page {i+1} after {max_retries} retries: {error_detail} -->", "", []
                    break
                time.sleep(retry_backoff_seconds * attempt)
        
        # Skip pages that failed or returned errors
        if text and text != "No text" and not text.startswith("Error:") and "No text extracted" not in text:
            texts.append(f"### Page {i + 1}\n\n{text}")
            markdowns.append(f"### Page {i + 1}\n\n{md}")
            raws.append(f"=== Page {i + 1} ===\n{raw}")
            all_crops.extend(crops)
            logger.info(f"Successfully processed page {i+1}")
        elif text and text.startswith("Error: GPU not available"):
            # If GPU error, add a note and stop processing
            logger.error(f"Stopping PDF processing due to GPU error on page {i+1}")
            texts.append(f"### Page {i + 1}\n\n{text}")
            markdowns.append(f"### Page {i + 1}\n\n{text}")
            raws.append(f"=== Page {i + 1} ===\n{text}")
            break  # Stop processing remaining pages
        else:
            logger.warning(f"Skipping page {i+1} - empty result or error: {text[:100] if text else 'None'}")
    
    doc.close()
    
    sep = "\n\n---\n\n" if insert_separators else "\n\n"
    return (sep.join(texts) if texts else "",
            sep.join(markdowns) if markdowns else "",
            "\n\n".join(raws), None, all_crops)

def process_pdf_all(path, mode_label, task_label, custom_prompt, dpi=300, page_range_text="", embed_figures=False, high_accuracy=False, insert_separators=True, batch_size=5, max_retries=5, retry_backoff_seconds=5):
    logger.info(f"Starting PDF processing for: {path}")
    doc = fitz.open(path)
    total_pages = len(doc)
    doc.close()
    logger.info(f"PDF has {total_pages} pages")
    
    # Wait for GPU and move model (ZeroGPU may take time to attach)
    # Since we're in @spaces.GPU() context, assume GPU will become available
    logger.info("Waiting for GPU to become available (ZeroGPU may take time to attach)...")
    gpu_ready = wait_for_gpu_and_move_model(max_wait_seconds=60, retry_interval=1)
    
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()
    logger.info(f"PDF processing - Device: {device.type}, CUDA available: {cuda_available}, GPU ready: {gpu_ready}")
    
    # Only error if GPU is still not ready after waiting
    if device.type == 'cpu':
        error_msg = f"Error: GPU not available after waiting. Model is on CPU. ZeroGPU may not be attached or quota limit reached. Current device: {device.type}, CUDA available: {cuda_available}"
        logger.error(error_msg)
        return (error_msg, error_msg, error_msg, None, [])
    
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
    logger.info(f"Processing pages: {target_pages}")

    texts_all, mds_all, raws_all, crops_all = [], [], [], []
    for start in range(0, len(target_pages), batch_size):
        batch = target_pages[start:start+batch_size]
        logger.info(f"Processing batch: pages {batch}")
        # Orchestrate retries outside GPU scope (retries at chunk level)
        attempt = 0
        while True:
            try:
                tx, mdx, rawx, _, cropsx = process_pdf(path, mode_label, task_label, custom_prompt, dpi=dpi, page_indices=batch, embed_figures=embed_figures, high_accuracy=high_accuracy, insert_separators=insert_separators)
                break
            except Exception as e:
                attempt += 1
                logger.error(f"Error processing batch {start//batch_size+1}, attempt {attempt}/{max_retries}: {str(e)}", exc_info=True)
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
    # Check if we got GPU errors
    has_gpu_error = any("GPU not available" in txt for txt in texts_all) or any("GPU not available" in md for md in mds_all)
    if has_gpu_error:
        error_msg = "Error: GPU not available. ZeroGPU limit may be reached. Please wait and try again."
        logger.error(error_msg)
        return (error_msg, error_msg, error_msg, None, crops_all)
    
    # Check if we have any results
    if not texts_all:
        device = next(model.parameters()).device
        cuda_available = torch.cuda.is_available()
        error_msg = f"No text in PDF. This may occur if:\n1. GPU is unavailable (device: {device.type}, CUDA available: {cuda_available})\n2. All pages failed to process\n3. PDF contains only images without text\n\nCheck logs for detailed error information."
        logger.error(error_msg)
        return (error_msg, error_msg, error_msg, None, crops_all)
    
    logger.info(f"Successfully processed PDF with {len(texts_all)} batches")
    return (sep.join(texts_all),
            sep.join(mds_all),
            "\n\n".join(raws_all), None, crops_all)

def process_file(path, mode_label, task_label, custom_prompt, dpi=300, page_range_text="", embed_figures=False, high_accuracy=False, insert_separators=True):
    if not path:
        return "Error Upload file", "", "", None, []
    
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
    with gr.Blocks(theme=theme, title="DeepSeek-OCR") as demo:
        gr.Markdown("""
        # DeepSeek-OCR WebUI
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
                # Processing options container
                mode = gr.Dropdown(list(MODE_LABEL_TO_KEY.keys()), value="Gundam", label="Mode")
                task = gr.Dropdown(list(TASK_LABEL_TO_KEY.keys()), value="Markdown", label="Task")
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
            ### Modes
            - ⚡ Gundam: 1024 base + 640 tiles with cropping - Best balance
            - 🧩 Tiny: 512×512, no crop - Fastest
            - 📄 Small: 640×640, no crop - Quick
            - 📚 Base: 1024×1024, no crop - Standard
            - 🖼️ Large: 1280×1280, no crop - Highest quality
            
            ### Tasks
            - Markdown: Convert document to structured markdown (grounding ✅)
            - Tables: Extract tables only as Markdown (grounding ✅)
            - Locate: Find specific text in image (grounding ✅)
            - Describe: General image description
            - Custom: Your own prompt (add `<|grounding|>` for boxes)
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
        
        def run(image, file_path, mode_label, task_label, custom_prompt, dpi_val, page_range_text, embed, hiacc, sep_pages):
            # Normalize file path value from Gradio (can be str or dict)
            fp = None
            if isinstance(file_path, str):
                fp = file_path
            elif isinstance(file_path, dict):
                fp = file_path.get('name') or file_path.get('path')
            
            # Prioritize file path for PDFs to process all pages
            if fp and isinstance(fp, str) and fp.lower().endswith('.pdf'):
                text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages)
            elif image is not None:
                text, md, raw, img, crops = process_image(image, mode_label, task_label, custom_prompt, embed_figures=embed, high_accuracy=hiacc)
            elif fp:
                text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages)
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
        
        btn.click(run, [input_img, file_in, mode, task, prompt, dpi, page_range, embed_fig, high_acc, page_seps],
                  [text_out, md_out, raw_out, img_out, gallery, dl_md, dl_txt, dl_md_zip])
        
        return demo

# Build two themed experiences as a light/dark separator without custom CSS/JS
light_demo = build_blocks(gr.themes.Soft())
dark_demo = build_blocks(gr.themes.Monochrome())

app = gr.TabbedInterface([light_demo, dark_demo], ["Light", "Dark"]) 

if __name__ == "__main__":
    app.queue(max_size=20).launch()