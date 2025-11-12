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
import json

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token.*")
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
warnings.filterwarnings("ignore", message=".*get_usable_length.*")
warnings.filterwarnings("ignore", message=".*get_max_cache.*")
warnings.filterwarnings("ignore", message=".*get_seq_length.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*position_embeddings.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

# Patch DynamicCache to fix deprecated attribute errors
# This is a compatibility fix for transformers >= 4.47.0 where seen_tokens, get_max_length, and get_usable_length were deprecated
try:
    from transformers.cache_utils import DynamicCache
    
    if not hasattr(DynamicCache, 'seen_tokens'):
        # Add seen_tokens property for backward compatibility
        def _get_seen_tokens(self):
            """Backward compatibility property for seen_tokens"""
            # Calculate seen_tokens from the cache structure
            if hasattr(self, 'key_cache') and self.key_cache:
                # Return the length of the first layer's key cache
                first_layer_keys = list(self.key_cache.values())[0] if self.key_cache else None
                if first_layer_keys is not None and len(first_layer_keys) > 0:
                    return first_layer_keys[0].shape[-2] if hasattr(first_layer_keys[0], 'shape') else 0
            return 0
        
        DynamicCache.seen_tokens = property(_get_seen_tokens)
    
    if not hasattr(DynamicCache, 'get_max_length'):
        # Add get_max_length method for backward compatibility
        # In newer transformers, this was replaced with cache_position or similar
        def _get_max_length(self):
            """Backward compatibility method for get_max_length"""
            # Try to get max length from cache structure
            if hasattr(self, 'key_cache') and self.key_cache:
                first_layer_keys = list(self.key_cache.values())[0] if self.key_cache else None
                if first_layer_keys is not None and len(first_layer_keys) > 0:
                    # Return the sequence length dimension
                    if hasattr(first_layer_keys[0], 'shape') and len(first_layer_keys[0].shape) >= 2:
                        return first_layer_keys[0].shape[-2]
            # Fallback: try cache_position if available (newer API)
            if hasattr(self, 'cache_position') and self.cache_position is not None:
                if hasattr(self.cache_position, '__len__'):
                    return len(self.cache_position)
                elif hasattr(self.cache_position, 'shape'):
                    return self.cache_position.shape[-1] if len(self.cache_position.shape) > 0 else 0
            # Default fallback
            return 0
        
        DynamicCache.get_max_length = _get_max_length
    
    if not hasattr(DynamicCache, 'get_usable_length'):
        # Add get_usable_length method for backward compatibility
        # In newer transformers, this was replaced with get_seq_length or similar
        def _get_usable_length(self, seq_length=None):
            """Backward compatibility method for get_usable_length
            
            Args:
                seq_length: Optional sequence length parameter (for compatibility with old API)
            """
            # Try to use get_seq_length if available (newer API)
            if hasattr(self, 'get_seq_length'):
                try:
                    return self.get_seq_length()
                except:
                    pass
            
            # Try to get usable length from cache structure
            if hasattr(self, 'key_cache') and self.key_cache:
                first_layer_keys = list(self.key_cache.values())[0] if self.key_cache else None
                if first_layer_keys is not None and len(first_layer_keys) > 0:
                    # Return the sequence length dimension
                    if hasattr(first_layer_keys[0], 'shape') and len(first_layer_keys[0].shape) >= 2:
                        cache_length = first_layer_keys[0].shape[-2]
                        # If seq_length is provided, return the minimum (usable portion)
                        if seq_length is not None:
                            return min(cache_length, seq_length)
                        return cache_length
            
            # Fallback: try cache_position if available (newer API)
            if hasattr(self, 'cache_position') and self.cache_position is not None:
                if hasattr(self.cache_position, '__len__'):
                    pos_len = len(self.cache_position)
                    if seq_length is not None:
                        return min(pos_len, seq_length)
                    return pos_len
                elif hasattr(self.cache_position, 'shape'):
                    pos_len = self.cache_position.shape[-1] if len(self.cache_position.shape) > 0 else 0
                    if seq_length is not None:
                        return min(pos_len, seq_length)
                    return pos_len
            
            # If seq_length is provided, return it; otherwise return 0
            return seq_length if seq_length is not None else 0
        
        DynamicCache.get_usable_length = _get_usable_length
    
    # Also add get_seq_length as an alias if it doesn't exist and get_usable_length does
    if not hasattr(DynamicCache, 'get_seq_length') and hasattr(DynamicCache, 'get_usable_length'):
        def _get_seq_length(self):
            """Backward compatibility method for get_seq_length (alias for get_usable_length)"""
            return self.get_usable_length()
        
        DynamicCache.get_seq_length = _get_seq_length
        
except (ImportError, AttributeError):
    # If DynamicCache doesn't exist or patch fails, continue anyway
    pass

# Optional dependency installer (used to keep base image small on Spaces)
_OPTIONAL_INSTALL_LOCK = Lock()
_OPTIONAL_INSTALL_CACHE = set()

def _install_optional_packages(packages, context="optional dependency"):
    """Install optional packages lazily using pip."""
    pending = [pkg for pkg in packages if pkg not in _OPTIONAL_INSTALL_CACHE]
    if not pending:
        return
    with _OPTIONAL_INSTALL_LOCK:
        pending = [pkg for pkg in pending if pkg not in _OPTIONAL_INSTALL_CACHE]
        if not pending:
            return
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pending])
            _OPTIONAL_INSTALL_CACHE.update(pending)
            importlib.invalidate_caches()
        except Exception as install_error:
            warnings.warn(f"Failed to install {context} packages {pending}: {install_error}")
            raise


_DEEPSEEK_OPTIONAL_PACKAGES = {
    "matplotlib": os.getenv("DEEPSEEK_MATPLOTLIB_SPEC", "matplotlib>=3.8.0"),
    "torchvision": os.getenv("DEEPSEEK_TORCHVISION_SPEC", "torchvision>=0.19.0"),
}

def _ensure_deepseek_visual_deps():
    """Ensure DeepSeekOCR's optional visualization dependencies are installed."""
    install_specs = []
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        install_specs.append(_DEEPSEEK_OPTIONAL_PACKAGES["matplotlib"])
    try:
        import torchvision  # noqa: F401
    except ImportError:
        install_specs.append(_DEEPSEEK_OPTIONAL_PACKAGES["torchvision"])
    if install_specs:
        _install_optional_packages(install_specs, "DeepSeekOCR visual dependencies")
        import matplotlib  # noqa: F401
        import torchvision  # noqa: F401

# PaddleOCR-VL imports are deferred to reduce build-time dependencies
PADDLEOCRVL_AVAILABLE = True
PaddleOCRVL = None
PaddleOCR = None
PADDLEOCRVL_ERROR_MESSAGE = None
_PADDLE_OPTIONAL_PACKAGES = [
    os.getenv("PADDLEPADDLE_SPEC", "paddlepaddle==2.5.2"),
    os.getenv("PADDLEOCR_SPEC", "paddleocr[doc-parser]==2.7.0"),
]

def _import_paddleocr():
    global PaddleOCR, PaddleOCRVL, PADDLEOCRVL_AVAILABLE, PADDLEOCRVL_ERROR_MESSAGE
    try:
        import paddleocr  # type: ignore
    except ImportError:
        _install_optional_packages(_PADDLE_OPTIONAL_PACKAGES, "PaddleOCR support")
        import paddleocr  # type: ignore
    PaddleOCR = paddleocr.PaddleOCR
    
    try:
        from paddleocr import PaddleOCRVL  # type: ignore
        PADDLEOCRVL_AVAILABLE = True
    except ImportError:
        try:
            from paddleocr.paddleocr_vl import PaddleOCRVL  # type: ignore
            PADDLEOCRVL_AVAILABLE = True
        except ImportError:
            try:
                if hasattr(paddleocr, 'PaddleOCRVL'):
                    PaddleOCRVL = paddleocr.PaddleOCRVL
                    PADDLEOCRVL_AVAILABLE = True
                elif hasattr(paddleocr, 'paddleocr_vl'):
                    try:
                        from paddleocr.paddleocr_vl import PaddleOCRVL  # type: ignore
                        PADDLEOCRVL_AVAILABLE = True
                    except Exception:
                        pass
            except Exception as e:
                PADDLEOCRVL_ERROR_MESSAGE = f"PaddleOCR-VL import failed: {str(e)}"
                
    if not PADDLEOCRVL_AVAILABLE:
        try:
            _ = PaddleOCR(use_angle_cls=True, lang='en')
            PADDLEOCRVL_ERROR_MESSAGE = "PaddleOCR-VL class not found. Using regular PaddleOCR instead. For document parsing, ensure 'paddleocr[doc-parser]' is installed."
        except Exception as test_error:
            PADDLEOCRVL_ERROR_MESSAGE = f"PaddleOCR not working: {str(test_error)}"

# Gemini imports (optional)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    types = None
    warnings.warn("Gemini SDK not available. Install with: pip install google-generativeai")

# olmOCR imports (optional)
OLMOCR_AVAILABLE = False
OLMOCR_MODEL = None
OLMOCR_PROCESSOR = None
OLMOCR_ERROR_MESSAGE = None

# Try to install olmocr conditionally if Python >= 3.11
import sys
if sys.version_info >= (3, 11):
    try:
        import olmocr
    except ImportError:
        # Try to install olmocr if not available
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 
                'git+https://github.com/allenai/olmocr.git'
            ])
            # Reload import after installation
            import importlib
            importlib.invalidate_caches()
        except Exception as install_error:
            warnings.warn(f"Failed to auto-install olmocr: {install_error}. You may need to install it manually: pip install git+https://github.com/allenai/olmocr.git")

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    OLMOCR_AVAILABLE = True
except ImportError as e:
    OLMOCR_AVAILABLE = False
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 11):
        # Check if we're on ZeroGPU (which is locked to Python 3.10.13)
        is_zerogpu = os.getenv("SPACES_ZERO_GPU", "").lower() in ("true", "1") or "zerogpu" in os.getenv("SPACE_ID", "").lower()
        if is_zerogpu:
            OLMOCR_ERROR_MESSAGE = f"olmOCR requires Python >=3.11, but ZeroGPU Spaces are locked to Python 3.10.13. olmOCR is not available on ZeroGPU. Use a regular GPU Space or a different OCR engine."
        else:
            OLMOCR_ERROR_MESSAGE = f"olmOCR requires Python >=3.11, but you have Python {python_version}. For Hugging Face Spaces, create a runtime.txt file with 'python-3.11' or higher. Note: ZeroGPU Spaces are locked to Python 3.10.13 and do not support olmOCR."
    else:
        OLMOCR_ERROR_MESSAGE = f"olmOCR not available. Install with: pip install git+https://github.com/allenai/olmocr.git. Error: {str(e)}"
    warnings.warn(OLMOCR_ERROR_MESSAGE)
except Exception as e:
    OLMOCR_AVAILABLE = False
    OLMOCR_ERROR_MESSAGE = f"olmOCR setup failed: {str(e)}"
    warnings.warn(OLMOCR_ERROR_MESSAGE)

# dots.ocr imports (optional)
DOTSOCR_AVAILABLE = True
DOTSOCR_MODEL = None
DOTSOCR_PROCESSOR = None
DOTSOCR_ERROR_MESSAGE = None
_QWEN_OPTIONAL_PACKAGES = [
    os.getenv("QWEN_VL_UTILS_SPEC", "qwen-vl-utils>=0.1.0"),
]

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as e:
    DOTSOCR_AVAILABLE = False
    DOTSOCR_ERROR_MESSAGE = f"dots.ocr not available. Install with: pip install transformers. Error: {str(e)}"
    warnings.warn(DOTSOCR_ERROR_MESSAGE)
except Exception as e:
    DOTSOCR_AVAILABLE = False
    DOTSOCR_ERROR_MESSAGE = f"dots.ocr setup failed: {str(e)}"
    warnings.warn(DOTSOCR_ERROR_MESSAGE)

process_vision_info = None

def _ensure_dotsocr_dependencies():
    """Ensure qwen-vl-utils is available before using dots.ocr."""
    global process_vision_info, DOTSOCR_AVAILABLE, DOTSOCR_ERROR_MESSAGE
    if process_vision_info is not None:
        return
    try:
        from qwen_vl_utils import process_vision_info as _process_vision_info
        process_vision_info = _process_vision_info
    except ImportError:
        try:
            _install_optional_packages(_QWEN_OPTIONAL_PACKAGES, "dots.ocr support (qwen-vl-utils)")
            from qwen_vl_utils import process_vision_info as _process_vision_info  # type: ignore
            process_vision_info = _process_vision_info
        except Exception as install_error:
            DOTSOCR_AVAILABLE = False
            DOTSOCR_ERROR_MESSAGE = f"dots.ocr not available. Install with: pip install qwen-vl-utils. Error: {install_error}"
            raise

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
                contents=[content]
            )
            return response.text
        except Exception as e:
            logger.error(f"[LLM] ❌ Error calling Gemini API: {e}")
            return "Error generating response from Gemini."

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

# Monkey-patch LlamaFlashAttention2 if it doesn't exist (for compatibility with transformers >= 4.47.0)
# The model's custom code tries to import LlamaFlashAttention2 which was removed in transformers 4.47.0+
try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    # LlamaFlashAttention2 doesn't exist, create a compatibility wrapper
    try:
        import transformers.models.llama.modeling_llama as llama_module
        import inspect
        
        # Try to use LlamaSdpaAttention or LlamaAttention as base
        BaseAttention = None
        if hasattr(llama_module, 'LlamaSdpaAttention'):
            BaseAttention = llama_module.LlamaSdpaAttention
        elif hasattr(llama_module, 'LlamaAttention'):
            BaseAttention = llama_module.LlamaAttention
        
        if BaseAttention is not None:
            # Create a compatibility wrapper class that handles signature differences
            class LlamaFlashAttention2(BaseAttention):
                """Compatibility wrapper for LlamaFlashAttention2 (removed in transformers 4.47.0+)"""
                
                def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                          past_key_value=None, output_attentions=False, use_cache=False,
                          cache_position=None, position_embeddings=None, **kwargs):
                    """
                    Forward method that adapts between old LlamaFlashAttention2 signature
                    and new LlamaAttention signature.
                    """
                    # Get the signature of the parent class forward method
                    parent_forward = super().forward
                    sig = inspect.signature(parent_forward)
                    params = sig.parameters
                    
                    # Build arguments dict, excluding 'self'
                    forward_kwargs = {
                        'hidden_states': hidden_states,
                    }
                    
                    # Add optional arguments only if they're in the parent signature
                    if 'attention_mask' in params:
                        forward_kwargs['attention_mask'] = attention_mask
                    if 'position_ids' in params:
                        forward_kwargs['position_ids'] = position_ids
                    if 'past_key_value' in params:
                        forward_kwargs['past_key_value'] = past_key_value
                    if 'output_attentions' in params:
                        forward_kwargs['output_attentions'] = output_attentions
                    if 'use_cache' in params:
                        forward_kwargs['use_cache'] = use_cache
                    if 'cache_position' in params:
                        forward_kwargs['cache_position'] = cache_position
                    
                    # Handle position_embeddings - critical for compatibility
                    if 'position_embeddings' in params:
                        # Parent accepts position_embeddings
                        param = params['position_embeddings']
                        if position_embeddings is not None:
                            forward_kwargs['position_embeddings'] = position_embeddings
                        elif param.default is inspect.Parameter.empty:
                            # Required parameter but not provided - try to generate it
                            if hasattr(self, 'rotary_emb') and position_ids is not None:
                                try:
                                    # Generate position embeddings using rotary_emb
                                    forward_kwargs['position_embeddings'] = self.rotary_emb(
                                        hidden_states, position_ids=position_ids
                                    )
                                except Exception:
                                    # If generation fails, pass None and let parent handle it
                                    forward_kwargs['position_embeddings'] = None
                            else:
                                # Can't generate, pass None and hope parent can handle it
                                forward_kwargs['position_embeddings'] = None
                        # If it has a default, we don't need to pass it
                    # If parent doesn't accept position_embeddings, we ignore it
                    # (parent will generate it internally from position_ids)
                    
                    # Add any additional kwargs that parent accepts
                    for key, value in kwargs.items():
                        if key in params:
                            forward_kwargs[key] = value
                    
                    # Call parent forward with adapted arguments
                    return parent_forward(**forward_kwargs)
            
            llama_module.LlamaFlashAttention2 = LlamaFlashAttention2
        else:
            # Last resort: create a minimal dummy class
            class LlamaFlashAttention2:
                """Compatibility alias for LlamaFlashAttention2 (removed in transformers 4.47.0+)"""
                def __init__(self, *args, **kwargs):
                    pass
            llama_module.LlamaFlashAttention2 = LlamaFlashAttention2
    except Exception as e:
        warnings.warn(f"Could not create LlamaFlashAttention2 compatibility layer: {e}. Model loading may fail.")

# Ensure DeepSeek visual dependencies are ready (installs lazily if missing)
try:
    _ensure_deepseek_visual_deps()
except Exception as dep_error:
    warnings.warn(f"DeepSeekOCR dependencies missing: {dep_error}")

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
model = None
# Try loading with flash attention first if available
if flash_ok and torch.cuda.is_available():
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_safetensors=True,
        )
        model = model.eval().cuda()
    except (ImportError, AttributeError) as e:
        error_str = str(e)
        # If LlamaFlashAttention2 import fails, fall back to default attention
        if "LlamaFlashAttention2" in error_str or "cannot import name" in error_str:
            warnings.warn(f"Flash attention not available due to transformers version ({error_str}); falling back to default attention.")
            model = None  # Will be loaded below
        else:
            # Other import errors, try fallback
            warnings.warn(f"Flash attention unavailable ({error_str}); falling back to default attention.")
            model = None
    except Exception as e:
        warnings.warn(f"Flash attention/CUDA unavailable ({e}); falling back to default attention.")
        model = None

# Load with default attention if flash attention failed or wasn't attempted
if model is None:
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation=None,  # Explicitly use default attention
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_safetensors=True,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
    except (ImportError, AttributeError) as e:
        error_str = str(e)
        # If still failing due to LlamaFlashAttention2, try without specifying attn_implementation
        if "LlamaFlashAttention2" in error_str or "cannot import name" in error_str:
            warnings.warn(f"Model custom code requires LlamaFlashAttention2 but it's not available. Trying without explicit attention setting.")
            try:
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device).eval()
            except Exception as e2:
                try:
                    import transformers
                    transformers_version = transformers.__version__
                except:
                    transformers_version = "unknown"
                raise RuntimeError(f"Failed to load DeepSeekOCR model. The model's custom code requires LlamaFlashAttention2 which is not available in transformers {transformers_version}. Please upgrade transformers: pip install transformers>=4.47.0. Error: {e2}")
        else:
            raise RuntimeError(f"Failed to load DeepSeekOCR model: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load DeepSeekOCR model: {e}")

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

# Defer olmOCR model init until first use to avoid startup errors
olmocr_model = None
olmocr_processor = None

# Defer dots.ocr model init until first use to avoid startup errors
dotsocr_model = None
dotsocr_processor = None

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
    global paddleocrvl_pipeline, PADDLEOCRVL_AVAILABLE, PADDLEOCRVL_ERROR_MESSAGE, PaddleOCRVL
    if PaddleOCR is None or PaddleOCRVL is None:
        try:
            _import_paddleocr()
        except Exception as e:
            PADDLEOCRVL_AVAILABLE = False
            PADDLEOCRVL_ERROR_MESSAGE = f"PaddleOCR-VL setup failed: {e}"
    
    if not PADDLEOCRVL_AVAILABLE or PaddleOCRVL is None:
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

def _init_olmocr_model():
    """Lazy initialization of olmOCR model."""
    global olmocr_model, olmocr_processor, OLMOCR_AVAILABLE, OLMOCR_ERROR_MESSAGE
    
    if not OLMOCR_AVAILABLE:
        msg = OLMOCR_ERROR_MESSAGE or "olmOCR not available. Install with: pip install git+https://github.com/allenai/olmocr.git (requires Python >=3.11)"
        raise RuntimeError(msg)
    
    if olmocr_model is None or olmocr_processor is None:
        try:
            model_name = "allenai/olmOCR-2-7B-1025"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            olmocr_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            ).eval()
            olmocr_model.to(device)
            
            olmocr_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
        except Exception as e:
            OLMOCR_AVAILABLE = False
            OLMOCR_ERROR_MESSAGE = f"olmOCR model initialization failed: {str(e)}"
            raise RuntimeError(OLMOCR_ERROR_MESSAGE)

def _resize_image_for_olmocr(image: Image.Image, target_longest_dim: int = 1288) -> Image.Image:
    """Resize image so longest dimension is target_longest_dim pixels."""
    width, height = image.size
    longest_dim = max(width, height)
    
    if longest_dim == target_longest_dim:
        return image
    
    scale = target_longest_dim / longest_dim
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def _init_dotsocr_model():
    """Lazy initialization of dots.ocr model."""
    global dotsocr_model, dotsocr_processor, DOTSOCR_AVAILABLE, DOTSOCR_ERROR_MESSAGE
    
    if not DOTSOCR_AVAILABLE:
        msg = DOTSOCR_ERROR_MESSAGE or "dots.ocr not available. Install with: pip install qwen-vl-utils"
        raise RuntimeError(msg)
    
    try:
        _ensure_dotsocr_dependencies()
    except Exception as dep_error:
        raise RuntimeError(str(dep_error))
    
    if dotsocr_model is None or dotsocr_processor is None:
        try:
            model_path = "rednote-hilab/dots.ocr"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Check transformers version
            try:
                import transformers
                transformers_version = transformers.__version__
                # dots.ocr may require transformers >= 4.47.0 for Qwen2_5_VLProcessor
                try:
                    from packaging import version
                    if version.parse(transformers_version) < version.parse("4.47.0"):
                        DOTSOCR_AVAILABLE = False
                        DOTSOCR_ERROR_MESSAGE = f"dots.ocr requires transformers >= 4.47.0, but you have {transformers_version}. Please upgrade: pip install transformers>=4.47.0"
                        raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
                except ImportError:
                    # packaging not available, try to check version manually
                    version_parts = transformers_version.split('.')
                    if len(version_parts) >= 2:
                        major, minor = int(version_parts[0]), int(version_parts[1])
                        if major < 4 or (major == 4 and minor < 47):
                            DOTSOCR_AVAILABLE = False
                            DOTSOCR_ERROR_MESSAGE = f"dots.ocr requires transformers >= 4.47.0, but you have {transformers_version}. Please upgrade: pip install transformers>=4.47.0"
                            raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
            except RuntimeError:
                # Re-raise version check errors
                raise
            except Exception:
                # If version check fails, continue anyway
                pass
            
            # Check for flash attention
            flash_attn_available = False
            try:
                importlib.import_module('flash_attn')
                flash_attn_available = True
            except:
                pass
            
            # Try loading with flash attention first, fallback to default if it fails
            try:
                dotsocr_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation="flash_attention_2" if flash_attn_available else None,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                ).eval()
            except (ImportError, AttributeError) as e:
                error_str = str(e)
                # If flash attention fails (e.g., LlamaFlashAttention2 not available), try without it
                if "LlamaFlashAttention2" in error_str or "flash_attention" in error_str.lower() or "cannot import name" in error_str:
                    warnings.warn(f"Flash attention not available for dots.ocr, falling back to default attention: {error_str}")
                    dotsocr_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        attn_implementation=None,  # Use default attention
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    ).eval()
                else:
                    raise
            
            if not torch.cuda.is_available():
                dotsocr_model.to(device)
            
            try:
                dotsocr_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except (ImportError, AttributeError, TypeError) as e:
                error_str = str(e)
                # Handle video_processor error - use manual patch (skip gated repo)
                if "video_processor" in error_str or "BaseVideoProcessor" in error_str:
                    # Try manual patch first (skip gated repo to avoid access issues)
                        try:
                            from transformers import Qwen2_5_VLProcessor
                            from transformers import AutoImageProcessor, AutoTokenizer
                            image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
                            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                            # Create processor manually with video_processor=None
                            # Qwen2_5_VLProcessor may require video_processor, so we create a dummy one if needed
                            try:
                                dotsocr_processor = Qwen2_5_VLProcessor(
                                    image_processor=image_processor,
                                    tokenizer=tokenizer
                                )
                            except TypeError:
                            # If video_processor is required, try to create a minimal one or pass None explicitly
                                # Some versions may accept None, others may need a dummy processor
                                try:
                                    # Try with explicit None
                                    dotsocr_processor = Qwen2_5_VLProcessor(
                                        image_processor=image_processor,
                                        tokenizer=tokenizer,
                                        video_processor=None
                                    )
                                except (TypeError, ValueError):
                                    # Last resort: try to create processor by patching the class signature
                                    # Check if we can inspect the __init__ signature
                                    import inspect
                                    try:
                                        sig = inspect.signature(Qwen2_5_VLProcessor.__init__)
                                        # If video_processor has a default, we can call it
                                        params = sig.parameters
                                        if 'video_processor' in params and params['video_processor'].default is not inspect.Parameter.empty:
                                            # video_processor has a default, call normally
                                            dotsocr_processor = Qwen2_5_VLProcessor(
                                                image_processor=image_processor,
                                                tokenizer=tokenizer
                                            )
                                        else:
                                            # Need to patch the class to accept None
                                            # Create processor without video_processor argument
                                            dotsocr_processor = Qwen2_5_VLProcessor.__new__(Qwen2_5_VLProcessor)
                                            # Try to call parent __init__ if available
                                            from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
                                            if hasattr(Qwen2VLProcessor, '__init__'):
                                                try:
                                                    Qwen2VLProcessor.__init__(dotsocr_processor, image_processor=image_processor, tokenizer=tokenizer)
                                                except:
                                                    # Fallback to manual assignment
                                                    dotsocr_processor.image_processor = image_processor
                                                    dotsocr_processor.tokenizer = tokenizer
                                            else:
                                                dotsocr_processor.image_processor = image_processor
                                                dotsocr_processor.tokenizer = tokenizer
                                            # Set video_processor to None if the attribute exists
                                            if hasattr(dotsocr_processor, 'video_processor'):
                                                dotsocr_processor.video_processor = None
                                    except Exception:
                                        # Final fallback: create processor without video_processor argument
                                        dotsocr_processor = Qwen2_5_VLProcessor.__new__(Qwen2_5_VLProcessor)
                                        dotsocr_processor.image_processor = image_processor
                                        dotsocr_processor.tokenizer = tokenizer
                                        if hasattr(dotsocr_processor, 'video_processor'):
                                            dotsocr_processor.video_processor = None
                        except Exception as patch_error:
                            DOTSOCR_AVAILABLE = False
                            DOTSOCR_ERROR_MESSAGE = f"dots.ocr processor initialization failed with video_processor error. Manual patch failed. Original error: {error_str}, Patch error: {str(patch_error)}. Note: The gated repo fix is not accessible. Please ensure you have transformers >= 4.47.0 installed."
                            raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
                elif "LlamaFlashAttention2" in error_str or "cannot import name" in error_str:
                    # If processor loading fails due to flash attention, it's likely a transformers version issue
                    DOTSOCR_AVAILABLE = False
                    DOTSOCR_ERROR_MESSAGE = f"dots.ocr processor initialization failed. This may require a newer transformers version. Error: {error_str}"
                    raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
                else:
                    raise
        except RuntimeError:
            # Re-raise RuntimeError as-is (from version check)
            raise
        except ImportError as e:
            error_str = str(e)
            if "Qwen2_5_VLProcessor" in error_str:
                DOTSOCR_AVAILABLE = False
                DOTSOCR_ERROR_MESSAGE = f"dots.ocr requires transformers >= 4.47.0 for Qwen2_5_VLProcessor. Current transformers version may be too old. Please upgrade: pip install transformers>=4.47.0. Error: {error_str}"
            elif "LlamaFlashAttention2" in error_str or "cannot import name" in error_str:
                # This error was already handled above, but if it reaches here, provide helpful message
                DOTSOCR_AVAILABLE = False
                DOTSOCR_ERROR_MESSAGE = f"dots.ocr model initialization failed due to flash attention compatibility issue. This may require a newer transformers version or disabling flash attention. Error: {error_str}"
            else:
                DOTSOCR_AVAILABLE = False
                DOTSOCR_ERROR_MESSAGE = f"dots.ocr model initialization failed: {error_str}"
            raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
        except Exception as e:
            error_str = str(e)
            if "video_processor" in error_str or "BaseVideoProcessor" in error_str:
                # Try manual patch for processor (skip gated repo)
                try:
                    from transformers import Qwen2_5_VLProcessor
                    from transformers import AutoImageProcessor, AutoTokenizer
                    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    # Create processor manually
                    try:
                        dotsocr_processor = Qwen2_5_VLProcessor(
                            image_processor=image_processor,
                            tokenizer=tokenizer
                        )
                    except TypeError:
                        try:
                            dotsocr_processor = Qwen2_5_VLProcessor(
                                image_processor=image_processor,
                                tokenizer=tokenizer,
                                video_processor=None
                            )
                        except (TypeError, ValueError):
                            # Last resort: try to create processor by patching the class signature
                            import inspect
                            try:
                                sig = inspect.signature(Qwen2_5_VLProcessor.__init__)
                                params = sig.parameters
                                if 'video_processor' in params and params['video_processor'].default is not inspect.Parameter.empty:
                                    dotsocr_processor = Qwen2_5_VLProcessor(
                                        image_processor=image_processor,
                                        tokenizer=tokenizer
                                    )
                                else:
                                    # Create processor without video_processor argument
                                    dotsocr_processor = Qwen2_5_VLProcessor.__new__(Qwen2_5_VLProcessor)
                                    from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
                                    if hasattr(Qwen2VLProcessor, '__init__'):
                                        try:
                                            Qwen2VLProcessor.__init__(dotsocr_processor, image_processor=image_processor, tokenizer=tokenizer)
                                        except Exception:
                                            dotsocr_processor.image_processor = image_processor
                                            dotsocr_processor.tokenizer = tokenizer
                                    else:
                                        dotsocr_processor.image_processor = image_processor
                                        dotsocr_processor.tokenizer = tokenizer
                                    if hasattr(dotsocr_processor, 'video_processor'):
                                        dotsocr_processor.video_processor = None
                            except Exception:
                                # Final fallback
                                dotsocr_processor = Qwen2_5_VLProcessor.__new__(Qwen2_5_VLProcessor)
                                dotsocr_processor.image_processor = image_processor
                                dotsocr_processor.tokenizer = tokenizer
                                if hasattr(dotsocr_processor, 'video_processor'):
                                    dotsocr_processor.video_processor = None
                except Exception as patch_error:
                    DOTSOCR_AVAILABLE = False
                    DOTSOCR_ERROR_MESSAGE = f"dots.ocr model initialization failed with video_processor error. Manual patch failed. Original error: {error_str}, Patch error: {str(patch_error)}. Note: The gated repo fix is not accessible. Please ensure you have transformers >= 4.47.0 installed."
                    raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
            elif "Qwen2_5_VLProcessor" in error_str:
                DOTSOCR_AVAILABLE = False
                DOTSOCR_ERROR_MESSAGE = f"dots.ocr requires transformers >= 4.47.0 for Qwen2_5_VLProcessor. Current transformers version may be too old. Please upgrade: pip install transformers>=4.47.0. Error: {error_str}"
                raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
            elif "LlamaFlashAttention2" in error_str or "cannot import name" in error_str:
                # This error was already handled above, but if it reaches here, provide helpful message
                DOTSOCR_AVAILABLE = False
                DOTSOCR_ERROR_MESSAGE = f"dots.ocr model initialization failed due to flash attention compatibility issue. This may require a newer transformers version or disabling flash attention. Error: {error_str}"
                raise RuntimeError(DOTSOCR_ERROR_MESSAGE)
            else:
                DOTSOCR_AVAILABLE = False
                DOTSOCR_ERROR_MESSAGE = f"dots.ocr model initialization failed: {error_str}"
                raise RuntimeError(DOTSOCR_ERROR_MESSAGE)

def process_image_olmocr(image, prompt=None):
    """Process image using olmOCR and return results in Markdown format.
    
    Args:
        image: PIL Image to process
        prompt: Optional custom prompt. If None, uses default document parsing prompt.
    """
    if image is None:
        return " Error Upload image", "", "", None, []
    
    # Lazy init to avoid import-time errors
    global olmocr_model, olmocr_processor, OLMOCR_AVAILABLE, OLMOCR_ERROR_MESSAGE
    if not OLMOCR_AVAILABLE:
        msg = OLMOCR_ERROR_MESSAGE or "olmOCR not available. Install with: pip install git+https://github.com/allenai/olmocr.git (requires Python >=3.11)"
        return f" {msg}", "", "", None, []
    
    try:
        _init_olmocr_model()
    except RuntimeError as e:
        return f" {str(e)}", "", "", None, []
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    # Resize image so longest dimension is 1288 pixels
    image_resized = _resize_image_for_olmocr(image, target_longest_dim=1288)
    
    # Build the prompt
    if prompt:
        text_prompt = prompt
    else:
        text_prompt = build_no_anchoring_v4_yaml_prompt()
    
    # Convert image to base64
    buf = BytesIO()
    image_resized.save(buf, format='PNG')
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    
    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    
    try:
        # Apply chat template and processor
        text = olmocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
        
        inputs = olmocr_processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for (key, value) in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            output = olmocr_model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=2048,
                num_return_sequences=1,
                do_sample=True,
            )
        
        # Decode output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = olmocr_processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        result = text_output[0] if text_output else ""
        
        if not result or not result.strip():
            return "No text detected", "", "", None, []
        
        # Extract markdown from YAML frontmatter if present
        # olmOCR output format: ---\n...\n---\n<content>
        if result.startswith("---"):
            parts = result.split("---", 2)
            if len(parts) >= 3:
                result = parts[2].strip()
        
        # Return same content for text, markdown, and raw
        return result, result, result, None, []
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        warnings.warn(error_msg)
        return f"Error: {str(e)}", "", "", None, []

def process_image_dotsocr(image, prompt=None):
    """Process image using dots.ocr and return results in Markdown format.
    
    Args:
        image: PIL Image to process
        prompt: Optional custom prompt. If None, uses default document parsing prompt.
    """
    if image is None:
        return " Error Upload image", "", "", None, []
    
    # Lazy init to avoid import-time errors
    global dotsocr_model, dotsocr_processor, DOTSOCR_AVAILABLE, DOTSOCR_ERROR_MESSAGE
    if not DOTSOCR_AVAILABLE:
        msg = DOTSOCR_ERROR_MESSAGE or "dots.ocr not available. Install with: pip install qwen-vl-utils"
        return f" {msg}", "", "", None, []
    
    try:
        _init_dotsocr_model()
    except RuntimeError as e:
        return f" {str(e)}", "", "", None, []
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    # Save image to temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    
    # Build the prompt
    if prompt:
        text_prompt = prompt
    else:
        # Default prompt for dots.ocr document parsing
        text_prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""
    
    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": tmp.name
                },
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    try:
        # Preparation for inference
        text = dotsocr_processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = dotsocr_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for (key, value) in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            generated_ids = dotsocr_model.generate(**inputs, max_new_tokens=24000)
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = dotsocr_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        result = output_text[0] if output_text else ""
        
        # Clean up temp file
        os.unlink(tmp.name)
        
        if not result or not result.strip():
            return "No text detected", "", "", None, []
        
        # Try to parse JSON and convert to markdown
        try:
            layout_data = json.loads(result)
            
            # Convert layout data to markdown
            markdown_parts = []
            if isinstance(layout_data, dict) and "layout" in layout_data:
                layout_items = layout_data["layout"]
            elif isinstance(layout_data, list):
                layout_items = layout_data
            else:
                layout_items = [layout_data]
            
            for item in layout_items:
                if isinstance(item, dict):
                    category = item.get("category", "")
                    text_content = item.get("text", "")
                    bbox = item.get("bbox", [])
                    
                    if category == "Title":
                        markdown_parts.append(f"# {text_content}")
                    elif category == "Section-header":
                        markdown_parts.append(f"## {text_content}")
                    elif category == "Table":
                        markdown_parts.append(text_content)  # Already HTML formatted
                    elif category == "Formula":
                        markdown_parts.append(f"$${text_content}$$")  # LaTeX formula
                    elif category == "Picture":
                        markdown_parts.append(f"![Image](bbox: {bbox})")
                    else:
                        markdown_parts.append(text_content)
            
            markdown_result = "\n\n".join(markdown_parts)
            return markdown_result, markdown_result, result, None, []
        except json.JSONDecodeError:
            # If not JSON, return as-is
            return result, result, result, None, []
    
    except Exception as e:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except:
            pass
        
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
    global paddleocrvl_pipeline, PADDLEOCRVL_AVAILABLE, PADDLEOCRVL_ERROR_MESSAGE, PaddleOCRVL
    if PaddleOCR is None or PaddleOCRVL is None:
        try:
            _import_paddleocr()
        except Exception as e:
            PADDLEOCRVL_AVAILABLE = False
            PADDLEOCRVL_ERROR_MESSAGE = f"PaddleOCR-VL setup failed: {e}"
    # Early exit if engine is unavailable
    if not PADDLEOCRVL_AVAILABLE or paddleocrvl_pipeline is None or PaddleOCRVL is None:
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
    global paddleocrvl_pipeline, PADDLEOCRVL_AVAILABLE, PADDLEOCRVL_ERROR_MESSAGE, PaddleOCRVL
    if PaddleOCR is None or PaddleOCRVL is None:
        try:
            _import_paddleocr()
        except Exception as e:
            PADDLEOCRVL_AVAILABLE = False
            PADDLEOCRVL_ERROR_MESSAGE = f"PaddleOCR-VL setup failed: {e}"
    # Early exit if engine is unavailable
    if not PADDLEOCRVL_AVAILABLE or paddleocrvl_pipeline is None or PaddleOCRVL is None:
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

def process_pdf_olmocr(path, dpi=300, page_indices=None, insert_separators=True, max_retries=3, retry_backoff_seconds=3):
    """Process PDF using olmOCR."""
    # Early exit if engine is unavailable
    if not OLMOCR_AVAILABLE:
        msg = OLMOCR_ERROR_MESSAGE or "olmOCR not available. Install with: pip install git+https://github.com/allenai/olmocr.git (requires Python >=3.11)"
        return msg, f"<!-- {msg} -->", msg, None, []
    
    try:
        _init_olmocr_model()
    except RuntimeError as e:   
        return str(e), f"<!-- {str(e)} -->", str(e), None, []
    
    doc = fitz.open(path)
    texts, markdowns, raws, all_crops = [], [], [], []
    if page_indices is None:
        page_indices = list(range(len(doc)))
    
    for i in page_indices:
        cache_key = ("olmOCR", path, int(dpi), int(i))
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
                text, md, raw, _, crops = process_image_olmocr(img)
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

def process_pdf_all_olmocr(path, dpi=300, page_range_text="", insert_separators=True, batch_size=3, max_retries=5, retry_backoff_seconds=5):
    """Process all pages of PDF using olmOCR."""
    # Early exit if engine is unavailable
    if not OLMOCR_AVAILABLE:
        msg = OLMOCR_ERROR_MESSAGE or "olmOCR not available. Install with: pip install git+https://github.com/allenai/olmocr.git (requires Python >=3.11)"
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
                tx, mdx, rawx, _, cropsx = process_pdf_olmocr(path, dpi=dpi, page_indices=batch, insert_separators=insert_separators)
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

def process_pdf_dotsocr(path, dpi=300, page_indices=None, insert_separators=True, max_retries=3, retry_backoff_seconds=3):
    """Process PDF using dots.ocr."""
    # Early exit if engine is unavailable
    if not DOTSOCR_AVAILABLE:
        msg = DOTSOCR_ERROR_MESSAGE or "dots.ocr not available. Install with: pip install qwen-vl-utils"
        return msg, f"<!-- {msg} -->", msg, None, []
    
    try:
        _init_dotsocr_model()
    except RuntimeError as e:
        return str(e), f"<!-- {str(e)} -->", str(e), None, []
    
    doc = fitz.open(path)
    texts, markdowns, raws, all_crops = [], [], [], []
    if page_indices is None:
        page_indices = list(range(len(doc)))
    
    for i in page_indices:
        cache_key = ("dots.ocr", path, int(dpi), int(i))
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
                text, md, raw, _, crops = process_image_dotsocr(img)
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

def process_pdf_all_dotsocr(path, dpi=300, page_range_text="", insert_separators=True, batch_size=3, max_retries=5, retry_backoff_seconds=5):
    """Process all pages of PDF using dots.ocr."""
    # Early exit if engine is unavailable
    if not DOTSOCR_AVAILABLE:
        msg = DOTSOCR_ERROR_MESSAGE or "dots.ocr not available. Install with: pip install qwen-vl-utils"
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
                tx, mdx, rawx, _, cropsx = process_pdf_dotsocr(path, dpi=dpi, page_indices=batch, insert_separators=insert_separators)
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
    elif ocr_engine == "olmOCR":
        if path.lower().endswith('.pdf'):
            return process_pdf_all_olmocr(path, dpi=dpi, page_range_text=page_range_text, insert_separators=insert_separators)
        else:
            return process_image_olmocr(Image.open(path))
    elif ocr_engine == "dots.ocr":
        if path.lower().endswith('.pdf'):
            return process_pdf_all_dotsocr(path, dpi=dpi, page_range_text=page_range_text, insert_separators=insert_separators)
        else:
            return process_image_dotsocr(Image.open(path))
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
                    choices=[c for c in ["DeepSeekOCR", "PaddleOCR-VL", "olmOCR", "dots.ocr",  "Gemini Flash 2.5"] if (c != "PaddleOCR-VL" or PADDLEOCRVL_AVAILABLE) and (c != "Gemini Flash 2.5" or GEMINI_AVAILABLE) and (c != "olmOCR" or OLMOCR_AVAILABLE) and (c != "dots.ocr" or DOTSOCR_AVAILABLE)],
                    value="DeepSeekOCR",
                    label="OCR Engine",
                    info="Choose between DeepSeekOCR, PaddleOCR-VL, olmOCR, or dots.ocr, Gemini Flash 2.5"
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
            - **olmOCR**: Vision-language model for document OCR (requires Python >=3.11)
            - **dots.ocr**: Multilingual document parser with SOTA performance on layout detection and content recognition (install with: `pip install qwen-vl-utils`)
            
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
            
            ### olmOCR
            - Vision-language model based on Qwen2.5-VL-7B-Instruct
            - Automatically converts documents to markdown format
            - Supports both images and PDFs
            - Model: allenai/olmOCR-2-7B-1025
            - **Requires Python >=3.11** - For Hugging Face Spaces, create a `runtime.txt` file with `python-3.11` or higher
            
            ### dots.ocr
            - Multilingual document parser based on 1.7B LLM with SOTA performance
            - Achieves state-of-the-art results for text, tables, and reading order
            - Supports both images and PDFs
            - Model: rednote-hilab/dots.ocr
            - **Requires transformers >= 4.47.0** - Please upgrade transformers if you see import errors

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
            if engine in ["PaddleOCR-VL", "Gemini Flash 2.5", "olmOCR", "dots.ocr"]:
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
            elif ocr_engine_val == "olmOCR":
                # olmOCR processing
                if fp and isinstance(fp, str) and fp.lower().endswith('.pdf'):
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="olmOCR")
                elif image is not None:
                    text, md, raw, img, crops = process_image_olmocr(image)
                elif fp:
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="olmOCR")
                else:
                    return "Error uploading file or image", "", "", None, [], None, None, None
            elif ocr_engine_val == "dots.ocr":
                # dots.ocr processing
                if fp and isinstance(fp, str) and fp.lower().endswith('.pdf'):
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="dots.ocr")
                elif image is not None:
                    text, md, raw, img, crops = process_image_dotsocr(image)
                elif fp:
                    text, md, raw, img, crops = process_file(fp, mode_label, task_label, custom_prompt, dpi=int(dpi_val), page_range_text=page_range_text, embed_figures=embed, high_accuracy=hiacc, insert_separators=sep_pages, ocr_engine="dots.ocr")
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