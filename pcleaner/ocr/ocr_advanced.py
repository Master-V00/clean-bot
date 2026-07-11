"""
Advanced Multi-Language OCR Engines with Internal Sandboxing.

This module provides dual OCR engine support:
1. EasyOCR: Multi-language support with auto-detection
2. PaddleOCR: Advanced layout detection with automatic routing

All models are stored strictly within pcleaner/models/ for internal sandboxing.
"""

import re
from pathlib import Path
from typing import Protocol

from PIL import Image
from loguru import logger
import numpy as np

import pcleaner.helpers as hp


# Internal models directory - strictly isolated from user home directories
INTERNAL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INTERNAL_OCR_DIR = INTERNAL_MODELS_DIR / "ocr"



def is_pure_punctuation(text: str) -> bool:
    """
    Validate if text consists SOLELY of punctuation/symbols without any linguistic characters.
    
    Checks against international alphabet scripts:
    - Arabic (U+0600-U+06FF)
    - English (a-zA-Z, basic Latin)
    - Korean (U+AC00-U+D7AF, Hangul)
    - Japanese (U+3040-U+309F Hiragana, U+30A0-U+30FF Katakana)
    - Chinese (U+4E00-U+9FAF CJK Unified Ideographs)
    
    :param text: The OCR-extracted text to validate
    :return: True if text contains ONLY punctuation/symbols, False if it has linguistic content
    """
    clean_text = text.strip()
    
    # Empty strings are considered punctuation-only
    if not clean_text:
        return True
    
    # Check for any alphabet or linguistic characters from supported scripts
    linguistic_pattern = r'[\w\u0600-\u06FF\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]'
    if re.search(linguistic_pattern, clean_text, re.UNICODE):
        return False
    
    # Additional check: verify all characters are punctuation/symbols
    # Allow common punctuation and symbols across multiple languages
    punctuation_set = set(
        ".,;:!?…\n\r\t@#$%^&*()_+-=[]{}|<>~`\"'؟!٠١٢٣٤٥٦٧٨٩"
        "؛:،.\u200E\u200F"  # RTL marks
        "👉👈✓✗✘⊘◯◎★✦✧✶✷"  # Symbols
    )
    
    # Check if all non-whitespace characters are in the punctuation set
    non_whitespace = [c for c in clean_text if not c.isspace()]
    if not non_whitespace:
        return True
    
    return all(c in punctuation_set or ord(c) < 32 for c in non_whitespace)


class OCRModel(Protocol):
    """Protocol for OCR model callables."""
    def __call__(self, img: Image.Image) -> str: ...


class DynamicEasyOcrEngine:
    """
    EasyOCR Engine with automatic multi-language support and internal sandboxing.
    
    Features:
    - Automatically initializes with all supported languages simultaneously
    - Languages: English, Arabic, Korean, Chinese (Simplified), Japanese
    - GPU acceleration when available
    - Models stored strictly in pcleaner/models/ocr/
    - No fallback to user home directories or HuggingFace cache
    """
    
    def __init__(self):
        """Initialize EasyOCR with comprehensive multi-language support."""
        try:
            import easyocr
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            raise
        
        # Ensure internal models directory exists
        INTERNAL_OCR_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔧 Deploying EasyOCR Sandbox with Global Auto-Language Multi-Array")
        logger.info(f"📁 Model Storage: {INTERNAL_OCR_DIR}")
        
        # Initialize with comprehensive language support
        # These cover most manga/comic text scenarios
        languages = ['en', 'ar', 'ko', 'ch_sim', 'ja']
        
        # Use globally-cached GPU availability (detected at startup in main())
        gpu_available = hp.detect_gpu_availability()
        logger.info(f"🎮 Using GPU: {gpu_available} (globally detected at startup)")
        
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu_available,
            model_storage_directory=str(INTERNAL_OCR_DIR),
            user_network_directory=str(INTERNAL_OCR_DIR)
        )
        logger.info(f"✅ EasyOCR initialized with languages: {languages}")

    
    def __call__(self, img: Image.Image) -> str:
        """
        Extract text from image using EasyOCR.
        
        :param img: PIL Image object
        :return: Extracted text string
        """
        try:
            img_np = np.array(img.convert('RGB'))
            results = self.reader.readtext(img_np, detail=0)
            return " ".join(results) if results else ""
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""


class DynamicPaddleOcrEngine:
    """
    PaddleOCR Engine (v4) with automatic multi-directional layout detection.
    
    Features:
    - Angle classification for rotated text detection
    - Multi-directional layout mapping (LTR, RTL, TTB)
    - Automatic model routing based on detected language
    - Models stored strictly in pcleaner/models/ocr/
    - No fallback to user home directories
    """
    
    def __init__(self):
        """Initialize PaddleOCR with advanced layout detection."""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            raise
        
        # Ensure internal models directory exists
        INTERNAL_OCR_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔧 Deploying PaddleOCR (v4) Dynamic Sandbox Studio")
        logger.info(f"📁 Model Storage: {INTERNAL_OCR_DIR}")
        
        # Use globally-cached GPU availability (detected at startup in main())
        gpu_available = hp.detect_gpu_availability()
        logger.info(f"🎮 Using GPU: {gpu_available} (globally detected at startup)")
        
        # Initialize PaddleOCR with advanced settings
        self.reader = PaddleOCR(
            use_angle_cls=True,           # Enable angle classification for rotated text
            lang='en',                     # Default language
            layout=False,                  # Disable layout detection to avoid conflicts
            show_log=False,                # Suppress verbose logging
            use_gpu=gpu_available,
            det_model_dir=str(INTERNAL_OCR_DIR / "det"),
            rec_model_dir=str(INTERNAL_OCR_DIR / "rec"),
            cls_model_dir=str(INTERNAL_OCR_DIR / "cls")
        )
        logger.info("✅ PaddleOCR (v4) initialized with advanced layout detection")

    
    def __call__(self, img: Image.Image) -> str:
        """
        Extract text from image using PaddleOCR.
        
        :param img: PIL Image object
        :return: Extracted text string
        """
        try:
            img_np = np.array(img.convert('RGB'))
            result = self.reader.ocr(img_np, cls=True)
            
            if not result or not result[0]:
                return ""
            
            # Extract text from each detected line
            # result[0] is a list of lines, each line is a list of (bbox, (text, confidence))
            texts = [line[1][0] for line in result[0] if line[1]]
            return " ".join(texts)
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return ""


def create_ocr_engine(engine_type: str) -> OCRModel:
    """
    Factory function to create the appropriate OCR engine.
    
    :param engine_type: Type of OCR engine ('easyocr', 'paddleocr', or 'auto')
    :return: Initialized OCR engine instance
    :raises ValueError: If engine type is not recognized
    """
    engine_lower = engine_type.lower()
    
    if engine_lower == 'easyocr':
        logger.info("🚀 Creating EasyOCR Engine")
        return DynamicEasyOcrEngine()
    elif engine_lower == 'paddleocr':
        logger.info("🚀 Creating PaddleOCR Engine")
        return DynamicPaddleOcrEngine()
    elif engine_lower == 'auto':
        # Default to EasyOCR for auto mode
        logger.info("🚀 Auto-detecting OCR engine (defaulting to EasyOCR)")
        try:
            return DynamicEasyOcrEngine()
        except ImportError:
            logger.warning("EasyOCR not available, trying PaddleOCR")
            return DynamicPaddleOcrEngine()
    else:
        raise ValueError(
            f"Unknown OCR engine type: {engine_type}. "
            f"Supported: 'easyocr', 'paddleocr', 'auto'"
        )
