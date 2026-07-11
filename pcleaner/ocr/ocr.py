from pathlib import Path
from typing import Protocol, TypeAlias, Callable
from io import StringIO
import itertools
import csv

from PIL import Image
from natsort import natsorted
from loguru import logger

import pcleaner.config as cfg
import pcleaner.structures as st
import pcleaner.helpers as hp
from pcleaner.ocr.ocr_mangaocr import MangaOcr
from pcleaner.ocr.ocr_tesseract import TesseractOcr
import pcleaner.ocr.supported_languages as osl
from pcleaner.ocr.ocr_advanced import (
    create_ocr_engine,
    is_pure_punctuation,
    INTERNAL_OCR_DIR,
)


class OCRModel(Protocol):
    def __call__(self, img_or_path: Image.Image | Path | str, **kwargs) -> str: ...


OCREngineFactory: TypeAlias = Callable[[osl.LanguageCode], OCRModel]


def tesseract_ok(profile: cfg.Profile) -> bool:
    want_tess = profile.preprocessor.ocr_use_tesseract
    has_language_packs: bool = bool(TesseractOcr.langs())
    return not want_tess or (want_tess and has_language_packs)


def ocr_engines():
    return {
        cfg.OCREngine.MANGAOCR: MangaOcr(),
        cfg.OCREngine.TESSERACT: TesseractOcr(),
    }


def get_all_available_langs() -> set[osl.LanguageCode]:
    union_langs = set()
    for ocr in ocr_engines().values():
        union_langs |= ocr.langs()
    return union_langs


def build_ocr_engine_factory(
    tesseract_enabled: bool, ocr_engine_preference: cfg.OCREngine,
    ocr_model_name: str | None = None,
) -> OCREngineFactory:
    """
    Create a factory function that returns the appropriate OCR engine for a given language.

    Supports both legacy engines (MangaOCR, Tesseract) and advanced engines (EasyOCR, PaddleOCR).

    :param tesseract_enabled: Whether Tesseract should be considered for use.
    :param ocr_engine_preference: The preferred OCR engine.
    :return: The factory function.
    """
    # Legacy engine language support
    tess_langs: set[osl.LanguageCode] = TesseractOcr.langs()
    mocr_langs: set[osl.LanguageCode] = MangaOcr.langs()
    
    if tesseract_enabled and not tess_langs:
        logger.error(
            f"Tesseract OCR is not installed or not found. "
            "Please see the instructions in the README to install Tesseract correctly. Falling back to manga-ocr."
        )

    if not tesseract_enabled:
        ocr_engine_preference = cfg.OCREngine.MANGAOCR

    # Apply profile model name to MangaOcr singleton so it's used when initialized
    if ocr_model_name:
        MangaOcr().set_model_name(ocr_model_name)

    # Initialize advanced engines once at factory creation (if selected)
    advanced_engine = None
    if ocr_engine_preference in (cfg.OCREngine.EASYOCR, cfg.OCREngine.PADDLEOCR):
        try:
            advanced_engine = create_ocr_engine(ocr_engine_preference.value)
            logger.info(f"✅ Advanced OCR Engine initialized: {ocr_engine_preference.value}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {ocr_engine_preference.value}: {e}. Falling back to MangaOCR.")
            ocr_engine_preference = cfg.OCREngine.MANGAOCR
            advanced_engine = None

    def closure(lang: osl.LanguageCode) -> OCRModel:
        """Factory closure that selects the appropriate OCR engine for a language."""
        
        # Advanced OCR engines: language-agnostic (handle multi-lingual automatically)
        if ocr_engine_preference == cfg.OCREngine.EASYOCR and advanced_engine:
            logger.debug(f"🚀 Using EasyOCR for language: {lang}")
            return advanced_engine
        
        if ocr_engine_preference == cfg.OCREngine.PADDLEOCR and advanced_engine:
            logger.debug(f"🚀 Using PaddleOCR for language: {lang}")
            return advanced_engine
        
        # AUTO mode: select best engine based on language
        if ocr_engine_preference == cfg.OCREngine.AUTO:
            # Prefer MangaOCR for Japanese (its specialty)
            if lang in mocr_langs:
                return MangaOcr()
            # Try Tesseract for other languages
            if lang in tess_langs:
                return TesseractOcr(lang)
            # Fall back to MangaOCR
            return MangaOcr()
        
        # Explicit Tesseract selection
        elif ocr_engine_preference == cfg.OCREngine.TESSERACT:
            if lang in tess_langs:
                return TesseractOcr(lang)
            logger.error(
                f"Tesseract language pack for '{lang}' is not installed or not found. "
                "Please see the instructions to install Tesseract correctly. Falling back to manga-ocr."
            )
        
        # Fall back to manga-ocr for any unhandled case
        return MangaOcr()

    return closure


def format_output(
    ocr_analytics: list[st.OCRAnalytic],
    csv_output: bool,
    csv_column_names: tuple[str, str, str, str, str, str],
) -> str:
    """
    Format the output of the OCR process.

    :param ocr_analytics: A list of the OCR analytics per image.
    :param csv_output: If True, output the data in CSV format, otherwise in plain text.
    :param csv_column_names: The (localized) names of the columns in the CSV output:
        filename, startx, starty, endx, endy, text.
    :return: The formatted output.
    """
    # Format of the analytics:
    # number of boxes | sizes of all boxes | sizes of boxes that were OCRed | path to image, text, box coordinates
    # We do not need to show the first three columns, so we simplify the data structure.
    # path_texts_coords: list[tuple[Path, str, st.Box]] = list(
    #     itertools.chain.from_iterable(a.removed_box_data for a in ocr_analytics)
    # )

    # Build tuples of the form (path, text) for the removed texts.
    # This requires adding the path from the analytic to all the texts.
    path_texts_coords: list[tuple[Path, str, st.Box]] = []
    for analytic in ocr_analytics:
        for text, box in analytic.removed_box_data:
            path_texts_coords.append((analytic.path, text, box))

    if path_texts_coords:
        paths, texts, boxes = zip(*path_texts_coords)
        paths = hp.trim_prefix_from_paths(paths)
        path_texts_coords = list(zip(paths, texts, boxes))
        # Sort by path.
        path_texts_coords = natsorted(path_texts_coords, key=lambda x: x[0])

    if csv_output:
        return format_output_csv(path_texts_coords, csv_column_names)
    return format_output_plain(path_texts_coords)


def format_output_csv(
    path_texts_coords: list[tuple[Path, str, st.Box]],
    csv_column_names: tuple[str, str, str, str, str, str],
) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(csv_column_names)

    for path, bubble, box in path_texts_coords:
        if "\n" in bubble:
            logger.warning(f"Detected newline in bubble: {path} {bubble} {box}")
            bubble = bubble.replace("\n", "\\n")
        writer.writerow([path, *box.as_tuple, bubble])

    return buffer.getvalue()


def format_output_plain(path_texts_coords: list[tuple[Path, str, st.Box]]) -> str:
    # Place the file path on it's own line, and only if it's different from the previous one.
    buffer = StringIO()
    current_path = ""
    for path, bubble, _ in path_texts_coords:
        if path != current_path:
            buffer.write(f"\n\n{path}: ")
            current_path = path
        buffer.write(f"\n{bubble}")
        if "\n" in bubble:
            logger.warning(f"Detected newline in bubble: {path} {bubble}")

    return buffer.getvalue()
