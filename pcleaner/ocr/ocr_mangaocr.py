from pathlib import Path

from loguru import logger
from manga_ocr import MangaOcr as MangaOcrModel
from PIL import Image

import pcleaner.ocr.supported_languages as osl


class MangaOcr:
    _instance = None
    _model = None
    _init_args = ((), {})

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Creating the MangaOcr instance")
            cls._instance = super(MangaOcr, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._init_args = (args, kwargs)  # store just in case
            cls._instance._model_name = None
        return cls._instance

    def __init__(self, *args, **kwargs):
        # Store model_name if given at construction
        if args:
            self._model_name = args[0]
        elif "model_name" in kwargs:
            self._model_name = kwargs["model_name"]

    def set_model_name(self, model_name: str) -> None:
        """Change the OCR model name and force reinitialization on next use."""
        self._model_name = model_name
        self._model = None  # force reload with new model

    @staticmethod
    def langs() -> set[osl.LanguageCode]:
        return {osl.LanguageCode.jpn}

    def initialize_model(self, *args, **kwargs):
        if self._model is None:
            model_name = kwargs.pop("model_name", None) or self._model_name
            if model_name:
                self._model = MangaOcrModel(model_name)
            elif args:
                self._model = MangaOcrModel(*args, **kwargs)
            else:
                init_args = self._init_args
                self._model = MangaOcrModel(*init_args[0], **init_args[1])
        return self._model

    def __call__(self, img_or_path: Image.Image | Path | str, **kwargs) -> str:
        model = self.initialize_model()
        return model(img_or_path)
