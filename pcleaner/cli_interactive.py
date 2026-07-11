"""
Interactive CLI Runtime Controller for PanelCleaner.

Provides a numbered terminal menu for profile selection and real-time
parameter mutation before launching processing pipelines.
"""

from pathlib import Path
from typing import Optional
from loguru import logger

import pcleaner.config as cfg
import pcleaner.profile_cli as pc


class InteractiveCLIController:
    """
    Interactive CLI controller for profile and parameter selection.
    
    Features:
    - Numbered terminal menu for profile selection
    - Real-time parameter mutation (OCR engine, inpainting mode, etc.)
    - Pre-launch validation and confirmation
    """
    
    def __init__(self, config: cfg.Config):
        """
        Initialize the interactive CLI controller.
        
        :param config: Loaded PanelCleaner configuration
        """
        self.config = config
        self.selected_profile: Optional[cfg.Profile] = None
    
    def display_profile_menu(self) -> Optional[str]:
        """
        Display a numbered menu of available profiles and get user selection.
        
        :return: Selected profile name or None if no selection
        """
        # Get list of available profiles
        profile_names = list(self.config.saved_profiles.keys())
        
        if not profile_names:
            logger.warning("⚠️  No saved profiles available!")
            return None
        
        print("\n" + "="*60)
        print("📋 AVAILABLE PROFILES")
        print("="*60)
        
        for idx, name in enumerate(profile_names, 1):
            marker = " ✓" if name == self.config.current_profile.name else ""
            print(f"  {idx:2d}. {name}{marker}")
        
        print("="*60)
        print("  0. Use current profile")
        print("="*60)
        
        while True:
            try:
                choice = input("\n🔍 Select profile (0-{}): ".format(len(profile_names))).strip()
                choice_int = int(choice)
                
                if choice_int == 0:
                    return self.config.current_profile.name
                elif 1 <= choice_int <= len(profile_names):
                    return profile_names[choice_int - 1]
                else:
                    print("❌ Invalid selection. Try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def display_parameter_menu(self, profile: cfg.Profile) -> None:
        """
        Display a menu to mutate runtime parameters before processing.
        
        :param profile: The selected profile to modify
        """
        while True:
            print("\n" + "="*60)
            print("⚙️  RUNTIME PARAMETERS")
            print("="*60)
            print(f"  1. OCR Engine: {profile.preprocessor.ocr_engine}")
            print(f"  2. Full LaMa Inpainting: {getattr(profile.inpainter, 'full_lama_inpainting', False)}")
            print(f"  3. Denoising Enabled: {profile.denoiser.denoising_enabled}")
            print(f"  4. Inpainting Enabled: {profile.inpainter.inpainting_enabled}")
            print("="*60)
            print("  0. Proceed with processing")
            print("  q. Quit")
            print("="*60)
            
            choice = input("\n🔧 Select parameter to modify (0-4/q): ").strip().lower()
            
            if choice == '0':
                logger.info("✅ Proceeding with pipeline execution...")
                break
            elif choice == 'q':
                raise KeyboardInterrupt("User cancelled processing")
            elif choice == '1':
                self._mutate_ocr_engine(profile)
            elif choice == '2':
                self._mutate_full_lama(profile)
            elif choice == '3':
                self._mutate_denoising(profile)
            elif choice == '4':
                self._mutate_inpainting(profile)
            else:
                print("❌ Invalid selection.")
    
    def _mutate_ocr_engine(self, profile: cfg.Profile) -> None:
        """Allow user to select OCR engine at runtime."""
        print("\n" + "-"*60)
        print("🔤 OCR ENGINE SELECTION")
        print("-"*60)
        print("  1. auto (Auto-detect)")
        print("  2. manga-ocr (MangaOCR)")
        print("  3. tesseract (Tesseract)")
        print("  4. easyocr (EasyOCR - Multi-lingual)")
        print("  5. paddleocr (PaddleOCR - Advanced Layout)")
        print("-"*60)
        
        try:
            choice = input("🔍 Select OCR engine (1-5): ").strip()
            engines = {
                '1': cfg.OCREngine.AUTO,
                '2': cfg.OCREngine.MANGAOCR,
                '3': cfg.OCREngine.TESSERACT,
                '4': cfg.OCREngine.EASYOCR,
                '5': cfg.OCREngine.PADDLEOCR,
            }
            
            if choice in engines:
                profile.preprocessor.ocr_engine = engines[choice]
                logger.info(f"✅ OCR Engine changed to: {profile.preprocessor.ocr_engine}")
            else:
                print("❌ Invalid selection.")
        except Exception as e:
            logger.error(f"Error changing OCR engine: {e}")
    
    def _mutate_full_lama(self, profile: cfg.Profile) -> None:
        """Allow user to toggle full LaMa inpainting mode."""
        print("\n" + "-"*60)
        print("🎨 FULL LAMA INPAINTING")
        print("-"*60)
        print("  1. Enable full LaMa inpainting")
        print("  2. Disable full LaMa inpainting")
        print("-"*60)
        
        try:
            choice = input("🔍 Select option (1-2): ").strip()
            if choice == '1':
                if hasattr(profile.inpainter, 'full_lama_inpainting'):
                    profile.inpainter.full_lama_inpainting = True
                logger.info("✅ Full LaMa Inpainting enabled")
            elif choice == '2':
                if hasattr(profile.inpainter, 'full_lama_inpainting'):
                    profile.inpainter.full_lama_inpainting = False
                logger.info("✅ Full LaMa Inpainting disabled")
            else:
                print("❌ Invalid selection.")
        except Exception as e:
            logger.error(f"Error toggling full LaMa: {e}")
    
    def _mutate_denoising(self, profile: cfg.Profile) -> None:
        """Allow user to toggle denoising."""
        print("\n" + "-"*60)
        print("🔊 DENOISING")
        print("-"*60)
        print(f"  Current: {profile.denoiser.denoising_enabled}")
        print("  1. Enable denoising")
        print("  2. Disable denoising")
        print("-"*60)
        
        try:
            choice = input("🔍 Select option (1-2): ").strip()
            if choice == '1':
                profile.denoiser.denoising_enabled = True
                logger.info("✅ Denoising enabled")
            elif choice == '2':
                profile.denoiser.denoising_enabled = False
                logger.info("✅ Denoising disabled")
            else:
                print("❌ Invalid selection.")
        except Exception as e:
            logger.error(f"Error toggling denoising: {e}")
    
    def _mutate_inpainting(self, profile: cfg.Profile) -> None:
        """Allow user to toggle inpainting."""
        print("\n" + "-"*60)
        print("🎯 INPAINTING")
        print("-"*60)
        print(f"  Current: {profile.inpainter.inpainting_enabled}")
        print("  1. Enable inpainting")
        print("  2. Disable inpainting")
        print("-"*60)
        
        try:
            choice = input("🔍 Select option (1-2): ").strip()
            if choice == '1':
                profile.inpainter.inpainting_enabled = True
                logger.info("✅ Inpainting enabled")
            elif choice == '2':
                profile.inpainter.inpainting_enabled = False
                logger.info("✅ Inpainting disabled")
            else:
                print("❌ Invalid selection.")
        except Exception as e:
            logger.error(f"Error toggling inpainting: {e}")
    
    def run_interactive_session(self) -> Optional[cfg.Profile]:
        """
        Run the interactive CLI session and return the configured profile.
        
        :return: The configured Profile object, or None if cancelled
        """
        print("\n" + "="*60)
        print("🚀 PANELCLEANER INTERACTIVE RUNTIME CONTROLLER")
        print("="*60)
        
        # Step 1: Profile selection
        profile_name = self.display_profile_menu()
        if profile_name is None:
            logger.warning("No profile selected. Using default.")
            profile_name = self.config.current_profile.name
        
        # Load selected profile
        try:
            self.config.load_profile(profile_name)
            self.selected_profile = self.config.current_profile
            logger.info(f"✅ Loaded profile: {profile_name}")
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return None

        # Step 2a: Inpainting interactive questions
        try:
            import pcleaner.model_downloader as md
            import os

            print("\n" + "-" * 60)
            print("🎨 LAMA Inpainting Setup")
            print("-" * 60)
            print("  1. Run LaMa on all bubbles/text (Full LaMa Inpainting)")
            print("  2. Run LaMa only on complex bubbles; use pixel method for simple text (Complex-only)")
            print("-" * 60)
            choice = input("Select inpainting mode (1-2, default 2): ").strip() or "2"
            if choice == '1':
                if hasattr(self.selected_profile.inpainter, 'full_lama_inpainting'):
                    self.selected_profile.general.full_lama_inpainting = True
                logger.info("✅ Full LaMa Inpainting selected")
            else:
                if hasattr(self.selected_profile.inpainter, 'full_lama_inpainting'):
                    self.selected_profile.general.full_lama_inpainting = False
                logger.info("✅ Complex-only LaMa Inpainting selected")

            # Present local inpainting models first
            local_models = list(md.INTERNAL_INPAINTING_DIR.glob("*.pt"))
            print("\nAvailable local LaMa models:")
            for idx, p in enumerate(local_models, 1):
                print(f"  {idx}. {p.name}")
            # Present remote options supported by code
            remote_keys = list(md.INPAINTING_ALTERNATIVES.keys())
            start_idx = len(local_models) + 1
            print("\nRemote models available for download:")
            for i, key in enumerate(remote_keys, start_idx):
                print(f"  {i}. {key} (download)")
            print(f"  0. Skip and use default model")

            while True:
                sel = input(f"Select model (0-{start_idx+len(remote_keys)-1}): ").strip()
                if sel == "0":
                    logger.info("Using default inpainting model (no change).")
                    break
                try:
                    sel_i = int(sel)
                    if 1 <= sel_i <= len(local_models):
                        chosen = local_models[sel_i - 1]
                        os.environ['PCL_INPAINTING_MODEL'] = str(chosen)
                        logger.info(f"Selected local inpainting model: {chosen.name}")
                        break
                    elif start_idx <= sel_i < start_idx + len(remote_keys):
                        chosen_key = remote_keys[sel_i - start_idx]
                        print(f"Downloading {chosen_key}...")
                        downloaded = md.download_named_inpainting_model(chosen_key, md.INTERNAL_INPAINTING_DIR)
                        if downloaded is None:
                            print("Download failed. Try another option.")
                            continue
                        os.environ['PCL_INPAINTING_MODEL'] = str(downloaded)
                        logger.info(f"Downloaded and selected: {downloaded.name}")
                        break
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a number.")
        except Exception as e:
            logger.warning(f"Inpainting interactive setup failed: {e}")

        # Step 2b: OCR Engine interactive selection
        try:
            import pcleaner.model_downloader as md
            import pcleaner.ocr.ocr_advanced as ocr_adv
            from pcleaner.ocr.ocr_mangaocr import MangaOcr
            from pcleaner.ocr.ocr_tesseract import TesseractOcr
            
            print("\n" + "-" * 60)
            print("🔤 OCR Engine Selection")
            print("-" * 60)
            print("  1. MangaOCR (best for Japanese/vertical text)")
            print("  2. Tesseract (if installed, best for multiple languages)")
            print("  3. EasyOCR (download if needed, multi-language support)")
            print("  4. PaddleOCR v4 (download if needed, advanced layout detection)")
            print("  0. Skip (use auto-selection)")
            print("-" * 60)
            
            choice = input("Select OCR engine (0-4, default 0): ").strip() or "0"
            
            if choice == "0":
                logger.info("✅ OCR Engine: Using auto-selection")
                self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
            elif choice == "1":
                logger.info("✅ OCR Engine: MangaOCR selected")
                self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.MANGAOCR
            elif choice == "2":
                # Check if Tesseract is available
                if not TesseractOcr.langs():
                    print("⚠️ Tesseract not found. Please install Tesseract language packs.")
                    print("Falling back to auto-selection.")
                    self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
                else:
                    logger.info("✅ OCR Engine: Tesseract selected")
                    self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.TESSERACT
            elif choice == "3":
                print("Checking EasyOCR...")
                try:
                    import easyocr
                    print("✅ EasyOCR is already installed")
                except ImportError:
                    print("EasyOCR not found. Attempting to install...")
                    try:
                        import subprocess
                        import sys
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "easyocr"],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode != 0:
                            print(f"Installation failed: {result.stderr}")
                            print("Falling back to auto-selection.")
                            self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
                        else:
                            logger.info("✅ OCR Engine: EasyOCR selected and installed")
                            self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.EASYOCR
                    except Exception as e:
                        logger.error(f"Failed to install EasyOCR: {e}")
                        self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
                else:
                    logger.info("✅ OCR Engine: EasyOCR selected")
                    self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.EASYOCR
            elif choice == "4":
                print("Checking PaddleOCR...")
                try:
                    from paddleocr import PaddleOCR
                    print("✅ PaddleOCR is already installed")
                except ImportError:
                    print("PaddleOCR not found. Attempting to install...")
                    try:
                        import subprocess
                        import sys
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "paddleocr"],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode != 0:
                            print(f"Installation failed: {result.stderr}")
                            print("Falling back to auto-selection.")
                            self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
                        else:
                            logger.info("✅ OCR Engine: PaddleOCR selected and installed")
                            self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.PADDLEOCR
                    except Exception as e:
                        logger.error(f"Failed to install PaddleOCR: {e}")
                        self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
                else:
                    logger.info("✅ OCR Engine: PaddleOCR selected")
                    self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.PADDLEOCR
            else:
                print("❌ Invalid selection. Using auto-selection.")
                self.selected_profile.preprocessor.ocr_engine = cfg.OCREngine.AUTO
                
        except Exception as e:
            logger.warning(f"OCR interactive setup failed: {e}")

        # Step 3: Parameter mutation (other runtime settings)
        try:
            self.display_parameter_menu(self.selected_profile)
        except KeyboardInterrupt as e:
            logger.warning(f"Operation cancelled: {e}")
            return None

        return self.selected_profile


def launch_interactive_runtime() -> Optional[cfg.Profile]:
    """
    Launch the interactive CLI runtime controller.
    
    :return: The configured Profile object, or None if cancelled
    """
    config = cfg.load_config()
    controller = InteractiveCLIController(config)
    return controller.run_interactive_session()
