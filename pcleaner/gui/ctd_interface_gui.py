import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Sequence

import pcleaner.config as cfg
import pcleaner.gui.image_file as imf
import pcleaner.output_structures as ost
import pcleaner.gui.worker_thread as wt
from pcleaner.comic_text_detector.inference import TextDetector
from pcleaner.ctd_interface import process_image
from pcleaner.runtime.device import cuda_available


# It's simply slower for anything less than this many images.
MIN_IMAGES_FOR_MULTIPROCESSING = 20


def model2annotations_gui(
    config_general: cfg.GeneralConfig,
    config_detector: cfg.TextDetectorConfig,
    text_detector_model_path: Path,
    img_list: Sequence[imf.ImageFile],
    save_dir: Path,
    no_text_detection: bool,
    visualize_raw_boxes: bool,
    partial_progress_data: partial,
    progress_callback: ost.ProgressSignal | None,
    abort_flag: wt.SharableFlag,
):
    """
    Run the model on a directory of images and produce the following
    for each image inside the save_dir directory:
    - A copy of the original image as a .png file.
    - A .png file containing the text mask, filename: <image_name>_mask.png.
    - A .json file containing each box of text, as well as other metadata,
      filename: <image_name>.json.

    For this modified version, include the image name and mask name in the
    json file.

    :param config_general: General configuration, part of the profile.
    :param config_detector: Text detector configuration, part of the profile.
    :param text_detector_model_path: Path to the model file.
    :param img_list: A list of ImageFile objects.
    :param save_dir: Path to the directory where the results will be saved.
    :param no_text_detection: If True, skip the text detection step.
    :param visualize_raw_boxes: If True, visualize the raw box data.
    :param partial_progress_data: Data to pass to the progress callback.
    :param progress_callback: A callback to report progress.
    :param abort_flag: A flag to signal abort.
    """

    num_processes = min(config_detector.concurrent_models, len(img_list))

    if num_processes > 1 and not no_text_detection:
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)
        # Distribute the images evenly among the processes.
        batches = [list() for _ in range(num_processes)]
        for i, img_obj in enumerate(img_list):
            batches[i % num_processes].append(img_obj)

        device = "cuda" if cuda_available() else "cpu"

        args = [
            (
                [img_obj.path for img_obj in batch],
                text_detector_model_path,
                device,
                save_dir,
                config_general.input_height_lower_target,
                config_general.input_height_upper_target,
                partial_progress_data,
                visualize_raw_boxes,
            )
            for batch in batches
        ]

        with mp.Pool(num_processes) as pool:
            for _ in pool.imap_unordered(process_image_batch, args):
                if abort_flag.get():
                    raise wt.Abort()

    else:
        # Load the model.
        cuda = cuda_available()
        device = "cuda" if cuda else "cpu"
        model = TextDetector(
            model_path=str(text_detector_model_path), input_size=1024, device=device
        )

        inc_progress = partial_progress_data

        for img_obj in img_list:
            if abort_flag.get():
                raise wt.Abort()

            process_image(
                img_obj.path,
                model,
                save_dir,
                config_general.input_height_lower_target,
                config_general.input_height_upper_target,
                no_text_detection,
                img_obj.uuid,
                visualize_raw_boxes,
            )
            inc_progress()


def process_image_batch(args) -> None:
    (
        img_batch,
        model_path,
        device,
        save_dir,
        lower_target,
        upper_target,
        inc_progress,
        visualize_raw_boxes,
    ) = args
    model = TextDetector(model_path=str(model_path), input_size=1024, device=device)
    for img_path, img_uuid in img_batch:
        process_image(
            img_path,
            model,
            save_dir,
            lower_target,
            upper_target,
            uuid=img_uuid,
            visualize_raw_data=visualize_raw_boxes,
        )
        inc_progress()

    del model

    if device == "cuda":
        import torch
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
