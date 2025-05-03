import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import cv2
import psutil
from PIL import Image
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .helper import pil_to_bytes
from .model.utils import torch_gc
from .model_manager import ModelManager
from .schema import InpaintRequest


def glob_images(path: Path) -> Dict[str, Path]:
    if path.is_file():
        return {path.stem: path}
    elif path.is_dir():
        return {
            it.stem: it
            for it in path.glob("*.*")
            if it.suffix.lower() in [".png", ".jpg", ".jpeg"]
        }
    return {}


def batch_inpaint(
    model: str,
    device,
    image: Path,
    mask: Path,
    output: Path,
    config: Optional[Path] = None,
    concat: bool = False,
):
    if image.is_dir() and output.is_file():
        logger.error("When image is a directory, output must be a directory")
        exit(-1)
    output.mkdir(parents=True, exist_ok=True)

    image_paths = glob_images(image)
    mask_paths = glob_images(mask)
    if not image_paths:
        logger.error("No valid images found in image path")
        exit(-1)
    if not mask_paths:
        logger.error("No valid masks found in mask path")
        exit(-1)

    inpaint_request = InpaintRequest(**(json.loads(config.read_text()) if config else {}))
    model_manager = ModelManager(name=model, device=device)
    first_mask = next(iter(mask_paths.values()))  # 获取第一个掩膜作为备用

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        
        task = progress.add_task("Batch processing...", total=len(image_paths))
        
        for stem, image_p in image_paths.items():
            # 处理掩膜匹配
            mask_p = mask_paths.get(stem)
            if not mask_p:
                if mask.is_dir():
                    progress.log(f"Mask for {image_p.name} not found, skipped")
                    progress.update(task, advance=1)
                    continue
                mask_p = first_mask  # 使用第一个掩膜作为默认

            try:
                # 使用 PIL 读取原图并转换
                with Image.open(image_p) as pil_img:
                    img = pil_img.convert("RGB")
                    img_array = np.array(img)
                    infos = pil_img.info  # 保留元数据

                # 处理掩膜
                with Image.open(mask_p) as mask_img:
                    mask_array = np.array(mask_img.convert("L"))  # 转换为灰度

                # 尺寸校验和调整
                if mask_array.shape != img_array.shape[:2]:
                    progress.log(f"Resizing mask {mask_p.name} to match {image_p.name}")
                    mask_array = cv2.resize(
                        mask_array,
                        (img_array.shape[1], img_array.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # 二值化处理
                mask_array = np.where(mask_array >= 127, 255, 0).astype(np.uint8)

                # 执行修复
                inpaint_result = model_manager(img_array, mask_array, inpaint_request)
                
                # 转换颜色空间 (假设模型返回BGR)
                inpaint_rgb = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)

                # 拼接结果（如果需要）
                if concat:
                    orig_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # 保持显示一致性
                    mask_vis = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB)
                    inpaint_rgb = cv2.hconcat([orig_rgb, mask_vis, inpaint_rgb])

                # 保存结果
                result_img = Image.fromarray(inpaint_rgb)
                save_path = output / f"{stem}.png"
                save_path.write_bytes(pil_to_bytes(result_img, "png", 100, infos))

            except Exception as e:
                logger.error(f"Error processing {image_p.name}: {str(e)}")
            finally:
                progress.update(task, advance=1)
                torch_gc()  # 清理显存

        # 内存监控（可选）
        if logger.level("DEBUG"):
            mem = psutil.Process().memory_info().rss / 1024 ** 2
            logger.debug(f"Memory usage: {mem:.2f} MB")
