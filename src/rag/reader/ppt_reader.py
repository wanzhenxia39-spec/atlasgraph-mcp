from llama_index.core.readers.base import BaseReader

from fsspec import AbstractFileSystem
from pathlib import Path
from typing import Optional, Dict, List
from llama_index.core.schema import Document
import subprocess
from rag.utils.log import rag_logger as log
from rag.utils.config import rag_config as config
from pptx import Presentation
import tempfile
import os
from pathlib import Path
from rag.model.atlas_model import AtlasLLM


class PptReader(BaseReader):
    def __init__(self) -> None:
        if config.mllm_model is None:
            self.model = None
        else:
            self.model = AtlasLLM(
                model=config.mllm_model,
                api_url=config.mllm_url,
                api_base=config.mllm_base,
                api_key=config.mllm_key,
            )

    def caption_image(self, tmp_image_file: str) -> str:
        """Generate text caption of image."""
        if self.model is None:
            return "无法识别的图片"
        return str(
            self.model.complete_with_image("请描述一下图片内容", Path(tmp_image_file))
        )

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        output_file: Optional[Path] = None
        if file.suffix == ".ppt":
            output_file = file.with_suffix(".pptx")
            try:
                m_result = subprocess.run(
                    [
                        "libreoffice",
                        "--headless",
                        "--convert-to",
                        "pptx",
                        "--outdir",
                        str(file.parent),
                        str(file),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                log.info(m_result.stdout)
                log.info(f"Converted {file} to {output_file}")
            except subprocess.CalledProcessError as e:
                log.info(m_result.stderr)
                log.error(f"Failed to convert {file} to {output_file}")
                raise e
        if fs:
            with fs.open(output_file or file) as f:
                presentation = Presentation(f)
        else:
            presentation = Presentation(output_file or file)
        result = ""
        for i, slide in enumerate(presentation.slides):
            result += f"\n\nSlide #{i}: \n"
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    image = shape.image
                    # get image "file" contents
                    image_bytes = image.blob
                    # temporarily save the image to feed into model
                    f = tempfile.NamedTemporaryFile("wb", delete=False)
                    try:
                        f.write(image_bytes)
                        f.close()
                        result += f"\n Image: {self.caption_image(f.name)}\n\n"
                    finally:
                        os.unlink(f.name)

                if hasattr(shape, "text"):
                    result += f"{shape.text}\n"
        if output_file:
            output_file.unlink()
        return [Document(text=result, metadata=extra_info or {})]  # type: ignore[call-arg]
