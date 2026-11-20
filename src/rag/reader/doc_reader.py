from llama_index.core.readers.base import BaseReader
from fsspec import AbstractFileSystem
from pathlib import Path
from typing import Optional, Dict, List
from llama_index.core.schema import Document
import subprocess
from rag.utils.log import rag_logger as log


class DocReader(BaseReader):

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )
        output_file: Optional[Path] = None
        if file.suffix == ".doc":
            output_file = file.with_suffix(".docx")
            try:
                result = subprocess.run(
                    [
                        "libreoffice",
                        "--headless",
                        "--convert-to",
                        "docx",
                        "--outdir",
                        str(file.parent),
                        str(file),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                log.info(result.stdout)
                log.info(f"Converted {file} to {output_file}")
            except subprocess.CalledProcessError as e:
                log.info(result.stderr)
                log.error(f"Failed to convert {file} to {output_file}")
                raise e
        if fs:
            with fs.open(str(output_file or file)) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(output_file or file)
        if output_file:
            output_file.unlink()
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)
        return [Document(text=text, metadata=metadata or {})]  # type: ignore[call-arg]
