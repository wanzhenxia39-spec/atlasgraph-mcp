import os
import cv2
import numpy as np
from paddleocr import PPStructure
from paddle.utils import try_import
from PIL import Image

"""
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
"""
from typing import List
from cv2.typing import MatLike
from rag.utils.config import rag_config as config
from rag.model.atlas_model import AtlasLLM
from pathlib import Path
import subprocess
from rag.utils.log import rag_logger as log
import shutil


class PdfParser:
    def __init__(self, file_dir: str, work_path: str):
        self.file_dir = file_dir
        self.work_path = work_path
        if config.mllm_model is None:
            self.mllm = None
        else:
            self.mllm = AtlasLLM(
                model=config.mllm_model,
                api_url=config.mllm_url,
                api_base=config.mllm_base,
                api_key=config.mllm_key,
            )
        self.ocr_engine = PPStructure(
            table=True,
            ocr=True,
            show_log=True,
            structure_version="PP-StructureV2",
            lang="ch",
        )

    def vconcat_resize(
        self, img_list: List[np.ndarray], interpolation: int = cv2.INTER_CUBIC
    ) -> MatLike:
        # take maximum widths
        v_max = max(img.shape[1] for img in img_list)

        # image resizing
        im_list_resize = [
            cv2.resize(
                img,
                (v_max, int(img.shape[0] * v_max / img.shape[1])),
                interpolation=interpolation,
            )
            for img in img_list
        ]

        # return final image
        return cv2.vconcat(im_list_resize)

    def convert_to_text(self, filename: str) -> str:
        """convert table to plain text

        Args:
            filename (str): path to the image file of the table

        Returns:
            str: plain text of the table
        """
        if self.mllm is None:
            return "无法识别的表格"
        response = self.mllm.complete_with_image(
            "请将这张表格写成可以阅读的文本格式。",
            Path(filename),
        )
        return str(response).replace("\n\n", "\n")

    def convert_to_description(self, filename: str) -> str:
        """Convert table to description

        Args:
            filename (str): path to image file of the table

        Returns:
            str: description text of the table
        """
        if self.mllm is None:
            return "无法识别的表格"
        response = self.mllm.complete_with_image(
            """请你用自然语言逐行逐列详细描述这张表格中的全部内容和细节，需要包含表格中的所有数据，注意只能用自然语言，不要出现结构化的数据，
            最后对表格描述的内容做出总结。我希望得到的文字能够完全代替这张图，包括每行每列数据的标题、含义、数值等。""",
            Path(filename),
        )
        return str(response).replace("\n\n", "\n")

    def parse_file_1(self, file_path: str, filename: str) -> None:
        fitz = try_import("fitz")
        imgs = []
        with fitz.open(file_path) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)

        table_cnt = 0
        table_imgs = []
        figure_cnt = 0
        full_text = ""
        table_flag = False
        for index, img in enumerate(imgs):
            result = self.ocr_engine(img)
            # save_structure_res(result, save_folder, os.path.basename(file_path).split('.')[0], index)
            result = sorted(result, key=lambda x: x["bbox"][1])
            for line in result:
                if table_flag and line["type"] not in ["table", "header"]:
                    table_flag = False
                    img = self.vconcat_resize(table_imgs)
                    img_path = os.path.join(self.work_path, f"table_{table_cnt}.png")
                    cv2.imwrite(img_path, img)
                    full_text += "\n<paddle_table>\n<paddle_table_plain>\n"
                    full_text += self.convert_to_text(img_path) + "\n"
                    full_text += "\n<paddle_table_description>\n"
                    full_text += self.convert_to_description(img_path) + "\n"
                    full_text += "<end_paddle_table>\n"
                    table_imgs = []
                    table_cnt += 1

                if line["type"] == "text":
                    full_text += "\n<paddle_text_block>\n"
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n<end_paddle_text_block>\n"
                elif line["type"] == "title":
                    full_text += "\n<paddle_title>\n"
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n<end_paddle_title>\n"
                elif line["type"] == "table":
                    table_imgs.append(np.array(line["img"]))
                    table_flag = True
                elif line["type"] == "figure":
                    cv2.imwrite(
                        os.path.join(self.work_path, f"figure_{figure_cnt}.png"),
                        np.array(line["img"]),
                    )
                    figure_cnt += 1
                    full_text += "\n<paddle_figure>\n"
                elif line["type"] == "table_caption":
                    full_text += "\n<paddle_table_caption>\n"
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n<end_paddle_table_caption>\n"
                elif line["type"] == "figure_caption":
                    full_text += "\n<paddle_figure_caption>\n"
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n<end_paddle_figure_caption>\n"
                elif line["type"] != "header":
                    full_text += "\n<paddle_text_block>\n"
                    for item in line["res"]:
                        full_text += item["text"] + "\n"
                    full_text += "\n<end_paddle_text_block>\n"

        text_file_path = os.path.join(self.work_path, filename[:-4] + ".txt")
        with open(text_file_path, "w") as f:
            f.write(full_text)

    def parse_file(self, file_path: str, filename: str) -> None:
        fitz = try_import("fitz")
        imgs = []
        with fitz.open(file_path) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)

        table_cnt = 0
        table_imgs = []
        figure_cnt = 0
        full_text = ""
        table_flag = False
        for index, img in enumerate(imgs):
            result = self.ocr_engine(img)
            # save_structure_res(result, save_folder, os.path.basename(file_path).split('.')[0], index)
            result = sorted(result, key=lambda x: x["bbox"][1])
            for line in result:
                if table_flag and line["type"] not in ["table", "header"]:
                    table_flag = False
                    img = self.vconcat_resize(table_imgs)
                    img_path = os.path.join(self.work_path, f"table_{table_cnt}.png")
                    cv2.imwrite(img_path, img)
                    # full_text += self.convert_to_text(img_path) + '\n'
                    full_text += self.convert_to_description(img_path) + "\n"
                    table_imgs = []
                    table_cnt += 1
                if line["type"] == "text":
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n"
                elif line["type"] == "title":
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n"
                elif line["type"] == "table":
                    img = np.array(line["img"])
                    table_imgs.append(img)
                    img_path = os.path.join(
                        self.work_path, f"table_{table_cnt}_{len(table_imgs)}.png"
                    )
                    cv2.imwrite(img_path, img)
                    full_text += self.convert_to_text(img_path) + "\n"
                    table_flag = True
                elif line["type"] == "figure":
                    cv2.imwrite(
                        os.path.join(self.work_path, f"figure_{figure_cnt}.png"),
                        np.array(line["img"]),
                    )
                    figure_cnt += 1
                elif line["type"] == "table_caption":
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n"
                elif line["type"] == "figure_caption":
                    for item in line["res"]:
                        full_text += item["text"]
                    full_text += "\n"
                elif line["type"] != "header":
                    for item in line["res"]:
                        full_text += item["text"] + "\n"
                    full_text += "\n"

        if table_flag:
            table_flag = False
            img = self.vconcat_resize(table_imgs)
            img_path = os.path.join(self.work_path, f"table_{table_cnt}.png")
            cv2.imwrite(img_path, img)
            full_text += self.convert_to_description(img_path) + "\n"
            table_imgs = []
            table_cnt += 1

        text_file_path = os.path.join(self.work_path, filename[:-4] + ".txt")
        with open(text_file_path, "w") as f:
            f.write(full_text)

    def parse(self) -> None:
        file_list = []
        for dirpath, _, filenames in os.walk(self.file_dir):
            if Path(dirpath).name == "parse_pdf":
                continue
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if file_path.suffix == ".pdf":
                    file_list.append((file_path, file_path.name))
                elif file_path.suffix in [".doc", ".docx", ".ppt", ".pptx"]:
                    output_file = file_path.with_suffix(".pdf")
                    try:
                        m_result = subprocess.run(
                            [
                                "libreoffice",
                                "--headless",
                                "--convert-to",
                                "pdf",
                                "--outdir",
                                file_path.parent,
                                str(file_path),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        log.info(m_result.stdout)
                        log.info(f"Converted {file_path} to {output_file}")
                    except subprocess.CalledProcessError as e:
                        log.info(m_result.stderr)
                        log.error(f"Failed to convert {file_path} to {output_file}")
                        raise e
                    file_list.append((output_file, file_path.name))
                else:
                    shutil.copy(file_path, self.work_path)

        for file_path, file_name in file_list:
            self.parse_file(str(file_path), file_name)
