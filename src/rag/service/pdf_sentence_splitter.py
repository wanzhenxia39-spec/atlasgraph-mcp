"""Sentence splitter."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import DEFAULT_CHUNK_SIZE
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.node_parser.text.utils import (
    split_by_char,
    split_by_regex,
    split_by_sentence_tokenizer,
    split_by_sep,
)
from llama_index.core.schema import Document
from llama_index.core.utils import get_tokenizer
from rag.utils.log import rag_logger as log
SENTENCE_CHUNK_OVERLAP = 200
CHUNKING_REGEX = "[^,.;。？！]+[,.;。？！]?"
DEFAULT_PARAGRAPH_SEP = "\n\n\n"


@dataclass
class _Split:
    text: str  # the split text
    is_sentence: bool  # save whether this is a full sentence
    token_size: int  # token length of split text


class Paragraph:
    def __init__(self, type: str, text: str, caption: str = "") -> None:
        self.type = type
        self.text = text
        self.caption = caption


class PdfSentenceSplitter(MetadataAwareTextSplitter):
    """Parse text with a preference for complete sentences.

    In general, this class tries to keep sentences and paragraphs together. Therefore
    compared to the original TokenTextSplitter, there are less likely to be
    hanging sentences or parts of sentences at the end of the node chunk.
    """

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=SENTENCE_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
        gt=0,
    )
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    paragraph_separator: str = Field(
        default=DEFAULT_PARAGRAPH_SEP, description="Separator between paragraphs."
    )
    secondary_chunking_regex: str = Field(
        default=CHUNKING_REGEX, description="Backup regex for splitting into sentences."
    )

    _chunking_tokenizer_fn: Callable[[str], List[str]] = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()
    _split_fns: List[Callable] = PrivateAttr()
    _sub_sentence_split_fns: List[Callable] = PrivateAttr()

    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
        chunking_tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        secondary_chunking_regex: str = CHUNKING_REGEX,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        id_func = id_func or default_id_func

        callback_manager = callback_manager or CallbackManager([])
        self._chunking_tokenizer_fn = (
            chunking_tokenizer_fn or split_by_sentence_tokenizer()
        )
        self._tokenizer = tokenizer or get_tokenizer()

        self._split_fns = [
            split_by_sep(paragraph_separator),
            self._chunking_tokenizer_fn,
        ]

        self._sub_sentence_split_fns = [
            split_by_regex(secondary_chunking_regex),
            split_by_sep(separator),
            split_by_char(),
        ]

        super(PdfSentenceSplitter, self).__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            secondary_chunking_regex=secondary_chunking_regex,
            separator=separator,
            paragraph_separator=paragraph_separator,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )  # type: ignore[call-arg]

    @classmethod
    def from_defaults(
        cls,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
        chunking_tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        secondary_chunking_regex: str = CHUNKING_REGEX,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ) -> "PdfSentenceSplitter":
        """Initialize with parameters."""
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            paragraph_separator=paragraph_separator,
            chunking_tokenizer_fn=chunking_tokenizer_fn,
            secondary_chunking_regex=secondary_chunking_regex,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceSplitter"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        # log.info(text)
        # log.info(metadata_str)
        # log.info(self.chunk_size)
        metadata_len = len(self._tokenizer(metadata_str))
        effective_chunk_size = self.chunk_size - metadata_len
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_len}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        elif effective_chunk_size < 50:
            log.info(
                f"Metadata length ({metadata_len}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
            )

        return self._split_text(text, chunk_size=effective_chunk_size)

    def split_text(self, text: str) -> List[str]:
        # log.info(text)
        return self._split_text(text, chunk_size=self.chunk_size)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        _Split incoming text and return chunks with overlap size.

        Has a preference for complete sentences, phrases, and minimal overlap.
        """
        if text == "":
            return [text]

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            paragraphs = text.split("<paddle_")
            new_paragraphs: List[Paragraph] = []
            all_chunks = []
            cur_title = ""
            for paragraph in paragraphs:
                if paragraph == "\n" or paragraph == "":
                    continue
                if paragraph[:5] == "title":
                    txt = (
                        paragraph[7:].replace("\n<end_paddle_title>\n", "").strip("\n")
                    )
                    new_paragraphs.append(Paragraph("title", txt))
                    cur_title = txt
                elif paragraph[:4] == "text":
                    txt = (
                        paragraph[12:]
                        .replace("\n<end_paddle_text_block>\n", "")
                        .strip("\n")
                    )
                    if len(new_paragraphs) > 0 and new_paragraphs[-1].type == "text":
                        new_paragraphs[-1].text += txt
                    else:
                        new_paragraphs.append(Paragraph("text", txt, caption=cur_title))
                elif paragraph[:13] == "table_caption":
                    txt = (
                        paragraph[15:]
                        .replace("\n<end_paddle_table_caption>\n", "")
                        .strip("\n")
                    )
                    if (
                        len(new_paragraphs) > 0
                        and new_paragraphs[-1].type == "table"
                        and new_paragraphs[-1].caption == cur_title
                    ):
                        new_paragraphs[-1].caption += "\n" + txt
                    else:
                        new_paragraphs.append(
                            Paragraph("table", "", caption=cur_title + "\n" + txt)
                        )
                elif paragraph[:11] == "table_plain":
                    txt = paragraph[13:].strip("\n")
                    if (
                        len(new_paragraphs) > 0
                        and new_paragraphs[-1].type == "table"
                        and new_paragraphs[-1].text == ""
                    ):
                        new_paragraphs[-1].text = txt
                    else:
                        new_paragraphs.append(Paragraph("table", txt))
                elif paragraph[:17] == "table_description":
                    txt = (
                        paragraph[19:].replace("\n<end_paddle_table>\n", "").strip("\n")
                    )
                    new_paragraphs[-1].text += "\n\n\n" + txt

            acc_titles = ""
            for m_paragraph in new_paragraphs:
                if m_paragraph.type == "title":
                    acc_titles += m_paragraph.text + "\n"
                elif m_paragraph.type == "text":
                    full_text = acc_titles + m_paragraph.text
                    splits = self._split(full_text, chunk_size)
                    chunks = self._merge(splits, chunk_size, m_paragraph.caption + "\n")
                    acc_titles = ""
                    all_chunks += chunks
                elif m_paragraph.type == "table":
                    splits = self._split(m_paragraph.text, chunk_size)
                    # lines = paragraph.text.strip('\n').split('\n')
                    # headers = '\n'.join(lines[: min(len(lines), 2)])
                    chunks = self._merge(splits, chunk_size, m_paragraph.caption + "\n")
                    all_chunks += chunks

            event.on_end(payload={EventPayload.CHUNKS: chunks})

        return chunks

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        r"""Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by paragraph separator
        2. split by chunking tokenizer (default is nltk sentence tokenizer)
        3. split by second chunking regex (default is "[^,\.;]+[,\.;]?")
        4. split by default separator (" ")

        """
        # log.info(repr(text))
        # log.info(chunk_size)
        token_size = self._token_size(text)
        if self._token_size(text) <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits_by_fns, is_sentence = self._get_splits_by_fns(text)

        text_splits = []
        for text_split_by_fns in text_splits_by_fns:
            token_size = self._token_size(text_split_by_fns)
            if token_size <= chunk_size:
                text_splits.append(
                    _Split(
                        text_split_by_fns,
                        is_sentence=is_sentence,
                        token_size=token_size,
                    )
                )
            else:
                recursive_text_splits = self._split(
                    text_split_by_fns, chunk_size=chunk_size
                )
                text_splits.extend(recursive_text_splits)
        # log.info(text_splits)
        return text_splits

    def _merge(self, splits: List[_Split], chunk_size: int, caption: str) -> List[str]:
        """Merge splits into chunks."""
        chunks: List[str] = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        last_chunk: List[Tuple[str, int]] = []
        cur_chunk_len = 0
        new_chunk = True

        caption_len = len(self._tokenizer(caption))
        if caption_len > chunk_size:
            caption = ""
        else:
            chunk_size -= caption_len

        def close_chunk() -> None:
            nonlocal chunks, cur_chunk, last_chunk, cur_chunk_len, new_chunk

            chunks.append(caption + "".join([text for text, length in cur_chunk]))
            last_chunk = cur_chunk
            cur_chunk = []
            cur_chunk_len = 0
            new_chunk = True

            # add overlap to the next chunk using the last one first
            # there is a small issue with this logic. If the chunk directly after
            # the overlap is really big, then we could go over the chunk_size, and
            # in theory the correct thing to do would be to remove some/all of the
            # overlap. However, it would complicate the logic further without
            # much real world benefit, so it's not implemented now.
            if len(last_chunk) > 0:
                last_index = len(last_chunk) - 1
                while (
                    last_index >= 0
                    and cur_chunk_len + last_chunk[last_index][1] <= self.chunk_overlap
                ):
                    text, length = last_chunk[last_index]
                    cur_chunk_len += length
                    cur_chunk.insert(0, (text, length))
                    last_index -= 1

        while len(splits) > 0:
            cur_split = splits[0]
            if cur_split.token_size > chunk_size:
                raise ValueError("Single token exceeded chunk size")
            if cur_chunk_len + cur_split.token_size > chunk_size and not new_chunk:
                # if adding split to current chunk exceeds chunk size: close out chunk
                close_chunk()
            else:
                if (
                    cur_split.is_sentence
                    or cur_chunk_len + cur_split.token_size <= chunk_size
                    or new_chunk  # new chunk, always add at least one split
                ):
                    # add split to chunk
                    cur_chunk_len += cur_split.token_size
                    cur_chunk.append((cur_split.text, cur_split.token_size))
                    splits.pop(0)
                    new_chunk = False
                else:
                    # close out chunk
                    close_chunk()

        # handle the last chunk
        if not new_chunk:
            chunk = caption + "".join([text for text, length in cur_chunk])
            chunks.append(chunk)

        # run postprocessing to remove blank spaces
        return self._postprocess_chunks(chunks, caption)

    def _postprocess_chunks(self, chunks: List[str], caption: str) -> List[str]:
        """Post-process chunks.
        Remove whitespace only chunks and remove leading and trailing whitespace.
        """
        new_chunks = []
        for chunk in chunks:
            stripped_chunk = chunk.strip()
            if stripped_chunk == caption.strip():
                continue
            new_chunks.append(stripped_chunk)
        # log.info("-------------------chunks---------------------")
        # log.info(new_chunks)
        return new_chunks

    def _token_size(self, text: str) -> int:
        return len(self._tokenizer(text))

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        for split_fn in self._split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True
                break

        for split_fn in self._sub_sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False
