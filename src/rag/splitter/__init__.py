from .length_splitter import LengthSplitter
from .pattern_splitter import PatternSplitter
import tiktoken
from typing import List
from llama_index.core.schema import TextNode
from hashlib import sha256

__all__ = ["LengthSplitter", "PatternSplitter"]


def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_tokens(text: str, model: str = "gpt-4") -> List[str]:
    encoding = tiktoken.encoding_for_model(model)
    return [
        encoding.decode_single_token_bytes(token).decode("utf-8", errors="replace")
        for token in encoding.encode(text)
    ]

def chunk_hash(chunk:TextNode)->str:
    doc_identity = str(chunk.text) + str(chunk.metadata["file_name"])
    return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())
