from llama_index.core.schema import Document, TextNode
from typing import List, Generator


class LengthSplitter:

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: List[Document]) -> Generator[TextNode, None, None]:
        for doc in docs:
            yield from self.split_chunk(doc)

    def split_sentence(self, doc: Document) -> List[str]:
        sentence_delimiters = ".。？！?！"
        output = []
        start = 0
        content = doc.text
        for idx, char in enumerate(content):
            if char in sentence_delimiters or idx - start > self.chunk_size * 8:
                end = idx
                tmp = content[start : end + 1]
                if len(tmp) > 0:
                    output.append(tmp)
                start = idx + 1
        res = content[start:]
        if len(res) > 0:
            output.append(res)
        return output

    def split_chunk(self, doc: Document) -> List[TextNode]:
        content = self.split_sentence(doc)
        splitted = []
        cur = []
        cur_len = 0
        for sentence in content:
            if cur_len + len(sentence) > self.chunk_size:
                if cur:
                    splitted.append(cur)
                tmp = []
                cur_len = 0
                for item in cur[::-1]:
                    if cur_len >= self.chunk_overlap:
                        break
                    tmp.append(item)
                    cur_len += len(item)
                cur = tmp[::-1]

            cur.append(sentence)
            cur_len += len(sentence)
        if len(cur) > 0:
            splitted.append(cur)

        output = []
        start_char_idx = 0
        for sentences in splitted:
            chunk = TextNode(
                text="".join(sentences),
                start_char_idx=start_char_idx,
                metadata=doc.metadata,
            )  # type: ignore[call-arg]
            start_char_idx += len("".join(sentences))
            from . import chunk_hash

            chunk.node_id = chunk_hash(chunk)
            output.append(chunk)
        return output
