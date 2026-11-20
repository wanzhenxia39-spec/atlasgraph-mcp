from typing import Optional, List, Any, Sequence, Dict
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    ChatResponseGen,
    LLMMetadata,
    MessageRole,
    ChatMessage,
    ChatResponse,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
import requests
import json
from rag.utils.log import rag_logger as log
from rag.utils.config import rag_config as config
import base64
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

def to_message_dicts(messages: Sequence[ChatMessage]) -> List:
    return [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in messages
        if all([value is not None for value in message])
    ]


def get_additional_kwargs(response: Dict[str, Any]) -> Dict:
    return {
        "token_counts": response["usage"]["total_tokens"],
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
    }


class AtlasLLM(CustomLLM):
    num_output: int = DEFAULT_NUM_OUTPUTS
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    model: str = Field(
        default="qwen2.5-72b-instruct",
        description="The ChatGlM model to use. glm-4 or glm-3-turbo",
    )
    api_url: str = Field(
        default="http://127.0.0.1:8000/v1/chat/completions",
        description="The Atlas LLM API URL.",
    )

    api_key: Optional[str] = Field(
        default=None, description="The Atlas LLM API KEY"
    )

    def __init__(
        self,
        model: str = "Qwen2.5-72B-Instruct-AWQ",
        api_base: Optional[str] = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super(AtlasLLM, self).__init__(model=model, **kwargs)  # type: ignore[call-arg]
        if api_url:
            self.api_url = api_url
        else:
            self.api_url = f"{api_base}/chat/completions"
        self.api_key = api_key

    @classmethod
    def class_name(cls) -> str:
        return "atlas_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

    def _chat(self, messages: List) -> Any:
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        data.update(config.llm_config)
        response = requests.post(
            self.api_url,
            json=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            return response.json()
        except:
            log.error(
                f"llm response result cannot be parsed as json and response: {response}"
            )
            return {}

    def _stream_chat(self, messages: List) -> Any:
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        data.update(config.llm_config)
        stream_response = requests.post(
            self.api_url,
            json=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            stream=True,
        )
        try:
            chunk = ""
            for chunk in stream_response.iter_lines():
                if chunk:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")
                    chunk_json = json.loads(chunk[len("data: ") :])
                    yield chunk_json
                    if chunk_json["choices"][0]["finish_reason"] == "stop":
                        break
        except Exception as e:
            log.error(
                f"stream llm error: chunk: {chunk} \n detail: {str(e)}"
            )
            raise e

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts: List = to_message_dicts(messages)
        response = self._chat(message_dicts)
        rsp = ChatResponse(
            message=ChatMessage(
                content=response["choices"][0]["message"]["content"],
                role=MessageRole(response["choices"][0]["message"]["role"]),
                additional_kwargs={},
            ),
            raw=response,
            additional_kwargs=get_additional_kwargs(response),
        )
        return rsp

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        response_txt = ""
        message_dicts: List = to_message_dicts(messages)
        response: Dict = self._stream_chat(message_dicts)
        for chunk in response:
            token = chunk["choices"][0]["message"]["content"]
            response_txt += token
            yield ChatResponse(
                message=ChatMessage(
                    content=response_txt,
                    role=MessageRole(response.get("role")),
                    additional_kwargs={},
                ),
                delta=token,
                raw=chunk,
            )

    @llm_completion_callback()
    @retry(wait=wait_random_exponential(min=6, max=60), stop=stop_after_attempt(3))
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._chat(messages)
            rsp = CompletionResponse(
                text=str(response["choices"][0]["message"]["content"]),
                raw=response,
                additional_kwargs=get_additional_kwargs(response),
            )
        except:
            log.error(f"llm response result: {response}")
            raise RuntimeError(f"llm response result: {response}")
        return rsp

    @llm_completion_callback()
    def complete_with_image(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        image_url: Optional[str] = None,
    ) -> CompletionResponse:
        if image_path:
            with image_path.open("rb") as file:
                image_base64 = f'data:image/{image_path.suffix[1:]};base64,{base64.b64encode(file.read()).decode("utf-8")}'
        else:
            image_base64 = None
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url or image_base64},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        try:
            response = self._chat(messages)
            rsp = CompletionResponse(
                text=str(response["choices"][0]["message"]["content"]),
                raw=response,
                additional_kwargs=get_additional_kwargs(response),
            )
        except:
            if response:
                log.error(f"llm response result: {response}")
            return CompletionResponse(text="")
        return rsp

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response_txt = ""
        messages = [{"role": "user", "content": prompt}]
        response = self._stream_chat(messages)
        for chunk in response:
            token = chunk["choices"][0]["delta"]["content"]
            if token:
                response_txt += token
                yield CompletionResponse(text=response_txt, delta=token)


from llama_index.core.embeddings import BaseEmbedding


class AtlasEmbeddings(BaseEmbedding):
    model: str = Field(default="bge-large-zh", description="The Atlas Model to use. ")
    api_url: str = Field(
        default="http://127.0.0.1:8000/v1/embeddings",
        description="The Atlas Model API URL.",
    )
    api_key: Optional[str] = Field(
        default=None, description="The Atlas Embeddings API KEY"
    )

    def __init__(
        self,
        model: str = "bge-large-zh",
        api_base: Optional[str] = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super(AtlasEmbeddings, self).__init__(
            model=model,
            **kwargs,
        )  # type: ignore[call-arg]
        self.model = model
        if api_url:
            self.api_url = api_url
        else:
            self.api_url = f"{api_base}/embeddings"
        self.api_key = api_key

    @classmethod
    def class_name(cls) -> str:
        return "AtlasEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self.get_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self.get_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)

    def get_general_text_embedding(self, prompt: str) -> List[float]:
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "input": prompt,
                "user": "test",
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            response_json = response.json()
        except:
            log.error(
                f"embedding response result cannot be parsed as json and response: {response}"
            )
            raise RuntimeError(
                f"embedding response result cannot be parsed as json and response: {response}"
            )
        if "data" not in response_json or response_json["data"] is None:
            log.error(f"response json data: {response_json}")
            raise RuntimeError("embedding model connection error!")
        return response_json["data"][0]["embedding"]
