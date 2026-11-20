from abc import ABC, abstractmethod
from typing import Any, List, Generator
from rag.model.atlas_model import CompletionResponse, AtlasLLM
from tenacity import retry, stop_after_attempt

class BasePrompt(ABC):
    template: str = """
    {{
        "instruction": "你是一个专业的智能助手，能够根据已有的知识通过深入的思考，解决用户的问题，如果无法解决用户的问题，直接回复无法回答即可。请以JSON字符串格式回应。你可以参照examples的示例。",
        "examples": [
            {{
                "input": "你是谁？",
                "output": "我是AtlasRag智能助手，我是由Atlas团队开发，你可以使用我提供的知识库来回答你的问题。",
            }}
        ],
        "input": "{input}",
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    @abstractmethod
    def build_prompt(self, **kwargs: Any) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} need to implement `build_prompt` method."
        )

    @abstractmethod
    def parse_response(self, response: str | CompletionResponse) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} need to implement `parse_response` method."
        )

    @retry(stop=stop_after_attempt(3))
    def get_result(self, llm: AtlasLLM, **kwargs: Any) -> Any:
        prompt = self.build_prompt(**kwargs)
        return self.parse_response(llm.complete(prompt))

    def get_stream_result(
        self, llm: AtlasLLM, **kwargs: Any
    ) -> Generator[str, Any, None]:
        prompt = self.build_prompt(**kwargs)
        for answer in llm.stream_complete(prompt):
            if answer.delta:
                yield answer.delta
        return llm.stream_complete(answer)
