from typing import Dict, List, Tuple
from tqdm import tqdm
from rag.bench.until import get_em_f1, remove_think_tags
from pathlib import Path
import json
import websockets
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from rag.utils.log import rag_logger as log
import os
import requests
import asyncio
from functools import partial


class QAEvaluator:
    def __init__(
        self,
        qa_json_path: Path,
        addr: str,
        base_rag_config: Dict,
        max_workers: int = 1,
        metrics: List[str] = ["em", "f1"],
    ):
        self.qa_json_path = qa_json_path
        self.addr = addr
        self.max_workers = max_workers
        self.complete_url = f"ws://{self.addr}/ws/complete"
        self.create_session = f"http://{self.addr}/api/create_session"
        self.base_rag_config = base_rag_config
        response = requests.post(
            self.create_session,
            params={"username": "root", "question": "QA Evaluator"},
        )
        self.base_rag_config["session_id"] = response.json()["session_id"]
        self.metrics = metrics

    @property
    def instruct(self) -> str:
        # return """
        # Answer the question based on the given reference.
        # \nIf you can answer the question, Only give me the answer and do not output any other words
        # \nUse answers that appear in known texts whenever possible
        # \nIf you can't answer, reply directly to can not answer
        # """
        return """
        基于给定的上下文回答问题尽可能的回答问题
        如果你能回答该问题，请回答的尽可能的回答得精炼，尽可能从提供的相关文本中找答案并且生成的答案最多10个词
        你不需要给出其他的解释描述
        如果你不能回答该问题，直接回复无法回答,格式还是要保持instruct中要求JSON的格式
        """

    def build_cot_prompt(self, question: str) -> str:
        cot_prompt = """
        As an advanced reading comprehension assistant, your task is to answer complex multi-hop questions based on the context I provide. The context I offer includes two parts: a set of documents that are helpful for answering the question, and a step-by-step breakdown of the question along with an analytical thought process. Please combine these two parts of the context to answer the question. Your response start after "Thought: ", where you will methodically break down the reasoning process step by step, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.\n
        \nNOTE:
        \n1. I hope your answer matches the answer exactly, so ENSURE that the answer following "Answer:" is concise, such as 14 May, 1832  or yes. THE SHORTER, THE BETTER!!
        \n2. If the answer is a date, please provide the full date as much as possible, such as 18 May, 1932.3. Pay attention to the differences in part of speech, such as "Japan" and "Japanese," and provide the accurate format according to the question.
        \n3. If you believe the provided documents cannot answer the question, response with Answer: UNKNOWN.
        \nFor example:
        \nQuestion: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
        \nAnswer: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.
        \n\nQuestion:
        """
        return f"{cot_prompt} {question}\nAnswer:"

    async def complete(
        self, question: str, instruct: str, true_answer: str
    ) -> Tuple[str, str]:
        answer = ""
        try:
            async with websockets.connect(
                self.complete_url,
                open_timeout=None,
                close_timeout=None,
                ping_timeout=None,
            ) as websocket:
                base_rag_config = self.base_rag_config.copy()
                base_rag_config["question"] = question
                base_rag_config["instruct"] = instruct
                await websocket.send(json.dumps(base_rag_config, ensure_ascii=False))
                while True:
                    try:
                        response = json.loads(await websocket.recv())
                        if response["code"] == 1:
                            answer += response["answer"]
                    except websockets.ConnectionClosed:
                        return true_answer, remove_think_tags(answer)
        except:
            pass
        return true_answer, remove_think_tags(answer)

    def evaluate(self) -> Dict[str, float]:
        all_true = []
        all_pred = []
        with self.qa_json_path.open("r") as f:
            items = json.load(f)
        tasks: List[Future] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for item in items:
                if self.base_rag_config["index_mode"] in [
                    "vector",
                    "hybrid",
                    "tog",
                    "kg",
                    "agent"
                ]:
                    tasks.append(
                        executor.submit(
                            partial(
                                asyncio.run,
                                self.complete(
                                    item["question"],
                                    self.instruct,
                                    item["answer"],
                                ),
                            )
                        )
                    )
            with log.terminator_manager(), tqdm(
                total=len(tasks),
                file=open(os.devnull, "w"),
                ncols=160,
            ) as pbar:
                pbar.set_description(f"Evaluating")
                total_em = 0.0
                total_f1 = 0.0
                for index, task in enumerate(as_completed(tasks)):
                    true_answer, pred_answer = task.result()
                    log.info(
                        f"\n True Answer: {true_answer}\n Pred Answer: {pred_answer}"
                    )
                    pbar.update()
                    em_f1 = self.getEmAndF1(
                        [true_answer],
                        [pred_answer],
                    )
                    total_em += em_f1["em"]
                    total_f1 += em_f1["f1"]
                    pbar.set_postfix(
                        {
                            "em": round(total_em / (index + 1), 2),
                            "f1": round(total_f1 / (index + 1), 2),
                        }
                    )
                    if index != len(tasks) - 1:
                        log.set_terminator(end="\r")
                    else:
                        log.set_terminator(end="\n")
                    log.info(str(pbar))
                    all_true.append(true_answer)
                    all_pred.append(pred_answer)
        return self.getEmAndF1(all_true, all_pred)

    def getEmAndF1(
        self, predictionlist: List[str], goldlist: List[str]
    ) -> Dict[str, float]:
        """
        Calculates and returns evaluation metrics between predictions and ground truths.

        This function evaluates the match between predictions and ground truths by calculating
        the exact match (EM) and F1 score, as well as answer similarity.

        Parameters:
        predictionlist (List[str]): List of predicted values from the model.
        goldlist (List[str]): List of actual ground truth values.

        Returns:
        dict: Dictionary containing EM, F1 score, and answer similarity.
        """
        # Initialize total metrics
        total_metrics = {"em": 0.0, "f1": 0.0, "answer_similarity": 0.0}

        # Iterate over prediction and gold lists to calculate EM and F1 scores
        for prediction, gold in zip(predictionlist, goldlist):
            em, f1 = get_em_f1(
                prediction, gold
            )  # Call external function to calculate EM and F1
            total_metrics["em"] += em  # Accumulate EM score
            total_metrics["f1"] += f1  # Accumulate F1 score

        # Calculate average EM and F1 scores
        total_metrics["em"] /= len(predictionlist)
        total_metrics["f1"] /= len(predictionlist)
        return total_metrics


if __name__ == "__main__":
    import logging
    from driver3.atlas_logger.AtlasLogger import logger

    logger.setLevel(logging.CRITICAL)
    from rag.utils.config import rag_config as config

    config.set_rag_server_path(Path("../service/config.yaml"))
    evaluator = QAEvaluator(
        qa_json_path=Path("/tmp/rag-data/rag_eval_data/musique/qa/qa.json"),
        addr="0.0.0.0:21033",
        base_rag_config={
            "database": "musique",
            "username": "root",
            "index_mode": "agent",
            "vector_config": {
                "similarity_top_k": 10,
            },
            "tog_config": {
                "temperature_exploration": 0.4,
                "temperature_reasoning": 0.0,
                "depth": 3,
                "max_width": 10,
                "min_score": 0.35,
                "vector_prune": True,
            },
            "agent_config": {"max_depth": 5, "similarity_top_k": 10},
            "hybrid_config": {
                "similarity_top_k": 10,
                "retriever_mode": "coarse-grained",
                "max_depth": 4,
            },
            "kg_config": {"similarity_top_k": 10, "max_entity_num": 10, "depth": 3},
        },
        max_workers=10,
    )
    print(evaluator.evaluate())
