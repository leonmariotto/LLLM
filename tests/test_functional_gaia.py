"""
Run GAIA dataset tests.
Warning: a valid HF token must be set in environemnt var HF_TOKEN.
"""

import math
from pathlib import Path

import pytest
from loguru import logger

from ..LLLM.agent import Agent
from ..LLLM.agent_context import AgentResult
from ..LLLM.agent_llm import LlmClient
from ..LLLM.eval_gaia import GaiaTask, GaiaToolName, evaluate_gaia_agent
from ..LLLM.fetch import fetch_embedding_model_ir, fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.sentence_transformer import SentenceTransformerEmbedder
from ..LLLM.tool_common import Tool
from ..LLLM.tool_compute import compute_tool
from ..LLLM.tool_wiki import wiki_tools
from ..LLLM.vector_db import DEFAULT_EMBEDDING_MODEL

pytestmark = pytest.mark.slow


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"
QWEN3_4B_REPO_ID = "Qwen/Qwen3-4B"
GAIA_COMPUTE_WIKIPEDIA_TOOLS: tuple[GaiaToolName, ...] = (
    "calculator",
    "calculator (or ability to count)",
    "wikipedia",
)
GAIA_COMPUTE_WIKIPEDIA_TRACE_PATH = Path(
    "gaia_qwen3_06b_compute_wikipedia_trace.json"
)


def _wiki_tools() -> list[Tool]:
    ir = fetch_embedding_model_ir(DEFAULT_EMBEDDING_MODEL)
    embedder = SentenceTransformerEmbedder.from_ir(ir)
    return list(wiki_tools(embedder))


@pytest.fixture(scope="module")
def qwen3_06b_gaia_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


@pytest.fixture(scope="module")
def qwen3_06b_gaia_agent_with_tools() -> Agent:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    qwen3_generator = Generator(model=model, tokenizer=tokenizer, cache_length=16384)
    return Agent(
        LlmClient(
            qwen3_generator,
            max_generated_token=4096,
            temperature=0.6, top_p=0.95, top_k=20,
        ),
        [compute_tool(), *_wiki_tools()],
        agent_mode="dummy",
    )

@pytest.fixture(scope="module")
def qwen3_4b_gaia_agent() -> Agent:
    ir = fetch_model_ir(QWEN3_4B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    qwen3_generator = Generator(model=model, tokenizer=tokenizer, cache_length=16384)
    return Agent(
        LlmClient(
            qwen3_generator,
            max_generated_token=4096,
            temperature=0.6, top_p=0.95, top_k=20,
        ),
        [compute_tool(), *_wiki_tools()],
    )


@pytest.mark.slow
def test_functional_qwen3_06b_gaia_compute_wikipedia_validation(
    qwen3_06b_gaia_agent_with_tools: Agent,
) -> None:
    def agent(task: GaiaTask) -> AgentResult:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else ""
        )
        result = qwen3_06b_gaia_agent_with_tools.run(
            "Answer this GAIA benchmark question. You are running in dummy "
            "agent mode, so return a plain final answer string. Use the "
            "compute tool for arithmetic, counting, unit conversion, or exact "
            "calculation. Use Wikipedia tools step by step for encyclopedia "
            "facts: find_wiki_page to locate a page, search_in_wiki_page to "
            "look for a specific fact inside a known page, and read_wiki_page "
            "only when a whole page is needed. Keep using tools until you have "
            "the answer or the available tools clearly cannot answer. Return "
            "only the final answer using this exact format: "
            "FINAL ANSWER: <answer>\n\n"
            f"Question: {task.question}"
            f"{attachment_note}"
        )
        if not isinstance(result.output, str):
            raise AssertionError(
                f"agent did not return a final string: {result.output!r}"
            )
        return result

    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        allowed_tools=GAIA_COMPUTE_WIKIPEDIA_TOOLS,
        trace_output_path=GAIA_COMPUTE_WIKIPEDIA_TRACE_PATH,
    )
    logger.info(
        "evaluation={} trace_output_path={}",
        evaluation,
        GAIA_COMPUTE_WIKIPEDIA_TRACE_PATH,
    )

    assert evaluation.total_tasks > 0
    assert evaluation.scored_tasks == evaluation.total_tasks
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0
    assert all(result.error is None for result in evaluation.results)


@pytest.mark.slow
def test_functional_qwen3_06b_no_harness_gaia_validation(
    qwen3_06b_gaia_generator: Generator,
) -> None:
    def agent(task: GaiaTask) -> str:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else ""
        )
        messages = [
                {
                    "role": "user",
                    "content":"Answer this GAIA benchmark question. Return only the "
                    "final answer using this exact format: FINAL ANSWER: <answer>\n\n"
                    f"Question: {task.question}\n"
                    f"{attachment_note}"
                }
        ]
        completion = qwen3_06b_gaia_generator.generate_completion(
            messages,
            max_generated_token=4096,
            temperature=0.6, top_p=0.95, top_k=20
        )
        logger.info("Generated completion = [{}]", completion)
        return completion.message.content

    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        level=1,
        limit=1,
    )
    logger.info("evaluation={}", evaluation)

    #assert evaluation.total_tasks == 1
    #assert evaluation.scored_tasks == 1
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0

    result = evaluation.results[0]
    assert result.task_id
    assert result.question
    assert result.prediction
    assert result.error is None


@pytest.mark.slow
def test_functional_qwen3_4b_gaia_validation(
    qwen3_4b_gaia_agent: Agent,
) -> None:
    def agent(task: GaiaTask) -> str:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else ""
        )
        #TODO: GAIA output format should be part of gaia contract and derived
        #from a pydantic type.
        # TODO add informations like: is_solvable: bool and unsolvability_reason:str
        result = qwen3_4b_gaia_agent.run(
            "Answer this GAIA benchmark question. Use the compute tool when "
            "arithmetic or exact calculation is needed. Use the wiki tool when "
            "factual or external to your knowledge informations are needed. "
            "Return only the final "
            "answer using this exact format: FINAL ANSWER: <answer>\n\n"
            f"Question: {task.question}"
            f"{attachment_note}"
        )
        if not isinstance(result.output, str):
            raise AssertionError(
                f"agent did not return a final string: {result.output!r}"
            )
        return result.output.strip()

    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        level=1,
        limit=1,
    )
    logger.info("evaluation={}", evaluation)

    #assert evaluation.total_tasks == 1
    #assert evaluation.scored_tasks == 1
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0

    result = evaluation.results[0]
    assert result.task_id
    assert result.question
    assert result.prediction
    assert result.error is None
