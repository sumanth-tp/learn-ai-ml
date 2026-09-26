"""A real LLM judge: fixed temperature, structured output, retries, explicit errors."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable
from pydantic import ValidationError

from ragate.judge.base import JudgeError, JudgeOutput, Verdict
from ragate.judge.prompts import JudgePrompt
from ragate.retry import RetryExhaustedError, call_with_retries


class LLMJudge:
    def __init__(self, model: BaseChatModel, model_name: str, temperature: float,
                 attempts: int = 3) -> None:
        self._structured = model.with_structured_output(JudgeOutput)
        self._model_id = f"{model_name}@t{temperature:g}"
        self.attempts = attempts

    @property
    def model_id(self) -> str:
        return self._model_id

    @traceable(name="judge", run_type="llm")
    def evaluate(self, prompt: JudgePrompt, variables: dict[str, str]) -> Verdict:
        text = prompt.render(variables)

        def once() -> JudgeOutput:
            out = self._structured.invoke(text)
            if not isinstance(out, JudgeOutput):  # some providers return dicts
                out = JudgeOutput.model_validate(out)
            return out

        try:
            out = call_with_retries(once, what=f"judge:{prompt.name}", attempts=self.attempts)
        except RetryExhaustedError as exc:
            raise JudgeError(str(exc)) from exc
        except ValidationError as exc:  # pragma: no cover - covered by retry path
            raise JudgeError(f"invalid judge output: {exc}") from exc
        return Verdict(score=out.score / 10, reason=out.reason)
