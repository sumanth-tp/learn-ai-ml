"""Structured outputs the model must return. Field descriptions are sent to the model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StandaloneQuestion(BaseModel):
    """A follow-up question rewritten so it can be answered without the conversation."""

    question: str = Field(description="The question rewritten to stand on its own.")
    is_follow_up: bool = Field(description="True if it depended on earlier turns.")


class QueryPlan(BaseModel):
    """How to answer the question with SQL, before writing any SQL."""

    tables: list[str] = Field(description="Tables needed, from the provided schema only.")
    steps: list[str] = Field(description="Ordered steps: filters, joins, aggregations, sort.")
    metric_definition: str = Field(description="How the main metric is computed.")


class SQLDraft(BaseModel):
    """A single read-only DuckDB SELECT statement answering the question."""

    sql: str = Field(description="One DuckDB SELECT statement. No comments, no semicolon.")
    explanation: str = Field(description="One sentence on what the query computes.")


class Interpretation(BaseModel):
    """A plain-English answer grounded in the returned rows."""

    answer: str = Field(description="Two to four sentences answering the question from the rows.")
    chart_recommended: bool = Field(description="True if a chart would help (2+ rows, a metric).")


class ChartCode(BaseModel):
    """Python that draws a chart from a pandas DataFrame named df."""

    code: str = Field(
        description=(
            "matplotlib code using the existing names df (pandas DataFrame), plt and pd. "
            "Do not import anything, read or write files, or call savefig."
        )
    )
    title: str = Field(description="Chart title.")
