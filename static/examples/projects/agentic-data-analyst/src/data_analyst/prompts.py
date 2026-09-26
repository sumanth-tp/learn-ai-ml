"""Prompt templates. Kept apart from the nodes so they can be versioned and evaluated."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

PROMPT_VERSION = "2026-09-v1"

REWRITE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions about business data so they stand alone. "
            "Keep every filter, metric and grouping from the earlier question unless the "
            "follow-up changes it. If the question already stands alone, return it unchanged.",
        ),
        ("human", "Earlier turns (oldest first):\n{history}\n\nFollow-up: {question}"),
    ]
)

PLAN = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior analytics engineer. Plan how to answer the question with SQL "
            "over ONLY the tables below. Use the metric definitions in the table descriptions.\n\n"
            "{schema}",
        ),
        ("human", "{question}"),
    ]
)

GENERATE_SQL = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write ONE DuckDB SELECT statement that answers the question. Rules:\n"
            "- use only the tables and columns below; never invent columns\n"
            "- read-only: no INSERT, UPDATE, DELETE, DDL, PRAGMA, SET, COPY or ATTACH\n"
            "- no table functions (read_csv, query, glob ...)\n"
            "- give every computed column a readable alias; ROUND money to 2 decimals\n"
            "- add ORDER BY whenever the question implies a ranking or a time series\n"
            "- PII columns are masked; never try to unmask them\n\n"
            "{schema}\n\nPlan:\n{plan}",
        ),
        ("human", "Question: {question}{feedback}"),
    ]
)

FEEDBACK = (
    "\n\nYour previous attempt failed.\nPrevious SQL:\n{sql}\nError:\n{error}\n"
    "Fix the query. Do not repeat the same mistake."
)

INTERPRET = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the business question using only the query result below. Quote the "
            "numbers that matter. If the result was truncated, say so. Do not speculate "
            "beyond the data.",
        ),
        (
            "human",
            "Question: {question}\nSQL: {sql}\nColumns: {columns}\nRows (first {shown} of "
            "{total}{truncated}):\n{rows}",
        ),
    ]
)

CHART = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write matplotlib code that draws one clear chart of the DataFrame df. The names "
            "df, pd, np and plt already exist. Do not import, open files, call savefig or show. "
            "Label the axes and set a title.",
        ),
        ("human", "Question: {question}\nColumns and dtypes: {dtypes}\nFirst rows:\n{head}"),
    ]
)
