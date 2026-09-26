"""Wire the nodes into a LangGraph StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from data_analyst.graph.nodes import AnalystNodes, Deps
from data_analyst.graph.state import AnalystState


def build_graph(deps: Deps, checkpointer: Checkpointer = None) -> CompiledStateGraph:
    n = AnalystNodes(deps)
    g = StateGraph(AnalystState)
    g.add_node("contextualise", n.contextualise)
    g.add_node("cache_lookup", n.cache_lookup)
    g.add_node("retrieve_schema", n.retrieve_schema)
    g.add_node("plan", n.plan)
    g.add_node("generate_sql", n.generate_sql)
    g.add_node("validate", n.validate)
    g.add_node("estimate", n.estimate)
    g.add_node("approval", n.approval)
    g.add_node("execute", n.execute)
    g.add_node("interpret", n.interpret)
    g.add_node("chart", n.chart)
    g.add_node("finalize", n.finalize)

    g.add_edge(START, "contextualise")
    g.add_edge("contextualise", "cache_lookup")
    g.add_conditional_edges("cache_lookup", n.route_after_cache, ["validate", "retrieve_schema"])
    g.add_edge("retrieve_schema", "plan")
    g.add_edge("plan", "generate_sql")
    g.add_edge("generate_sql", "validate")
    g.add_conditional_edges(
        "validate",
        n.route_after_validate,
        ["estimate", "generate_sql", "retrieve_schema", "finalize"],
    )
    g.add_conditional_edges(
        "estimate",
        n.route_after_estimate,
        ["approval", "execute", "generate_sql", "retrieve_schema", "finalize"],
    )
    g.add_conditional_edges("approval", n.route_after_approval, ["execute", "finalize"])
    g.add_conditional_edges(
        "execute",
        n.route_after_execute,
        ["interpret", "generate_sql", "retrieve_schema", "finalize"],
    )
    g.add_conditional_edges("interpret", n.route_after_interpret, ["chart", "finalize"])
    g.add_edge("chart", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name="data-analyst")
