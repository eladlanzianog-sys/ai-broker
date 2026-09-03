"""LangGraph graph definition.

Wires all agent nodes into a StateGraph with fan-out/fan-in:
  START -> orchestrator_entry -> market_data_collector
        -> [technical_analyst, fundamental_analyst, sentiment_analyst] (parallel)
        -> risk_manager -> portfolio_strategist -> orchestrator_exit -> END
"""
from langgraph.graph import END, START, StateGraph

from src.agents.fundamental_analyst import analyze_fundamentals
from src.agents.market_data_collector import collect_market_data
from src.agents.orchestrator import orchestrator_entry, orchestrator_exit
from src.agents.portfolio_strategist import synthesize_recommendation
from src.agents.risk_manager import assess_risk
from src.agents.sentiment_analyst import analyze_sentiment
from src.agents.state import AnalysisState
from src.agents.technical_analyst import analyze_technicals


def build_analysis_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("orchestrator_entry", orchestrator_entry)
    graph.add_node("market_data_collector", collect_market_data)
    graph.add_node("technical_analyst", analyze_technicals)
    graph.add_node("fundamental_analyst", analyze_fundamentals)
    graph.add_node("sentiment_analyst", analyze_sentiment)
    graph.add_node("risk_manager", assess_risk)
    graph.add_node("portfolio_strategist", synthesize_recommendation)
    graph.add_node("orchestrator_exit", orchestrator_exit)

    graph.add_edge(START, "orchestrator_entry")
    graph.add_edge("orchestrator_entry", "market_data_collector")

    # Fan-out: data collector feeds three analysts in parallel
    graph.add_edge("market_data_collector", "technical_analyst")
    graph.add_edge("market_data_collector", "fundamental_analyst")
    graph.add_edge("market_data_collector", "sentiment_analyst")

    # Fan-in: all three analysts feed risk manager
    graph.add_edge("technical_analyst", "risk_manager")
    graph.add_edge("fundamental_analyst", "risk_manager")
    graph.add_edge("sentiment_analyst", "risk_manager")

    graph.add_edge("risk_manager", "portfolio_strategist")
    graph.add_edge("portfolio_strategist", "orchestrator_exit")
    graph.add_edge("orchestrator_exit", END)

    return graph


def compile_graph():
    graph = build_analysis_graph()
    return graph.compile()
