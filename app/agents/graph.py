from langgraph.graph import StateGraph, END
from app.agents.state import SupportState
from app.agents.nodes.intent_classifier import IntentClassifierNode
from app.agents.nodes.rag_retriever import RAGRetrieverNode
from app.agents.nodes.response_generator import ResponseGeneratorNode
from app.agents.nodes.escalation_handler import EscalationHandlerNode
from app.agents.nodes.ticket_manager import TicketManagerNode


def build_graph():
    workflow = StateGraph(SupportState)

    workflow.add_node("intent_classifier", IntentClassifierNode().run)
    workflow.add_node("rag_retriever", RAGRetrieverNode().run)
    workflow.add_node("response_generator", ResponseGeneratorNode().run)
    workflow.add_node("escalation_handler", EscalationHandlerNode().run)
    workflow.add_node("ticket_manager", TicketManagerNode().run)

    workflow.set_entry_point("intent_classifier")
    workflow.add_edge("intent_classifier", "rag_retriever")
    workflow.add_edge("rag_retriever", "response_generator")
    workflow.add_conditional_edges(
        "response_generator",
        EscalationHandlerNode().should_escalate,
        {True: "escalation_handler", False: "ticket_manager"},
    )
    workflow.add_edge("escalation_handler", "ticket_manager")
    workflow.add_edge("ticket_manager", END)

    return workflow.compile()