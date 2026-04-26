from langgraph.graph import END, StateGraph

from healthcare_app.schemas import HealthcareState


def route_by_intent(state: HealthcareState) -> str:
    mapping = {
        "query": "query_agent",
        "audit": "audit_agent",
        "desert": "desert_finder",
        "trust": "trust_scorer",
        "out_of_scope": "out_of_scope_handler",
    }
    return mapping.get(state.get("intent", "out_of_scope"), "out_of_scope_handler")


def build_graph(runtime):
    workflow = StateGraph(HealthcareState)
    workflow.add_node("detect_intent", runtime.detect_intent)
    workflow.add_node("query_agent", runtime.query_agent)
    workflow.add_node("audit_agent", runtime.audit_agent)
    workflow.add_node("trust_scorer", runtime.trust_scorer)
    workflow.add_node("desert_finder", runtime.desert_finder)
    workflow.add_node("validator_agent", runtime.validator_agent)
    workflow.add_node("response_builder", runtime.response_builder)
    workflow.add_node("out_of_scope_handler", runtime.out_of_scope_handler)

    workflow.set_entry_point("detect_intent")
    workflow.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "query_agent": "query_agent",
            "audit_agent": "audit_agent",
            "trust_scorer": "trust_scorer",
            "desert_finder": "desert_finder",
            "out_of_scope_handler": "out_of_scope_handler",
        },
    )

    for node in ["query_agent", "audit_agent", "trust_scorer", "desert_finder"]:
        workflow.add_edge(node, "validator_agent")
    workflow.add_edge("validator_agent", "response_builder")
    workflow.add_edge("response_builder", END)
    workflow.add_edge("out_of_scope_handler", END)
    return workflow.compile()

