def evaluate_ideas(state: CriticState) -> Dict:
    response = llm.invoke(evaluation_prompt.format_messages(
        ideas=state.get("shared_context", {}).get("research_ideas", []),
        gap_analysis=state.get("shared_context", {}).get("gap_analysis", {})
    )) 