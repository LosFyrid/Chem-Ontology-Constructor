def analyze_domain(state: DreamerState) -> Dict:
    return {
        "stage": "dreaming",
        "previous_stage": state.get("stage"),
        "status": "success",
        "shared_context": {
            "ontology_analysis": domain_analysis,
            "gap_analysis": gaps,
            "research_ideas": research_ideas
        },
        "messages": [
            f"Generated {len(research_ideas)} research ideas",
            f"Analyzed {len(gaps)} gap categories"
        ]
    }

async def assess_information_sufficiency(state: DreamerState) -> Dict:
    analysis = state.get("shared_context", {}).get("ontology_analysis", {}) 