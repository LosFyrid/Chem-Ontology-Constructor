from autology_constructor.idea.common.llm_helpers import invoke_llm

class BaseFinder:
    def generate_ideas(self, gaps: List[Dict]) -> List[Dict]:
        prompt = self._get_ideation_prompt()
        content = invoke_llm(prompt, gaps=gaps)
        return self._parse_ideas(content)

    def analyze_gaps(self, domain_analysis: Dict) -> List[Dict]:
        prompt = self._get_analysis_prompt()
        content = invoke_llm(prompt, analysis=domain_analysis)
        return self._parse_gaps(content) 