import dspy
from numpy import mean
from autonogy_constructor.signatures import ExtractOntologyElements, Assess
from autonogy_constructor.utils import ontology_to_string

from config.settings import ASSESSMENT_CRITERIA_CONFIG


entity = ASSESSMENT_CRITERIA_CONFIG["entity"]
entity_score = ASSESSMENT_CRITERIA_CONFIG["entity_score"]
hierachy = ASSESSMENT_CRITERIA_CONFIG["hierachy"]
hierachy_score = ASSESSMENT_CRITERIA_CONFIG["hierachy_score"]
disjointness = ASSESSMENT_CRITERIA_CONFIG["disjointness"]
disjointness_score = ASSESSMENT_CRITERIA_CONFIG["disjointness_score"]
data_property = ASSESSMENT_CRITERIA_CONFIG["data_property"]
data_property_score = ASSESSMENT_CRITERIA_CONFIG["data_property_score"]
object_property = ASSESSMENT_CRITERIA_CONFIG["object_property"]
object_property_score = ASSESSMENT_CRITERIA_CONFIG["object_property_score"]
ontology_structure = ASSESSMENT_CRITERIA_CONFIG["ontology_structure"]
ontology_structure_score = ASSESSMENT_CRITERIA_CONFIG["ontology_structure_score"]
overall_content = ASSESSMENT_CRITERIA_CONFIG["overall_content"]
overall_content_score = ASSESSMENT_CRITERIA_CONFIG["overall_content_score"]
full_score = ASSESSMENT_CRITERIA_CONFIG["full_score"]

class ChemOntology(dspy.Module):
    def __init__(self):
        super().__init__()

        self.extractor = dspy.ChainOfThought(ExtractOntologyElements)
    
    def forward(self, context):
        return self.extractor(text=context)

class Assessment(dspy.Module):
    def __init__(self, verbose=False, assertions=False):
        super().__init__()
        self.assessor = dspy.ChainOfThought(Assess)
        self.verbose = verbose
        self.assertions = assertions
    
    def forward(self, assessed_text, assessment_ontology):
        verbose = self.verbose
        assertions = self.assertions
        score_list = [
            self.assessor(assessed_text=assessed_text, assessment_ontology=ontology_to_string(assessment_ontology), assessment_criteria=criteria).assessment_score for criteria in [entity, data_property, object_property, ontology_structure, overall_content]
        ]
        print(score_list)
        normalized_score_list = [
            score_list[0]/entity_score,
            score_list[1]/data_property_score,
            score_list[2]/object_property_score,
            score_list[3]/ontology_structure_score,
            score_list[4]/overall_content_score
        ]
        if verbose or assertions:
            print("reason_list")
            reason_list = [
                self.assessor(assessed_text=assessed_text, assessment_ontology=ontology_to_string(assessment_ontology), assessment_criteria=criteria).assessment_reason for criteria in [entity, data_property, object_property, ontology_structure, overall_content]
            ]
        if assertions:
            print("assertions")
            return (
                {
                "entity": (score_list[0] >= (entity_score-0.5), reason_list[0]),
                "data_property": (score_list[1] >= (data_property_score-0.5), reason_list[1]),
                "object_property": (score_list[2] >= (object_property_score-0.5), reason_list[2]),
                "ontology_structure": (score_list[3] >= (ontology_structure_score-0.5), reason_list[3]),
                "overall_content": (score_list[4] >= (overall_content_score-4), reason_list[4]),
                },
                f"""
Entity Score: {normalized_score_list[0]}
Data Property Score: {normalized_score_list[1]}
Object Property Score: {normalized_score_list[2]}
Ontology Structure Score: {normalized_score_list[3]}
Overall Content Score: {normalized_score_list[4]}

Total Score: {sum(score_list)}/{full_score}
Percentage Score: {mean(normalized_score_list)*100:.2f}%

Reason:
Entity: {reason_list[0]}
Data Property: {reason_list[1]}
Object Property: {reason_list[2]}
Ontology Structure: {reason_list[3]}
Overall Content: {reason_list[4]}
"""
            )
        if verbose:
            print("verbose")
            return f"""
Entity Score: {normalized_score_list[0]}
Data Property Score: {normalized_score_list[1]}
Object Property Score: {normalized_score_list[2]}
Ontology Structure Score: {normalized_score_list[3]}
Overall Content Score: {normalized_score_list[4]}

Total Score: {sum(score_list)}/{full_score}
Percentage Score: {mean(normalized_score_list)*100:.2f}%

Reason:
Entity: {reason_list[0]}
Data Property: {reason_list[1]}
Object Property: {reason_list[2]}
Ontology Structure: {reason_list[3]}
Overall Content: {reason_list[4]}
"""
        return mean(normalized_score_list)