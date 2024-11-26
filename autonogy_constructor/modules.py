import dspy
from numpy import mean
from autonogy_constructor.base_data_structures import OntologyElements, OntologyProperties
from autonogy_constructor.signatures import ExtractOntologyElements, ExtractOntologyProperties, Assess
from autonogy_constructor.utils import ontology_elements_to_string, ontology_properties_to_string

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

        self.elements_extractor = dspy.ChainOfThought(ExtractOntologyElements)
        self.properties_extractor = dspy.ChainOfThought(ExtractOntologyProperties)
    
    def forward(self, context):
        elements = self.elements_extractor(text=context)
        properties = self.properties_extractor(text=context, existing_ontology_elements=elements.ontology_elements)
        return dspy.Prediction(context=context, ontology_elements=elements.ontology_elements, ontology_properties=properties.ontology_properties)

class Assessment(dspy.Module):
    def __init__(self, verbose=False, assertions=False):
        super().__init__()
        self.assessor = dspy.ChainOfThought(Assess)
        self.verbose = verbose
        self.assertions = assertions
    
    def forward(self, assessed_text, assessment_ontology):
        verbose = self.verbose
        assertions = self.assertions
        if isinstance(assessment_ontology, OntologyElements):
            print("elements")
            assessment_ontology = ontology_elements_to_string(assessment_ontology)
            print(assessment_ontology)
            criteria_list = [entity, hierachy, disjointness]
            score_list = [
                self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_score for criteria in criteria_list
            ]
            score_denominators = [entity_score, hierachy_score, disjointness_score]
            normalized_score_list = [score/denom for score, denom in zip(score_list, score_denominators)]
            if verbose or assertions:
                reason_list = [
                    self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_reason for criteria in criteria_list
                ]
        elif isinstance(assessment_ontology, OntologyProperties):
            print("properties")
            assessment_ontology = ontology_properties_to_string(assessment_ontology)
            print(assessment_ontology)
            criteria_list = [data_property, object_property]
            score_list = [
                self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_score for criteria in criteria_list
            ]
            score_denominators = [data_property_score, object_property_score]
            normalized_score_list = [score/denom for score, denom in zip(score_list, score_denominators)]
            if verbose or assertions:
                reason_list = [
                    self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_reason for criteria in criteria_list
                ]
        elif isinstance(assessment_ontology, tuple) and len(assessment_ontology) == 2:
            if isinstance(assessment_ontology[0], OntologyElements) and isinstance(assessment_ontology[1], OntologyProperties):
                print("elements and properties")
                assessment_ontology = ontology_elements_to_string(assessment_ontology[0].entities) + "\n" + ontology_properties_to_string(assessment_ontology[1])
                criteria_list = [ontology_structure, overall_content]
                score_list = [
                    self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_score for criteria in criteria_list
                ]
                score_denominators = [ontology_structure_score, overall_content_score]
                normalized_score_list = [score/denom for score, denom in zip(score_list, score_denominators)]
                if verbose or assertions:
                    reason_list = [
                        self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_reason for criteria in criteria_list
                    ]
        else:
            raise ValueError("assessment_ontology 必须是 OntologyElements、OntologyProperties 或者它们的元组类型")
        if assertions:
            result_dict = {}
            for i, criteria_name in enumerate(criteria_list):
                name = criteria_name.split('\n')[0].split('(')[0].strip().lower().replace(' ', '_')
                threshold = score_denominators[i] - (4 if name == 'overall_content' else 0.5)
                result_dict[name] = (score_list[i] >= threshold, reason_list[i])

            score_text = '\n'.join([f"{criteria.split('(')[0].strip()} Score: {score}" for criteria, score in zip(criteria_list, normalized_score_list)])
            reason_text = '\n'.join([f"{criteria.split('(')[0].strip()}: {reason}" for criteria, reason in zip(criteria_list, reason_list)])

            return (
                result_dict,
                f"""
{score_text}

Total Score: {sum(score_list)}/{sum(score_denominators)}
Percentage Score: {mean(normalized_score_list)*100:.2f}%

Reason:
{reason_text}
"""
            )

        if verbose:
            score_text = '\n'.join([f"{criteria.split('(')[0].strip()} Score: {score}" for criteria, score in zip(criteria_list, normalized_score_list)])
            reason_text = '\n'.join([f"{criteria.split('(')[0].strip()}: {reason}" for criteria, reason in zip(criteria_list, reason_list)])

            return f"""
{score_text}

Total Score: {sum(score_list)}/{sum(score_denominators)}
Percentage Score: {mean(normalized_score_list)*100:.2f}%

Reason:
{reason_text}
"""

        return mean(normalized_score_list)