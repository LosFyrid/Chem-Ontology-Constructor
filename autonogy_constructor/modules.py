from numpy import mean

import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.predict import Retry


from autonogy_constructor.base_data_structures import OntologyElements, OntologyDataProperties, OntologyObjectProperties, OntologyEntities
from autonogy_constructor.signatures import ExtractOntologyElements, ExtractOntologyDataProperties, ExtractOntologyObjectProperties, ExtractOntologyEntities, Assess
from autonogy_constructor.utils import ontology_entities_to_string, ontology_elements_to_string, ontology_data_properties_to_string, ontology_object_properties_to_string

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

def my_backtrack_handler(func):
    return backtrack_handler(func, max_backtracks=5)

class ChemOntology(dspy.Module):
    def __init__(self):
        super().__init__()

        self.entities_extractor = dspy.ChainOfThought(ExtractOntologyEntities)
        self.elements_extractor = dspy.ChainOfThought(ExtractOntologyElements)
        self.data_properties_extractor = dspy.ChainOfThought(ExtractOntologyDataProperties)
        self.object_properties_extractor = dspy.ChainOfThought(ExtractOntologyObjectProperties)
    
    def forward(self, context):
        entities = self.entities_extractor(text=context)
        elements = self.elements_extractor(text=context, ontology_entities=entities.ontology_entities)
        data_properties = self.data_properties_extractor(text=context, ontology_entities=entities.ontology_entities)
        object_properties = self.object_properties_extractor(text=context, ontology_entities=entities.ontology_entities)
        return dspy.Prediction(context=context, ontology_entities=entities.ontology_entities, ontology_elements=elements.ontology_elements, ontology_data_properties=data_properties.ontology_data_properties, ontology_object_properties=object_properties.ontology_object_properties)

class Assessment(dspy.Module):
    def __init__(self, verbose=False, assertions=False):
        super().__init__()
        self.assessor = dspy.ChainOfThought(Assess)
        self.verbose = verbose
        self.assertions = assertions
    
    def forward(self, assessed_text, assessment_ontology):
        verbose = self.verbose
        assertions = self.assertions
        if isinstance(assessment_ontology, OntologyEntities):
            print("entities")
            assessment_ontology = ontology_entities_to_string(assessment_ontology)
            print(assessment_ontology)
            criteria_list = [entity]
            score_list = [
                self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_score for criteria in criteria_list
            ]
            score_denominators = [entity_score]
            normalized_score_list = [score/denom for score, denom in zip(score_list, score_denominators)]
            if verbose or assertions:
                reason_list = [
                    self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_reason for criteria in criteria_list
                ]
        elif isinstance(assessment_ontology, OntologyElements):
            print("elements")
            assessment_ontology = ontology_elements_to_string(assessment_ontology)
            print(assessment_ontology)
            criteria_list = [hierachy, disjointness]
            score_list = [
                self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_score for criteria in criteria_list
            ]
            score_denominators = [hierachy_score, disjointness_score]
            normalized_score_list = [score/denom for score, denom in zip(score_list, score_denominators)]
            if verbose or assertions:
                reason_list = [
                    self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_reason for criteria in criteria_list
                ]
        elif isinstance(assessment_ontology, (OntologyDataProperties, OntologyObjectProperties)):
            print("properties")
            if isinstance(assessment_ontology, OntologyDataProperties):
                assessment_ontology = ontology_data_properties_to_string(assessment_ontology)
                criteria_list = [data_property]
                score_denominators = [data_property_score]
            else:
                assessment_ontology = ontology_object_properties_to_string(assessment_ontology)
                criteria_list = [object_property]
                score_denominators = [object_property_score]
            print(assessment_ontology)
            score_list = [
                self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_score for criteria in criteria_list
            ]
            normalized_score_list = [score/denom for score, denom in zip(score_list, score_denominators)]
            if verbose or assertions:
                reason_list = [
                    self.assessor(assessed_text=assessed_text, assessment_ontology=assessment_ontology, assessment_criteria=criteria).assessment_reason for criteria in criteria_list
                ]
        elif isinstance(assessment_ontology, tuple) and len(assessment_ontology) == 4:
            if isinstance(assessment_ontology[0], OntologyEntities) and isinstance(assessment_ontology[1], OntologyElements) and isinstance(assessment_ontology[2], OntologyDataProperties) and isinstance(assessment_ontology[3], OntologyObjectProperties):
                print("entities, elements and properties")
                assessment_ontology = ontology_entities_to_string(assessment_ontology[0]) + "\n" + ontology_elements_to_string(assessment_ontology[1]) + "\n" + ontology_data_properties_to_string(assessment_ontology[2]) + "\n" + ontology_object_properties_to_string(assessment_ontology[3])
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
            raise ValueError("assessment_ontology 必须是 OntologyEntities、OntologyElements、OntologyDataProperties、OntologyObjectProperties 或者它们的元组类型")
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

class ChemOntologyWithEntitiesAssertions(dspy.Module):
    def __init__(self):
        super().__init__()

        self.entities_extractor = dspy.ChainOfThought(ExtractOntologyEntities)
        self.assessor = Assessment(assertions=True)
    
    def forward(self, context):
        entities = self.entities_extractor(text=context)
        assertions, score_info = self.assessor(assessed_text=context, assessment_ontology=entities.ontology_entities)
        qualified = [x[0] for x in assertions.values()]
        suggestion = [x[1] for x in assertions.values()]
        dspy.Suggest(all(qualified),"".join(suggestion),target_module=self.entities_extractor)
        return dspy.Prediction(context=context, ontology_entities=entities.ontology_entities, score_info=score_info)

class ChemOntologyWithElementsAssertions(dspy.Module):
    def __init__(self):
        super().__init__()

        self.elements_extractor = dspy.ChainOfThought(ExtractOntologyElements)
        self.assessor = Assessment(assertions=True)
    
    def forward(self, context, entities):
        elements = self.elements_extractor(text=context, ontology_entities=entities.ontology_entities)
        assertions, score_info = self.assessor(assessed_text=context, assessment_ontology=elements.ontology_elements)
        qualified = [x[0] for x in assertions.values()]
        suggestion = [x[1] for x in assertions.values()]
        dspy.Suggest(all(qualified),"".join(suggestion),target_module=self.elements_extractor)

        return dspy.Prediction(context=context, ontology_elements=elements.ontology_elements, score_info=score_info)

class ChemOntologyWithDataPropertiesAssertions(dspy.Module):
    def __init__(self):
        super().__init__()

        self.data_properties_extractor = dspy.ChainOfThought(ExtractOntologyDataProperties)
        self.assessor = Assessment(assertions=True)

    def forward(self, context, entities):
        data_properties = self.data_properties_extractor(text=context, ontology_entities=entities.ontology_entities)
        assertions, score_info = self.assessor(assessed_text=context, assessment_ontology=data_properties.ontology_data_properties)
        qualified = [x[0] for x in assertions.values()]
        suggestion = [x[1] for x in assertions.values()]
        dspy.Suggest(all(qualified),"".join(suggestion),target_module=self.data_properties_extractor)
        return dspy.Prediction(context=context, ontology_entities=entities.ontology_entities, ontology_data_properties=data_properties.ontology_data_properties, score_info=score_info)

class ChemOntologyWithObjectPropertiesAssertions(dspy.Module):
    def __init__(self):
        super().__init__()

        self.object_properties_extractor = dspy.ChainOfThought(ExtractOntologyObjectProperties)
        self.assessor = Assessment(assertions=True)

    def forward(self, context, entities):
        object_properties = self.object_properties_extractor(text=context, ontology_entities=entities.ontology_entities)
        assertions, score_info = self.assessor(assessed_text=context, assessment_ontology=object_properties.ontology_object_properties)
        qualified = [x[0] for x in assertions.values()]
        suggestion = [x[1] for x in assertions.values()]
        dspy.Suggest(all(qualified),"".join(suggestion),target_module=self.object_properties_extractor)
        return dspy.Prediction(context=context, ontology_entities=entities.ontology_entities, ontology_object_properties=object_properties.ontology_object_properties, score_info=score_info)


chemonto_with_entities_assertions = assert_transform_module(ChemOntologyWithEntitiesAssertions().map_named_predictors(Retry), backtrack_handler) 
chemonto_with_elements_assertions = assert_transform_module(ChemOntologyWithElementsAssertions().map_named_predictors(Retry), backtrack_handler) 
chemonto_with_data_properties_assertions = assert_transform_module(ChemOntologyWithDataPropertiesAssertions().map_named_predictors(Retry), backtrack_handler) 
chemonto_with_object_properties_assertions = assert_transform_module(ChemOntologyWithObjectPropertiesAssertions().map_named_predictors(Retry), backtrack_handler) 
