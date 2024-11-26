from autonogy_constructor.modules import Assessment
from config.settings import ASSESSMENT_CRITERIA_CONFIG

def assessor_metric(gold, pred, trace=None):
    standard_score = gold['score']
    assessor_score = pred['assessment_score']
    return standard_score == assessor_score

def metric(gold, pred, trace=None, verbose=False):
    weights = ASSESSMENT_CRITERIA_CONFIG["weights"]
    assessment = Assessment()
    elements_score = assessment(assessed_text=gold['context'], assessment_ontology=pred['ontology_elements'])
    properties_score = assessment(assessed_text=gold['context'], assessment_ontology=pred['ontology_properties'])
    overall_score = assessment(assessed_text=gold['context'], assessment_ontology=(pred['ontology_elements'], pred['ontology_properties']))
    res = (elements_score * weights["elements"] + properties_score * weights["properties"] + overall_score * weights["overall"])
    # print(res)
    return res