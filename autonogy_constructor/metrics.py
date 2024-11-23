from autonogy_constructor.modules import Assessment

def assessor_metric(gold, pred, trace=None):
    standard_score = gold['score']
    assessor_score = pred['assessment_score']
    return standard_score == assessor_score

def metric(gold, pred, trace=None, verbose=False):
    assessment = Assessment()
    res = assessment(assessed_text=gold['context'], assessment_ontology=pred['ontology'])
    print(res)
    return res