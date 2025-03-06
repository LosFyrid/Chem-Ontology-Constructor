from owlready2 import *

from config.settings import ONTOLOGY_CONFIG

onto = ONTOLOGY_CONFIG["ontology"]
class_namespace = ONTOLOGY_CONFIG["classes"]

class1 = class_namespace["benzene_ring"]
class2 = class_namespace["alkyl_groups"]

op = onto["test_op"]

restriction_expr = op.only(class2)
class1.is_a.append(restriction_expr)


with onto:
    AnnotatedRelation(class1, rdfs_subclassof, restriction_expr).comment = ["test1"]

with onto:
    gca = GeneralClassAxiom(And([class1, class2]))
    gca.is_a.append(restriction_expr)
    gca.information = ["test"]
print(class1.is_a)

onto.save()