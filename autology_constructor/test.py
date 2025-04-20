from owlready2 import *
import owlready2.base
import owlready2.class_construct
import pytest
# Removed tempfile, os, shutil as they are no longer needed for temp ontology

from config.settings import OntologySettings # Keep this
# Keep OntologyTools import
from autology_constructor.idea.query_team.ontology_tools import OntologyTools

# Removed manual loading and printing section


# --- Pytest Fixture for Real Ontology ---

@pytest.fixture(scope="module")
def real_ontology_settings():
    """Provides the OntologySettings instance pointing to the real backup-2.owl."""
    # It might be slightly safer to re-instantiate here within the fixture scope
    # to ensure a clean load for the test module, though the top-level one might also work.
    settings = OntologySettings(
        base_iri="http://www.test.org/chem_ontologies/",
        ontology_file_name="backup-2.owl",
        directory_path="data/ontology/", # Ensure this path is correct relative to execution
        closed_ontology_file_name=None # Assuming None is correct for your setup
    )
    # No yield/teardown needed here unless we want specific cleanup for the real settings object
    return settings


# Removed the old test_ontology_settings fixture


@pytest.fixture(scope="module")
def ontology_tools(real_ontology_settings): # Depends on the real settings now
    """Provides an OntologyTools instance initialized with the real ontology."""
    assert real_ontology_settings.ontology is not None, "Ontology failed to load in real_ontology_settings fixture."
    # Add a check for critical namespaces to ensure they loaded correctly
    assert real_ontology_settings.meta is not None, "Meta namespace failed to load."
    assert real_ontology_settings.classes is not None, "Classes namespace failed to load."
    assert real_ontology_settings.object_properties is not None, "Object Properties namespace failed to load."
    assert real_ontology_settings.data_properties is not None, "Data Properties namespace failed to load."

    tools = OntologyTools(real_ontology_settings)
    # Check if tools initialized correctly with expected components
    assert tools.onto is not None, "OntologyTools did not initialize with an ontology."
    assert tools.meta_ns is not None, "OntologyTools did not initialize with meta namespace."
    assert tools.SourcedInformationClass is not None, "OntologyTools could not find SourcedInformation class in meta namespace."
    assert tools.has_information_prop is not None, "OntologyTools could not find has_information property in meta namespace."

    return tools

# --- Test Class ---

class TestOntologyTools:

    # Use names provided by the user
    EXISTING_CLASS = "calix(4)pyrrole"
    EXISTING_SUBCLASS = "aryl-extended_calix(4)pyrrole"
    EXISTING_PARENT = EXISTING_CLASS
    CLASS_WITH_CHILDREN = "macrocyclic_receptor"
    EXPECTED_CHILD = "calix(4)arene" # One of the children
    NON_EXISTENT_CLASS = "NonExistentClass"
    EXISTING_OBJ_PROP = "is_capable_of_binding"
    EXISTING_DATA_PROP = "binding_energy"
    META_OBJ_PROP = "has_information"
    META_DATA_PROP = "content"
    NON_EXISTENT_PROP = "nonExistentProperty"
    CLASS_FOR_PROPS_TEST = "super_aryl-extended_calix(4)pyrrole"
    CLASS_FOR_RESTRICTION_TEST = CLASS_FOR_PROPS_TEST
    PROP_FOR_RESTRICTION_TEST = "acts_as_protecting_group"
    EXPECTED_RESTRICTION_VALUE_CLASS = "bis-isonitrile"
    CLASS_FOR_RELATIONS_TEST = "macrocyclic_receptor"
    PROP_FOR_RELATIONS_TEST = "is_used_for"
    # Expected related classes are many, just check a few
    EXPECTED_RELATED_CLASSES = ["noncovalent_interaction", "molecular_sensing", "catalysis"]
    CLASS_FOR_DEFINITION_TEST = "aryl-extended_calix(4)pyrrole"
    OBJ_PROP_FOR_DEFINITION_TEST = "is_complexed_by"
    DATA_PROP_FOR_DEFINITION_TEST = "content"
    ROOT_FOR_HIERARCHY_TEST = "macrocyclic_receptor"
    EXPECTED_TOP_LEVEL_CLASSES = ["calix(4)pyrrole", "macrocyclic_receptor", "oligopyrrolic_cage"] # A few examples


    def test_initialization(self, ontology_tools, real_ontology_settings):
        """Test if OntologyTools initializes correctly with the real ontology."""
        assert ontology_tools is not None
        assert ontology_tools.onto is not None
        assert ontology_tools.onto == real_ontology_settings.ontology
        # Basic namespace checks (more thorough checks in the fixture setup)
        assert ontology_tools.meta_ns is not None
        assert ontology_tools._classes_ns is not None
        assert ontology_tools._obj_props_ns is not None
        assert ontology_tools._data_props_ns is not None
        assert ontology_tools.has_information_prop is not None
        assert ontology_tools.SourcedInformationClass is not None
        # Verify the name and type of critical components
        assert ontology_tools.SourcedInformationClass.name == "SourcedInformation"
        assert isinstance(ontology_tools.has_information_prop, ObjectProperty)
        assert ontology_tools.has_information_prop.name == "has_information"

    def test_get_class_by_name_found(self, ontology_tools):
        """Test finding an existing class by name."""
        cls = ontology_tools._get_class_by_name(self.EXISTING_CLASS)
        assert cls is not None
        assert cls.name == self.EXISTING_CLASS
        # Check if it's actually a ThingClass
        assert isinstance(cls, ThingClass)

    def test_get_class_by_name_not_found(self, ontology_tools):
        """Test finding a non-existent class by name."""
        cls = ontology_tools._get_class_by_name(self.NON_EXISTENT_CLASS)
        assert cls is None

    def test_get_property_by_name_found(self, ontology_tools):
        """Test finding existing properties by name."""
        prop_data = ontology_tools._get_property_by_name(self.EXISTING_DATA_PROP)
        assert prop_data is not None
        assert isinstance(prop_data, DataProperty)
        assert prop_data.name == self.EXISTING_DATA_PROP

        prop_obj = ontology_tools._get_property_by_name(self.EXISTING_OBJ_PROP)
        assert prop_obj is not None
        assert isinstance(prop_obj, ObjectProperty)
        assert prop_obj.name == self.EXISTING_OBJ_PROP

        prop_meta_obj = ontology_tools._get_property_by_name(self.META_OBJ_PROP)
        assert prop_meta_obj is not None
        assert isinstance(prop_meta_obj, ObjectProperty)
        assert prop_meta_obj.name == self.META_OBJ_PROP

        prop_meta_data = ontology_tools._get_property_by_name(self.META_DATA_PROP)
        assert prop_meta_data is not None
        assert isinstance(prop_meta_data, DataProperty)
        assert prop_meta_data.name == self.META_DATA_PROP


    def test_get_property_by_name_not_found(self, ontology_tools):
        """Test finding a non-existent property."""
        prop = ontology_tools._get_property_by_name(self.NON_EXISTENT_PROP)
        assert prop is None

    # Note: test_get_class_info relies on meta:has_information linking to meta:SourcedInformation
    # instances which have meta:content. Need to verify this structure exists for EXISTING_CLASS
    # or choose a different class known to have this structure in backup-2.owl.
    # Assuming EXISTING_SUBCLASS ('aryl-extended_calix(4)pyrrole') has info based on example 10.
    def test_get_class_info(self, ontology_tools):
        """Test get_class_info for a single class with expected SourcedInformation."""
        target_class = self.EXISTING_SUBCLASS # Using this based on user example 10
        info = ontology_tools.get_class_info(target_class)
        assert target_class in info
        assert info[target_class]["name"] == target_class
        # Check if sourced information content is retrieved (assuming it exists)
        # This assert might fail if the class has no linked SourcedInformation with 'content'
        assert "information" in info[target_class]
        assert isinstance(info[target_class]["information"], list)
        # We don't know the exact content, just check if the list is potentially populated or empty
        # A more robust test would require knowing the expected content or count.
        print(f"INFO for {target_class}: {info[target_class]['information']}") # Add print for debugging


    def test_get_class_info_list(self, ontology_tools):
        """Test get_class_info for a list of classes."""
        info = ontology_tools.get_class_info([self.EXISTING_CLASS, self.NON_EXISTENT_CLASS])
        assert self.EXISTING_CLASS in info
        assert self.NON_EXISTENT_CLASS in info
        assert info[self.EXISTING_CLASS]["name"] == self.EXISTING_CLASS
        assert "error" in info[self.NON_EXISTENT_CLASS]
        assert "not found" in info[self.NON_EXISTENT_CLASS]["error"]

    # Similar note as test_get_class_info regarding existence of SourcedInformation
    def test_get_information_sources(self, ontology_tools):
        """Test getting information sources for a class (if any exist)."""
        target_class = self.EXISTING_SUBCLASS # Assuming this class has linked info
        sources = ontology_tools.get_information_sources(target_class)
        assert target_class in sources
        assert isinstance(sources[target_class], list)
        # We don't know the exact source, check if list is returned.
        print(f"SOURCES for {target_class}: {sources[target_class]}") # Add print


    # Need a known class, known source, and known content piece for this test.
    # Skipping detailed content check for now, add if details are available.
    def test_get_information_by_source(self, ontology_tools):
        """Test getting information by a specific source (if structure exists)."""
        target_class = self.EXISTING_SUBCLASS
        # We need a KNOWN source from backup-2.owl linked to target_class
        # Example: known_source = "Source Document XYZ"
        # For now, let's test with a dummy source and expect an empty list
        info_wrong_source = ontology_tools.get_information_by_source(target_class, "NonExistentSource")
        assert target_class in info_wrong_source
        assert info_wrong_source[target_class] == [] # Expect empty list for non-existent source


    def test_get_parents(self, ontology_tools):
        """Test getting parent classes."""
        parents = ontology_tools.get_parents(self.EXISTING_SUBCLASS)
        assert self.EXISTING_SUBCLASS in parents
        assert isinstance(parents[self.EXISTING_SUBCLASS], list)
        assert self.EXISTING_PARENT in parents[self.EXISTING_SUBCLASS]

        parents_top = ontology_tools.get_parents(self.EXISTING_PARENT) # Should be top-level
        assert self.EXISTING_PARENT in parents_top
        assert parents_top[self.EXISTING_PARENT] == [] # Expect no named parents

    def test_get_children(self, ontology_tools):
        """Test getting child classes."""
        children = ontology_tools.get_children(self.CLASS_WITH_CHILDREN)
        assert self.CLASS_WITH_CHILDREN in children
        assert isinstance(children[self.CLASS_WITH_CHILDREN], list)
        assert self.EXPECTED_CHILD in children[self.CLASS_WITH_CHILDREN]

        children_leaf = ontology_tools.get_children(self.EXISTING_SUBCLASS) # This one might have children too? Check ontology. Assuming leaf for now.
        assert self.EXISTING_SUBCLASS in children_leaf
        # If it's truly a leaf, expect [], otherwise adjust based on backup-2.owl
        assert isinstance(children_leaf[self.EXISTING_SUBCLASS], list)


    def test_get_ancestors(self, ontology_tools):
        """Test getting ancestors."""
        ancestors = ontology_tools.get_ancestors(self.EXISTING_SUBCLASS)
        assert self.EXISTING_SUBCLASS in ancestors
        assert isinstance(ancestors[self.EXISTING_SUBCLASS], list)
        assert self.EXISTING_PARENT in ancestors[self.EXISTING_SUBCLASS] # Direct parent
        # Might have more ancestors, check if needed

    def test_get_descendants(self, ontology_tools):
        """Test getting descendants."""
        descendants = ontology_tools.get_descendants(self.CLASS_WITH_CHILDREN)
        assert self.CLASS_WITH_CHILDREN in descendants
        assert isinstance(descendants[self.CLASS_WITH_CHILDREN], list)
        assert self.EXPECTED_CHILD in descendants[self.CLASS_WITH_CHILDREN]
        # Might have more descendants

    def test_get_class_properties(self, ontology_tools):
        """Test getting properties associated with a class (incl. from restrictions)."""
        props = ontology_tools.get_class_properties(self.CLASS_FOR_PROPS_TEST)
        assert self.CLASS_FOR_PROPS_TEST in props
        props_list = props[self.CLASS_FOR_PROPS_TEST]
        assert isinstance(props_list, list)
        # Check for expected properties based on user example 6
        assert self.META_OBJ_PROP in props_list
        assert "hydrophobic_effect" in props_list
        assert "reaction_selectivity" in props_list
        assert self.PROP_FOR_RESTRICTION_TEST in props_list
        # Verify sorting if the tool guarantees it, otherwise check presence


    def test_get_property_restrictions(self, ontology_tools):
        """Test getting restrictions for a property on a class."""
        restrictions = ontology_tools.get_property_restrictions(
            self.CLASS_FOR_RESTRICTION_TEST, self.PROP_FOR_RESTRICTION_TEST
        )
        assert isinstance(restrictions, list)
        assert len(restrictions) >= 1 # Expect at least one restriction

        # Find the specific restriction mentioned in user example 7
        found_expected_restriction = False
        for r in restrictions:
            assert isinstance(r, dict)
            assert "type" in r
            assert "value" in r
            if r["type"] == "SOME" and r["value"] == self.EXPECTED_RESTRICTION_VALUE_CLASS:
                found_expected_restriction = True
                break
        assert found_expected_restriction, f"Expected SOME {self.EXPECTED_RESTRICTION_VALUE_CLASS} restriction, found: {restrictions}"

    def test_get_related_classes(self, ontology_tools):
         """Test getting classes related via object properties based on restrictions."""
         related = ontology_tools.get_related_classes(self.CLASS_FOR_RELATIONS_TEST)
         assert self.CLASS_FOR_RELATIONS_TEST in related
         related_map = related[self.CLASS_FOR_RELATIONS_TEST]
         assert isinstance(related_map, dict)
         # Check the specific property from user example 8
         assert self.PROP_FOR_RELATIONS_TEST in related_map
         related_list = related_map[self.PROP_FOR_RELATIONS_TEST]
         assert isinstance(related_list, list)
         # Check if some of the expected related classes are present
         for expected_rel_class in self.EXPECTED_RELATED_CLASSES:
             assert expected_rel_class in related_list


    def test_get_disjoint_classes(self, ontology_tools):
         """Test getting disjoint classes (expecting none based on user info)."""
         # Test with a couple of classes
         for class_name in [self.EXISTING_CLASS, self.EXISTING_SUBCLASS, self.CLASS_WITH_CHILDREN]:
             disjoint = ontology_tools.get_disjoint_classes(class_name)
             assert class_name in disjoint
             # Expecting empty list as per user info for backup-2.owl
             assert disjoint[class_name] == []


    def test_parse_class_definition(self, ontology_tools):
        """Test parsing the full definition of a class."""
        definition_map = ontology_tools.parse_class_definition(self.CLASS_FOR_DEFINITION_TEST)
        assert self.CLASS_FOR_DEFINITION_TEST in definition_map
        parsed = definition_map[self.CLASS_FOR_DEFINITION_TEST]

        # Check basic info
        assert "basic_info" in parsed
        assert parsed["basic_info"]["name"] == self.CLASS_FOR_DEFINITION_TEST
        # Sourced info check (assuming it exists, might be empty list)
        assert "information" in parsed["basic_info"]
        assert isinstance(parsed["basic_info"]["information"], list)

        # Check hierarchy (based on user example 4)
        assert "hierarchy" in parsed
        assert isinstance(parsed["hierarchy"], dict)
        assert "parents" in parsed["hierarchy"]
        assert isinstance(parsed["hierarchy"]["parents"], list)
        assert self.EXISTING_PARENT in parsed["hierarchy"]["parents"] # Parent is calix(4)pyrrole

        # Check properties - just verify the structure
        assert "properties" in parsed
        assert isinstance(parsed["properties"], dict)
        assert "data" in parsed["properties"]
        assert "object" in parsed["properties"]
        assert isinstance(parsed["properties"]["data"], list)
        assert isinstance(parsed["properties"]["object"], list)
        # Could add checks for specific properties if needed

        # Check relations - just verify the structure
        assert "relations" in parsed
        assert isinstance(parsed["relations"], dict)

        # Check disjoints (expecting empty based on user info)
        assert "disjoint_with" in parsed
        assert parsed["disjoint_with"] == []

    def test_parse_property_definition_data(self, ontology_tools):
        """Test parsing a data property definition."""
        definition_map = ontology_tools.parse_property_definition(self.DATA_PROP_FOR_DEFINITION_TEST)
        assert self.DATA_PROP_FOR_DEFINITION_TEST in definition_map
        parsed = definition_map[self.DATA_PROP_FOR_DEFINITION_TEST]

        assert parsed.get("name") == self.DATA_PROP_FOR_DEFINITION_TEST
        assert parsed.get("type") == "data" # Expecting 'data'
        assert "domain" in parsed # Check presence
        assert "range" in parsed  # Check presence
        # Check specific range based on user example 11
        assert "http://www.w3.org/2001/XMLSchema#string" in parsed.get("range", [])
        # Check specific domain based on user example 11
        assert "SourcedInformation" in parsed.get("domain", [])
        assert "characteristics" in parsed # Check presence

    def test_parse_property_definition_object(self, ontology_tools):
        """Test parsing an object property definition."""
        definition_map = ontology_tools.parse_property_definition(self.OBJ_PROP_FOR_DEFINITION_TEST)
        assert self.OBJ_PROP_FOR_DEFINITION_TEST in definition_map
        parsed = definition_map[self.OBJ_PROP_FOR_DEFINITION_TEST]

        assert parsed.get("name") == self.OBJ_PROP_FOR_DEFINITION_TEST
        assert parsed.get("type") == "object" # Expecting 'object'
        assert "domain" in parsed # Check presence
        assert "range" in parsed  # Check presence
        # Domain/range might be inferred, check if list exists
        assert isinstance(parsed.get("domain"), list)
        assert isinstance(parsed.get("range"), list)
        assert "characteristics" in parsed # Check presence
        assert "usage_in_restrictions" in parsed # Check presence
        assert isinstance(parsed.get("usage_in_restrictions"), list)


    def test_parse_hierarchy_structure_root(self, ontology_tools):
         """Test parsing hierarchy from a specific root."""
         hierarchy = ontology_tools.parse_hierarchy_structure(self.ROOT_FOR_HIERARCHY_TEST)
         assert isinstance(hierarchy, dict)
         assert hierarchy.get("name") == self.ROOT_FOR_HIERARCHY_TEST
         assert "children" in hierarchy
         assert isinstance(hierarchy["children"], list)
         # Check if at least one expected child is present
         child_names = [child.get("name") for child in hierarchy["children"]]
         assert self.EXPECTED_CHILD in child_names


    def test_parse_hierarchy_structure_full(self, ontology_tools):
         """Test parsing the full hierarchy (forest)."""
         forest = ontology_tools.parse_hierarchy_structure() # No root specified
         assert isinstance(forest, list)
         assert len(forest) > 0 # Expecting multiple top-level classes

         # Get names of the root nodes in the forest
         top_level_names = [node.get("name") for node in forest if isinstance(node, dict)]

         # Check if some of the expected top-level classes are present
         for expected_top_class in self.EXPECTED_TOP_LEVEL_CLASSES:
             assert expected_top_class in top_level_names

         # Find a specific node (e.g., the root used above) and verify structure briefly
         root_node = next((node for node in forest if isinstance(node, dict) and node.get("name") == self.ROOT_FOR_HIERARCHY_TEST), None)
         assert root_node is not None
         assert "children" in root_node
         child_names = [child.get("name") for child in root_node["children"]]
         assert self.EXPECTED_CHILD in child_names


# Removed commented-out example run block


