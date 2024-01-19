from rdflib import URIRef

RR = "http://www.w3.org/ns/r2rml#"

RML =  "http://semweb.mmlab.be/ns/rml#"

QL = "http://semweb.mmlab.be/ns/ql#"

CRML = "http://w3id.org/stlab/crml#"

FNML= "http://semweb.mmlab.be/ns/fnml#"

FNO = "https://w3id.org/function/ontology#"

CSVW= "http://www.w3.org/ns/csvw#"

SD = "http://www.w3.org/ns/sparql-service-description#"

CONDITION = URIRef(CRML + "condition")

HAS_JOINED_SOURCE = URIRef(CRML + "hasJoinedSource")

ON_SOURCE = URIRef(CRML + "onSource")

WITH_JOIN = URIRef(CRML + "withJoin")

JOIN = URIRef(RR + "Join")

JOIN_CONDITION = URIRef(RR + "joinCondition")

CHILD = URIRef(RR + "child")

PARENT = URIRef(RR + "parent")

CONSTANT = URIRef(RR + "constant")

REFERENCE = URIRef(RML + "reference")

LANGUAGE = URIRef(RR + "language")

LANGUAGE_MAP = URIRef(RML + "languageMap")

DATATYPE = URIRef(RR + "datatype")

PARENT_TRIPLES_MAP = URIRef(RR + "parentTriplesMap")

FUNCTION_VALUE = URIRef(FNML + "functionValue")

TEMPLATE = URIRef(RR + "template")

TERM_TYPE = URIRef(RR + "termType")

CLASS = URIRef(RR + "class")

GRAPH_MAP = URIRef(RR + "graphMap")

GRAPH = URIRef(RR + "graph")

CSV = URIRef(QL + "CSV")

JSON_PATH = URIRef(QL + "JSONPath")

XML = URIRef(QL + "XML")

REFERENCE_FORMULATION = URIRef(RML + "referenceFormulation")

ITERATOR = URIRef(RML + "iterator")

SOURCE = URIRef(RML + "source")

BASE_SOURCE = URIRef(RML + "BaseSource")

TRIPLES_MAP = URIRef(RR + "TriplesMap")

LOGICAL_SOURCE = URIRef(RML + "logicalSource")

SUBJECT_MAP = URIRef(RR + "subjectMap")

PREDICATE_OBJECT_MAP = URIRef(RR + "predicateObjectMap")

PREDICATE_OBJECT_MAP_CLASS = URIRef(RR + "PredicateObjectMap")

PREDICATE = URIRef(RR + "predicate")

PREDICATE_MAP = URIRef(RR + "predicateMap")

OBJECT_MAP = URIRef(RR + "objectMap")

OBJECT_MAP_CLASS = URIRef(RR + "ObjectMap")

SEPARATOR = URIRef(CRML + "separator")

IRI = URIRef(RR + "IRI")

BLANK_NODE = URIRef(RR + "BlankNode")

LITERAL = URIRef(RR + "Literal")

EXECUTES = URIRef(FNO + "executes")

SERVICE = URIRef(SD + "Service")