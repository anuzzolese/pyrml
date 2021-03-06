__author__ = "Andrea Giovanni Nuzzolese"
__email__ = "andrea.nuzzolese@cnr.it"
__license__ = "Apache 2"
__version__ = "0.1"
__status__ = "Pre-Alpha"

from abc import ABC, abstractclassmethod
from argparse import ArgumentParser
from builtins import staticmethod
import codecs
import os
from pathlib import Path
from typing import Dict, Union, Set

from lark import Lark
from lark.visitors import Transformer
from pandas.core.frame import DataFrame
from rdflib import URIRef, Graph
from rdflib.namespace import RDF
from rdflib.plugins.sparql.processor import prepareQuery
from rdflib.term import Node, BNode, Literal, Identifier
import unidecode, re

import logging as log
import pandas as pd
import rml_vocab as rml_vocab


class TermMap(ABC):
    
    def __init__(self, map_id: URIRef = None):
        if map_id is None:
            self._id = BNode()
        else:
            self._id = map_id
         
            
    def get_id(self) -> Union[URIRef, BNode]:
        return self._id
    
    @abstractclassmethod
    def get_mapped_entity(self) -> Node:
        pass
    
    @abstractclassmethod
    def to_rdf(self) -> Graph:
        pass
    
    @staticmethod
    @abstractclassmethod
    def from_rdf(g: Graph) -> Set[object]:
        pass
    
class AbstractMap(TermMap):
    def __init__(self, map_id: URIRef = None, mapped_entity: Node = None):
        super().__init__(map_id)
        self._mapped_entity = mapped_entity
        
    def get_mapped_entity(self) -> Node:
        return self._mapped_entity
    
    @abstractclassmethod
    def to_rdf(self):
        pass
    
    @staticmethod
    @abstractclassmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        pass
    



class ObjectMap(AbstractMap):
    
    def to_rdf(self):
        g = Graph()
        g.add((self._id, RDF.type, rml_vocab.OBJECT_MAP_CLASS))
        return g

    def apply(self, df):
        pass
    
    @staticmethod
    @abstractclassmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        pass
    
class ConstantObjectMap(ObjectMap):
    def __init__(self, value: Node, map_id: URIRef = None):
        super().__init__(map_id, value)
        self.__value = value
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        g.add((self._id, rml_vocab.CONSTANT, self.__value))
        return g
    
    def apply(self, df: DataFrame):
        
        return df.apply(lambda x: self.__value, axis=1)
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?c
                WHERE {
                    {
                        ?p rr:constant ?c1
                        BIND(?c1 AS ?c)
                    }
                    UNION
                    {
                        OPTIONAL{?p rr:constant ?c2}
                        FILTER(!BOUND(?c2))
                        FILTER(isIRI(?p))
                        BIND(?p AS ?c)
                    }
            }""", 
            initNs = { "rr": rml_vocab.RR})

        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            
            c = None
            if isinstance(row.p, URIRef):
                if row.p in mappings_dict:
                    c = mappings_dict.get(row.p)
                else:
                    c = ConstantObjectMap(row.c, row.p)
                    mappings_dict.add(c)
            else:
                c = ConstantObjectMap(row.c)
                
            term_maps.add(c)
           
        return term_maps
            
                
    
class LiteralObjectMap(ObjectMap):
    def __init__(self, reference: Literal = None, template: Literal = None, term_type : URIRef = None, language : Literal = None, datatype : URIRef = None, map_id: URIRef = None):
        super().__init__(map_id, reference if reference is not None else template)
        self._reference = reference
        self._template = template
        self._term_type = term_type
        self._language = language
        self._datatype = datatype 
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        if self._reference is not None:
            g.add((self._id, rml_vocab.REFERENCE, self._reference))
        elif self._template is not None:
            g.add((self._id, rml_vocab.TEMPLATE, self._template))
            if self._term_type is not None:
                g.add((self._id, rml_vocab.TERM_TYPE, self._term_type))
            
        if self._language is not None:
            g.add((self._id, rml_vocab.LANGUAGE, self._language))
        elif self._datatype is not None:
            g.add((self._id, rml_vocab.DATATYPE, self._datatype))
            
        return g
    
    
    def __convertion(self, row):
        
        literal = None
        
        if self._reference is not None:
            value = row[self._reference.value]
        if self._template is not None:
            value = TermUtils.eval_functions(self._template.value, row)
                
        
        if value != value:
            literal = None
        elif self._language is not None:
            language = TermUtils.eval_functions(self._language.value, row)
            literal = Literal(value, lang=language)
        elif self._datatype is not None:
            datatype = TermUtils.eval_functions(str(self._datatype), row)
            literal = Literal(value, datatype=datatype)
        else:
            literal = Literal(value)
        
        return literal
    
    def apply(self, df: DataFrame):
        
        l = lambda x: self.__convertion(x)
                
        df_1 = df.apply(l, axis=1)
        
        return df_1
    
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?reference ?template ?tt ?language ?datatype
                WHERE {
                    OPTIONAL{?p rml:reference ?reference} 
                    OPTIONAL{?p rr:template ?template}
                    OPTIONAL{?p rr:termType ?tt}
                    OPTIONAL {?p rr:language ?language}
                    OPTIONAL {?p rr:datatype ?datatype}
            }""", 
            initNs = { 
                "rr": rml_vocab.RR,
                "rml": rml_vocab.RML
                })

        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            term_maps.add(LiteralObjectMap(row.reference, row.template, row.tt, row.language, row.datatype, row.p))
           
        return term_maps     
        

class PredicateObjectMap(AbstractMap):
    def __init__(self, predicate: URIRef, object_map: ObjectMap, map_id: URIRef = None):
        super().__init__(map_id, predicate)
        self.__object_map = object_map
        
    def get_predicate(self) -> URIRef:
        return self.get_mapped_entity()
    
    def get_object_map(self) -> ObjectMap:
        return self.__object_map
    
    def to_rdf(self) -> Graph:
        g = Graph()
        g.add((self._id, RDF.type, rml_vocab.PREDICATE_OBJECT_MAP_CLASS))
        g.add((self._id, rml_vocab.PREDICATE, self.get_mapped_entity()))
        g.add((self._id, rml_vocab.OBJECT_MAP, self.__object_map.get_id()))
        
        g += self.__object_map.to_rdf()
        
        return g
    
    def apply(self, df: DataFrame):
        
        df_1 = self.__object_map.apply(df)
        df_1 = df_1.apply(lambda x: (self.get_mapped_entity(), x))
        
        return df_1
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?pom ?predicate ?om
                WHERE {
                    ?pom rr:predicate ?predicate ;
                        rr:objectMap ?om
            }""", 
            initNs = { "rr": rml_vocab.RR})

        if parent is not None:
            qres = g.query(query, initBindings = { "pom": parent})
        else:
            qres = g.query(query)
        
        mapping_dict = RMLConverter.get_instance().get_mapping_dict()
        for row in qres:
            
            pom = None
            if isinstance(row.pom, URIRef):
                if row.pom in mapping_dict:
                    pom = mapping_dict.get(row.pom)
                else:
                    pom = PredicateObjectMap.__build(g, row)
                    mapping_dict.add(pom)
            else:
                pom = PredicateObjectMap.__build(g, row)
                    
                     
            term_maps.add(pom)
           
        return term_maps
    
    @staticmethod
    def __build(g, row):
        mapping_dict = RMLConverter.get_instance().get_mapping_dict()
        
        object_map = None
        if isinstance(row.om, URIRef): 
            if row.om in mapping_dict:
                object_map = mapping_dict.get(row.om)
            else:
                object_map = ObjectMapBuilder.build(g, row.om)
                mapping_dict.add(object_map)
        else: 
            object_map = ObjectMapBuilder.build(g, row.om)

        if object_map is not None:
            return PredicateObjectMap(row.predicate, object_map, row.pom)
        else: 
            return None;
    
class ObjectMapBuilder():
    
    @staticmethod
    def build(g: Graph, parent: Union[URIRef, BNode]) -> Set[ObjectMap]:
        
        ret = None
        if (parent, rml_vocab.CONSTANT, None) in g:
            ret = ConstantObjectMap.from_rdf(g, parent)
        elif (parent, rml_vocab.REFERENCE, None) in g or (parent, rml_vocab.TEMPLATE, None) in g:
            ret = LiteralObjectMap.from_rdf(g, parent)
        elif (parent, rml_vocab.PARENT_TRIPLES_MAP, None) in g:
            ret = ReferencingObjectMap.from_rdf(g, parent)
        elif isinstance(parent, URIRef):
            ret = ConstantObjectMap.from_rdf(g, parent)
        else:
            return None
        
        if ret is None or len(ret) == 0:
            return None
        else:
            return ret.pop()


class Join(AbstractMap):
    def __init__(self, child: Literal, parent: Literal, map_id: URIRef = None):
        super().__init__(map_id)
        self.__child = child
        self.__parent = parent
        
    def get_child(self) -> str:
        return self.__child
    
    def get_parent(self) -> str:
        return self.__parent
    
    def to_rdf(self) -> Graph:
        g = Graph()
        
        if self.__child is not None and self.__parent is not None:
            join = self._id
            g.add((join, rml_vocab.CHILD, self.__child))
            g.add((join, rml_vocab.PARENT, self.__parent))
            
        return g
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?join ?child ?parent
                WHERE {
                    ?p rr:joinCondition ?join . 
                    ?join rr:child ?child ;
                        rr:parent ?parent
            }""", 
            initNs = { "rr": rml_vocab.RR})

        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            join = None
            if isinstance(row.join, URIRef):
                join = Join(row.child, row.parent, row.join)
            else:
                join = Join(child=row.child, parent=row.parent)
            
            term_maps.add(join)
           
        return term_maps
    

class LogicalSource(AbstractMap):
    def __init__(self, source: Literal, separator: str = None, map_id: URIRef = None):
        super().__init__(map_id, source)
        self.__separator = separator
        
    def get_source(self) -> Literal:
        return self.get_mapped_entity()
    
    def get_reference_formulation(self) -> URIRef:
        return rml_vocab.CSV
    
    def get_separator(self) -> str:
        return self.__separator
    
    def to_rdf(self):
        g = Graph()
        g.add((self._id, RDF.type, rml_vocab.BASE_SOURCE))
        g.add((self._id, rml_vocab.SOURCE, self.get_source()))
        g.add((self._id, rml_vocab.REFERENCE_FORMULATION, rml_vocab.CSV))
        if self.__separator is not None:
            g.add((self._id, rml_vocab.SEPARATOR, self.__separator))
        
        return g
    
    def apply(self):
        if self.__separator is None:
            sep = ','
        else:
            sep = self.__separator
        
        df = pd.read_csv(self._mapped_entity.value, sep=sep, dtype=str)
        
        return df 
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
            
        sparql = """
            SELECT DISTINCT ?ls ?source ?rf ?sep
            WHERE {
                ?p rml:logicalSource ?ls .
                ?ls rml:source ?source .
                OPTIONAL {?ls rml:referenceFormulation ?rf}
                OPTIONAL {?ls crml:separator ?sep}
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { 
                    "rml": rml_vocab.RML,
                    "crml": rml_vocab.CRML 
                })
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
                    
        for row in qres:
            
            source = row.source
            
            ls = LogicalSource(source, row.sep, row.ls)
            term_maps.add(ls)
            
        
           
        return term_maps
    

class GraphMap(AbstractMap):
    
    def __init__(self, mapped_entity: Node, map_id: URIRef = None):
        super().__init__(map_id, mapped_entity)
        
    
    def to_rdf(self):
        g = Graph()
        
        if isinstance(self._mapped_entity, Literal):
            g.add((self._id, rml_vocab.TEMPLATE, self._mapped_entity))
        elif isinstance(self._mapped_entity, URIRef):
            g.add((self._id, rml_vocab.CONSTANT, self._mapped_entity))
            
        return g
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
            
        sparql = """
            SELECT DISTINCT ?gm ?g
            WHERE {
                { ?p rr:graphMap ?gm . 
                  ?gm rr:constant ?g }
                UNION
                { ?p rr:graph ?g }
                UNION
                { ?p rr:graphMap ?gm . 
                  ?gm rr:template ?g }                
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { "rr": rml_vocab.RR })
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
                    
        for row in qres:
            graph_map = None
            if row.gm is not None:
                if isinstance(row.gm, URIRef):
                    if row.gm in mappings_dict:
                        graph_map = mappings_dict.get(row.gm)
                    else:
                        graph_map = GraphMap(row.g, row.gm)
                        mappings_dict.add(graph_map)
                else:
                    graph_map = GraphMap(row.g, row.gm)
            elif row.g is not None:
                graph_map = GraphMap(mapped_entity=row.g)
                
            term_maps.add(graph_map)
           
        return term_maps
    
class SubjectMap(AbstractMap):
    def __init__(self, mapped_entity: Node, class_: URIRef = None, graph_map: GraphMap = None, map_id: URIRef = None):
        super().__init__(map_id, mapped_entity)
        self.__class = class_
        self.__graph_map = graph_map
    
    def get_class(self) -> URIRef:
        return self.__class
    
    def get_graph_map(self) -> GraphMap:
        return self.__graph_map
                
    def to_rdf(self):
        g = Graph()
        subject_map = self._id
        
        if isinstance(self._mapped_entity, Literal):
            g.add((subject_map, rml_vocab.TEMPLATE, self._mapped_entity))
        elif isinstance(self._mapped_entity, URIRef):
            g.add((subject_map, rml_vocab.CONSTANT, self._mapped_entity))
            
        if self.__class is not None:
            g.add((subject_map, rml_vocab.CLASS, self.__class))
            
        if self.__graph_map is not None:
            graph_map_g = self.__graph_map.to_rdf()
             
            g.add((subject_map, rml_vocab.GRAPH_MAP, self.__graph_map.get_id()))
            g = g + graph_map_g
            
        
        return g
    
    def apply(self, df: DataFrame):
        
        l = lambda x: TermUtils.urify(self._mapped_entity, x)
                
        df_1 = df.apply(l, axis=1)
        
        return df_1
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
            
        sparql = """
            SELECT DISTINCT ?sm ?map ?type ?gm ?g
            WHERE {
                ?p rr:subjectMap ?sm .
                { ?sm rr:template ?map } 
                UNION
                { ?sm rr:constant ?map }
                OPTIONAL {?sm rr:class ?type}
                OPTIONAL {
                    { ?sm rr:graphMap ?gm }
                    UNION
                    { ?sm rr:graph ?g }
                }
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { "rr": rml_vocab.RR })
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
                    
        for row in qres:
            subject_map = None
            if isinstance(row.sm, URIRef):
                if row.sm in mappings_dict:
                    subject_map = mappings_dict.get(row.sm)
                else:
                    subject_map = SubjectMap.__create(row)
                    mappings_dict.add(subject_map)
            else: 
                subject_map = SubjectMap.__create(row)
                
            term_maps.add(subject_map)
           
        return term_maps
    
    @staticmethod
    def __create(row):
        
        graph_map = None
        if row.gm is not None or row.g is not None:
            graph_map = GraphMap.from_rdf(row.g, row.sm).pop()
                
        return SubjectMap(row.map, row.type, graph_map, row.sm)
    
    
class TripleMappings(AbstractMap):
    def __init__(self,
                 logical_source: LogicalSource, 
                 subject_map: SubjectMap,
                 predicate_object_maps: Dict[Identifier, ObjectMap] = None, 
                 iri: URIRef = None,
                 condition: str = None):
        super().__init__(iri, logical_source.get_id())
        self.__logical_source = logical_source
        self.__subject_map = subject_map
        self.__predicate_object_maps = predicate_object_maps
        self.__condition = condition
        
    def get_logical_source(self) -> LogicalSource:
        return self.__logical_source
        
    def get_subject_map(self) -> SubjectMap:
        return self.__subject_map
    
    def get_predicate_object_maps(self) -> Dict[Identifier, ObjectMap]:
        return self.__predicate_object_maps
    
    def set_predicate_object_maps(self, poms: Dict[Identifier, ObjectMap]):
        self.__predicate_object_maps = poms
    
    def get_condition(self):
        return self.__condition
         
    def add_object_map(self, object_map: ObjectMap):
        if self.__predicate_object_maps is None:
            self.__predicate_object_maps = dict()
        
        self.__predicate_object_maps.update({object_map.get_id(), object_map})
        
    def get_object_map(self, identifier: Union[URIRef, BNode]) -> ObjectMap:
        return self.__predicate_object_maps.get(identifier)
    
    def to_rdf(self) -> Graph:
        g = Graph()
        g.add((self._id, RDF.type, rml_vocab.TRIPLES_MAP))
        g.add((self._id, rml_vocab.LOGICAL_SOURCE, self.__logical_source.get_id()))
        g.add((self._id, rml_vocab.SUBJECT_MAP, self.__subject_map.get_id()))
        
        if self.__condition is not None:
            g.add((self._id, rml_vocab.CONDITION, self.__condition))
        
        g += self.__logical_source.to_rdf()
        g += self.__subject_map.to_rdf()
        
        for key, value in self.__predicate_object_maps.items():
            g.add((self._id, rml_vocab.PREDICATE_OBJECT_MAP, key))
            g += value.to_rdf()
            
        return g
    
    def apply(self):
        g = Graph()
        
        df = self.__logical_source.apply()
        
        if self.__condition is not None and self.__condition.strip() != '':
            df = df[eval(self.__condition)]
        
        sbj_representation = self.__subject_map.apply(df)
        if self.__predicate_object_maps is not None:
            for pom in self.__predicate_object_maps.values():
                pom_representation = pom.apply(df) 
            
            
                results = pd.concat([sbj_representation, pom_representation], axis=1, sort=False)
                for k,v in results.iterrows():
                    try:
                        g.add((v[0], v[1][0], v[1][1]))
                        
                        if self.__subject_map.get_class() is not None:
                            g.add((v[0], RDF.type, self.__subject_map.get_class()))
                        
                    except:
                        pass
        elif self.__subject_map.get_class() is not None:
            
            for k,v in sbj_representation.iteritems():
                try:
                    g.add((v, RDF.type, self.__subject_map.get_class()))
                    
                except:
                    pass
            
        return g
        
        
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        
            
        sparql = """
            SELECT DISTINCT ?tm ?source ?sm ?pom ?cond
            WHERE {
                %PLACE_HOLDER%
                ?tm rml:logicalSource ?source ;
                    rr:subjectMap ?sm 
                OPTIONAL {?tm rr:predicateObjectMap ?pom}
                OPTIONAL {?tm crml:condition ?cond}
            }"""
        
        if parent is not None:
            sparql = sparql.replace("%PLACE_HOLDER%", "?p rr:parentTriplesMap ?tm . ")
        else:
            sparql = sparql.replace("%PLACE_HOLDER%", "")
            
        query = prepareQuery(sparql, 
                initNs = { 
                    "rr": rml_vocab.RR, 
                    "rml": rml_vocab.RML,
                    "crml": rml_vocab.CRML})
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
            
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
        
        for row in qres:
            tm = None
            if isinstance(row.tm, URIRef):
                if row.tm in mappings_dict:
                    tm = mappings_dict.get(row.tm)
                    
                    if tm is not None:
                        pom = TripleMappings.__build_predicate_object_map(g, row)
                        poms = tm.get_predicate_object_maps()
                        if pom is not None and poms is None:
                            tm.set_predicate_object_maps({ pom.get_id(): pom })
                        elif pom is not None and pom.get_id() not in poms:
                            poms.update({ pom.get_id(): pom })
                                            
                else:
                    tm = TripleMappings.__build(g, row)
                    mappings_dict.add(tm)
            else:
                tm = TripleMappings.__build(g, row)
                mappings_dict.add(tm)
            
            if tm is not None:
                term_maps.add(tm)
            
            
           
        return term_maps
    
    @staticmethod
    def __build(g, row):
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
        
        source = None
        if row.source is not None:
            if isinstance(row.source, URIRef) and row.source in mappings_dict:
                source = mappings_dict.get(row.source)
            else:
                source = LogicalSource.from_rdf(g, row.tm).pop()
                mappings_dict.add(source)
                
        subject_map = None
        if row.sm is not None:
            if isinstance(row.sm, URIRef): 
                if row.sm in mappings_dict:
                    subject_map = mappings_dict.get(row.sm)
                else:
                    subject_map = SubjectMap.from_rdf(g, row.tm).pop()
                    mappings_dict.add(subject_map)
            else:
                subject_map = SubjectMap.from_rdf(g, row.tm).pop()
        
        predicate_object_map = TripleMappings.__build_predicate_object_map(g, row)
        
        if predicate_object_map is not None:
            pom_dict = { predicate_object_map.get_id(): predicate_object_map }
        else:
            pom_dict = None
        return TripleMappings(source, subject_map, pom_dict, row.tm, row.cond)
    
    @staticmethod
    def __build_predicate_object_map(g, row):
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
        
        predicate_object_map = None
        if row.pom is not None:
            if isinstance(row.pom, URIRef):
                if row.pom in mappings_dict:
                    predicate_object_map = mappings_dict.get(row.pom)
                else:
                    predicate_object_map = PredicateObjectMap.from_rdf(g, row.pom).pop()
                    mappings_dict.add(predicate_object_map)
            else:
                pom = PredicateObjectMap.from_rdf(g, row.pom)
                if len(pom) > 0:
                    predicate_object_map = pom.pop()
        return predicate_object_map
        
            
    
class ReferencingObjectMap(ObjectMap):
    def __init__(self, parent_triples_map: TripleMappings, join: Join = None, map_id: URIRef = None):
        super().__init__(map_id, parent_triples_map.get_id())
        self.__parent_triples_map = parent_triples_map
        self.__join = join
        
    def get_parent_triples_map(self) -> TripleMappings:
        return self.__parent_triples_map
    
    def get_join_condition(self) -> Join:
        return self.__join
    
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        if self.__child is not None and self.__parent is not None:
            g.add((self._id, rml_vocab.PARENT_TRIPLES_MAP, self.__parent_triples_map.get_id()))
            
        return g
    
    def apply(self, df: DataFrame):
        

        l = lambda x: TermUtils.urify(self.__parent_triples_map.get_subject_map().get_mapped_entity(), x)
                
        if self.__join is not None:
            
            left_on = self.__join.get_child()
            right_on = self.__join.get_parent()
            ptm = RMLConverter.get_instance().get_mapping_dict().get(self.__parent_triples_map.get_id())
            right = ptm.get_logical_source().apply()
            df_1 = df.join(right.set_index(right_on.value), how='inner', lsuffix="_l", rsuffix="_r", on=left_on.value, sort=False).rename(columns={left_on.value: right_on.value})
            
        else:
            df_1 = df
        
        df_1 = df_1.apply(l, axis=1)

        df_1.dropna(inplace=True)
        
            
        return df_1
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?parentTriples ?join
                WHERE {
                    ?p rr:parentTriplesMap ?parentTriples .
                    OPTIONAL {?p rr:joinCondition ?join}
            }""", 
            initNs = { "rr": rml_vocab.RR})
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            
            join = None
            if row.join is not None:
                join = Join.from_rdf(g, parent).pop()
            
            parent_triples = None
            if isinstance(row.parentTriples, URIRef):
                if row.parentTriples in mappings_dict:
                    parent_triples = mappings_dict.get(row.parentTriples)
                else:
                    mappings = TripleMappings.from_rdf(g, row.p)
                    if len(mappings) > 0:
                        parent_triples = mappings.pop()
                        mappings_dict.add(parent_triples)
            else:
                parent_triples = TripleMappings.from_rdf(g, row.p).pop()
            
            if parent_triples is not None:
                rmo = ReferencingObjectMap(parent_triples, join, row.p)
                term_maps.add(rmo)
           
        return term_maps
         
    
class MappingsDict():
    
    def __init__(self):
        self.__dict = dict()
        MappingsDict.__instance = self
    
    def __iter__(self):
        return self.__dict.__iter__()
    
    def __next__(self):
        return self.__dict.__next__()
    
    def add(self, term_map : TermMap):
        if isinstance(term_map.get_id(), URIRef):
            self.__dict.update( {term_map.get_id(): term_map} )
            
    def get(self, iri : URIRef):
        return self.__dict[iri]
                
                
class TermUtils():
    
    @staticmethod
    def generate_id(string : str) -> str:
        string = unidecode.unidecode(string).lower()

        string = re.sub("[';.,&\"???!/\\\\\(\)]", "", string)
        string = re.sub("[ <>]", "-", string)
        string = re.sub("(\-)+", "-", string)
        return string
    
    @staticmethod
    def urify(entity, row):
        if isinstance(entity, Literal):
            s = TermUtils.eval_functions(entity, row)
            if s is not None:
                return URIRef(s)
            else:
                return float('nan')
        elif isinstance(entity, URIRef):
            return entity
        
    @staticmethod
    def replace_place_holders(value, row):
        #p = re.compile('\{(.+)\/?\}')
        p = re.compile('(?<=\{).+?(?=\})')
        
        
        matches = p.finditer(value)
        
        s = value
        
        for match in matches:
            column = match.group(0)
            if row[column] != row[column]:
                return None
            else:
                text = "{" + column + "}"
                if column not in row.index:
                    column += "_l"
                s = re.sub(text, str(row[column]), s)
            
        return s
    
    @staticmethod
    def __eval_functions(text, row):
        
        expr = EvalParser.parse(text, row)
        return expr
        
        
        
    @staticmethod    
    def __eval_functions_old(text, row):
        
        start = text.find("(")
        end = text.rfind(")")
        name = text[0:start].strip()
        
        if start > 0:
            body = TermUtils.__eval_functions(text[start+1:end].strip(), row)
             
            if RMLConverter.get_instance().has_registerd_function(name):
                fun = RMLConverter.get_instance().get_registerd_function(name)
                body_parts = body.split(",")
                args = []
                
                for body_part in body_parts:
                    
                    body_part = body_part.strip()
                    if body_part == "*":
                        args.append(row)
                    else:
                        body_part = body_part.translate(str.maketrans({"'":  r"\'"}))
                        args.append(body_part)
                    #if body_part != "*":
                    #    args.append(body_part)
                
                #print("Args", args)
                    
                #if body.endswith('*'):
                #    out = fun(*args, row)
                #else:
                #    out = fun(*args)
                out = fun(*args)
                
            else:
                out = text
            return out
            
            
        else:
            return text
        
        
        
        
            
    
    @staticmethod
    def eval_functions(value, row):
        #p = re.compile('\{(.+)\/?\}')
        
        value = TermUtils.replace_place_holders(value, row)
        
        if value is not None:
        
            p = re.compile('(?<=\%eval:).+?(?=\%)')
        
        
            matches = p.finditer(value)
            s = value

            for match in matches:
                function = match.group(0)
                text = "%eval:" + function + "%"
            
                result = TermUtils.__eval_functions(function, row)
            
                s = s.replace(text, str(result))
            
            value = s
            
        return value
    
    
class RMLParser():
    
    @staticmethod
    def parse(source, format="ttl"):
        g = Graph()
        g.parse(source, format=format)
        
        return TripleMappings.from_rdf(g)

class RMLConverter():
    
    __instance = None
    
    def __init__(self):
        self.__function_registry = dict()
        self.__mapping_dict = MappingsDict()
        RMLConverter.__instance = self
        
    @staticmethod
    def get_instance():
        return RMLConverter.__instance
    
    def convert(self, rml_mapping) -> Graph:
        triple_mappings = RMLParser.parse(rml_mapping)
        
        g = Graph()
        for tm in triple_mappings:
            g += tm.apply()
            
        return g
    
    def get_mapping_dict(self):
        return self.__mapping_dict
    
    def register_fucntion(self, name, fun):
        self.__function_registry.update({name: fun})
        
    def has_registerd_function(self, name):
        return name in self.__function_registry
    
    def get_registerd_function(self, name):
        return self.__function_registry.get(name)
    
class EvalTransformer(Transformer):
    
    def __init__(self, row=None):
        self.__row = row
    
    def start(self, fun):
        return fun[0](*fun[1])
    
    def f_name(self, name):
        
        rml_converter = RMLConverter.get_instance()
        if rml_converter.has_registerd_function(name[0]):
            fun = rml_converter.get_registerd_function(name[0])
            name[0] = fun.__qualname__
            
            return fun
            
        return None
    
    def parameters(self, parameters):
        return parameters
    
    def paramvalue(self, param):
        return str(param[0])
    
    def row(self, val):
        return self.__row
    
    def string(self, val):
        return val[0][1:-1]
    
    def number(self, val):
        return val[0][1:-1]
    
    def const_true(self, val):
        return True
    
    def const_false(self, val):
        return False
    
    def const_none(self, val):
        return None
    
    
class EvalTransformerOld(Transformer):
    
    def __init__(self, row=None):
        self.__row = row
    
    def start(self, fun):
        return "".join(fun)
    
    def f_name(self, name):
        
        rml_converter = RMLConverter.get_instance()
        if rml_converter.has_registerd_function(name[0]):
            fun = rml_converter.get_registerd_function(name[0])
            name[0] = fun.__qualname__
            
        return "".join(name)
    
    def parameters(self, parameters):
        return "(" + ', '.join(parameters) + ")"
    
    def paramvalue(self, param):
        return str(param[0])
    
    def row(self, val):
        return self.__row
    
    def string(self, val):
        return "".join(val)
    
    def number(self, val):
        return "".join(val)
    
    def const_true(self, val):
        return True
    
    def const_false(self, val):
        return False
    
    def const_none(self, val):
        return None
    
class EvalParser():
    
    LARK = Lark.open('grammar.lark',parser='lalr')
    
    @staticmethod        
    def parse(expr, row):
        log.debug("Expr", expr)
        tree = EvalParser.LARK.parse(expr)
        return EvalTransformer(row).transform(tree)
      
      
class PyrmlCMDTool:

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-o", "--output", dest="output",
                    help="Output file. If no choice is provided then standard output is assumed as default.", metavar="RDF out file")
        parser.add_argument("input", help="The input RML mapping file for enabling RDF conversion.")

        self.__args = parser.parse_args()
        
        log.basicConfig(level=log.INFO) 
        
    def do_map(self):
        rml_converter = RMLConverter()
        g = rml_converter.convert(self.__args.input)
        
        if self.__args.output is not None:
            dest_folder = Path(self.__args.output).parent
            
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
                
            with codecs.open(self.__args.output, 'w', encoding='utf8') as out_file:
                out_file.write(g.serialize(format="ntriples").decode('utf-8'))
                
        else:
            log.info(g.serialize(format="ntriples").decode('utf-8'))
            
              
    

if __name__ == '__main__':
    
    PyrmlCMDTool().do_map()
    
    
    