from pandas.core.series import Series
__author__ = "Andrea Giovanni Nuzzolese"
__email__ = "andrea.nuzzolese@cnr.it"
__license__ = "Apache 2"
__version__ = "0.1"
__status__ = "Pre-Alpha"

from abc import ABC, abstractclassmethod
from builtins import staticmethod
import re, os, unidecode
from typing import Dict, Union, Set

from pandas.core.frame import DataFrame
from rdflib import URIRef, Graph, plugin
from rdflib.query import Processor, Result
from rdflib.namespace import RDF
from rdflib.plugins.sparql.processor import prepareQuery, SPARQLProcessor, SPARQLResult
from rdflib.term import Node, BNode, Literal, Identifier
from rdflib.parser import StringInputSource

import numpy as np
import pandas as pd
import pyrml.rml_vocab as rml_vocab
import sys

import multiprocessing
from multiprocessing import  Pool
import logging

from lark import Lark
from lark.visitors import Transformer

from jinja2 import Environment, FileSystemLoader, Template

from jsonpath_ng import jsonpath, parse
import json

def graph_add_all(g1, g2):
    for (s,p,o) in g2:
        g1.add((s,p,o))
    return g1

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
        g = Graph('IOMemory')
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
            if self._reference.value in row:
                value = row[self._reference.value]
            else:
                value = None
        if self._template is not None:
            
            value = TermUtils.eval_functions(self._template.value, row, False)
                
        
        if value != value:
            literal = None
        elif self._language is not None:
            language = TermUtils.eval_functions(self._language.value, row, False)
            literal = Literal(value, lang=language)
        elif self._datatype is not None:
            datatype = TermUtils.eval_functions(str(self._datatype), row, False)
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
        
        
class TermObjectMap(ObjectMap):
    def __init__(self, reference: Literal = None, template: Literal = None, constant: Union[Literal, URIRef] = None, term_type : URIRef = rml_vocab.LITERAL, language : Literal = None, datatype : URIRef = None, map_id: URIRef = None):
        super().__init__(map_id, reference if reference is not None else template)
        self._reference = reference
        self._template = template
        self._constant = constant
        self._term_type = term_type
        self._language = language
        self._datatype = datatype
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        if self._reference is not None:
            g.add((self._id, rml_vocab.REFERENCE, self._reference))
        elif seld._constant is not None:
            g.add((self._id, rml_vocab.CONSTANT, self._reference))
        elif self._template is not None:
            g.add((self._id, rml_vocab.TEMPLATE, self._template))
            if self._term_type is not None:
                g.add((self._id, rml_vocab.TERM_TYPE, self._term_type))
            
        if self._language is not None:
            g.add((self._id, rml_vocab.LANGUAGE, self._language))
        elif self._datatype is not None:
            g.add((self._id, rml_vocab.DATATYPE, self._datatype))
        
        if self._term_type is not None:
            g.add((self._id, rml_vocab.TERM_TYPE, self._term_type))
            
        return g
    
    
    def __convertion(self, row):
        
        term = None
        value = None
        
        if self._reference is not None:
            if self._reference.value in row:
                value = row[self._reference.value]
                if value == value and self._term_type is not None and self._term_type != rml_vocab.LITERAL:
                    value = TermUtils.irify(value)
            else:
                value = None
        elif self._template is not None:
            if self._term_type is None or self._term_type == rml_vocab.LITERAL:
                value = TermUtils.eval_functions(self._template.value, row, False)
            else:
                value = TermUtils.eval_functions(self._template.value, row, True)
        elif self._constant is not None:
            value = self._constant
        
        if value is not None and value==value:
            # The term is a literal
            if self._term_type is None or self._term_type == rml_vocab.LITERAL:
                if value != value:
                    term = None
                elif self._language is not None:
                    language = TermUtils.eval_functions(self._language.value, row, False)
                    term = Literal(value, lang=language)
                elif self._datatype is not None:
                    datatype = TermUtils.eval_functions(str(self._datatype), row, False)
                    term = Literal(value, datatype=datatype)
                else:
                    term = Literal(value)
            else:
                if self._term_type == rml_vocab.BLANK_NODE:
                    term = BNode(value)
                else:
                    term = URIRef(value)
        
        return term
    
    def apply(self, df: DataFrame):
        
        l = lambda x: self.__convertion(x)
                
        df_1 = df.apply(l, axis=1)
        
        return df_1
    
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?reference ?template ?constant ?tt ?language ?datatype
                WHERE {
                    OPTIONAL{?p rml:reference ?reference}
                    OPTIONAL{?p rr:template ?template}
                    OPTIONAL{?p rr:constant ?constant}
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
            term_maps.add(TermObjectMap(row.reference, row.template, row.constant, row.tt, row.language, row.datatype, row.p))
           
        return term_maps
        

class Predicate(AbstractMap):
    def __init__(self, map_id: URIRef = None, mapped_entity: URIRef = None):
        super().__init__(map_id, mapped_entity)
        
    @abstractclassmethod
    def to_rdf(self) -> Graph:
        pass
    
    
    
    @abstractclassmethod
    def apply(self, df: DataFrame):
        pass
        
    @staticmethod
    @abstractclassmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        pass


class ConstantPredicate(Predicate):
    
    def __init__(self, constant: URIRef, map_id: URIRef = None):
        super().__init__(map_id, constant)
        self._constant = constant
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        g.add((self._id, rml_vocab.PREDICATE, self._constant))
            
        return g
    
    
    def apply(self, df: DataFrame):
        
        return self._constant
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?predicate
                WHERE {
                    ?p rr:predicate ?predicate
                }
            """,
            initNs = {
                "rr": rml_vocab.RR,
                "rml": rml_vocab.RML
                })

        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            term_maps.add(ConstantPredicate(row.predicate, row.p))
           
        return term_maps


class PredicateMap(Predicate):
    def __init__(self, triple_mapping : Union[BNode, URIRef], reference: Literal = None, template: Literal = None, constant: URIRef = None, map_id: URIRef = None):
        super().__init__(map_id, reference if reference is not None else template if template is not None else constant)
        self._reference = reference
        self._template = template
        self._constant = constant
        
        self._triple_mapping = triple_mapping
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        g.add(self._triple_mapping, rml_vocab.PREDICATE_MAP, self._id)
        if self._reference is not None:
            g.add((self._id, rml_vocab.REFERENCE, self._reference))
        elif self._template is not None:
            g.add((self._id, rml_vocab.TEMPLATE, self._template))
        elif self._constant is not None:
            g.add((self._id, rml_vocab.CONSTANT, self._constant))
            
        return g
    
    
    def __convertion(self, row):
        
        predicate = None
        
        if self._reference is not None:
            if self._reference.value in row:
                value = row[self._reference.value]
            else:
                value = None
        elif self._template is not None:
            
            value = TermUtils.eval_functions(self._template.value, row, True)
            
        elif self._constant is not None:
            
            value = self._constant
                
        
        if value != value:
            predicate = None
        else:
            
            if isinstance(predicate, URIRef):
                predicate = value
            else:
                predicate = URIRef(value)
        
        return predicate
    
    def apply(self, df: DataFrame):
        
        l = lambda x: self.__convertion(x)
                
        df_1 = df.apply(l, axis=1)
        
        return df_1
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        query = prepareQuery(
            """
                SELECT DISTINCT ?tripleMap ?predicateMap ?reference ?template ?constant
                WHERE {
                    ?tripleMap rr:predicateMap ?predicateMap
                    OPTIONAL{?predicateMap rml:reference ?reference}
                    OPTIONAL{?predicateMap rr:template ?template}
                    OPTIONAL{?predicateMap rr:constant ?constant}
            }""",
            initNs = {
                "rr": rml_vocab.RR,
                "rml": rml_vocab.RML
                })

        if parent is not None:
            qres = g.query(query, initBindings = { "tripleMap": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            term_maps.add(PredicateMap(row.tripleMap, row.reference, row.template, row.constant, row.predicateMap))
           
        return term_maps
    
    
class PredicateBuilder():
    
    @staticmethod
    def build(g: Graph, parent: Union[URIRef, BNode]) -> Set[Predicate]:
        
        ret = None
        if (parent, rml_vocab.PREDICATE, None) in g:
            ret = ConstantPredicate.from_rdf(g, parent)
        elif (parent, rml_vocab.PREDICATE_MAP, None) in g:
            ret = PredicateMap.from_rdf(g, parent)
        else:
            return None
        
        if ret is None or len(ret) == 0:
            return None
        else:
            return ret.pop()
        

class PredicateObjectMap(AbstractMap):
    def __init__(self, predicate: Predicate, object_map: ObjectMap, map_id: URIRef = None):
        super().__init__(map_id, predicate)
        self._predicate = predicate
        self.__object_map = object_map
        
    def get_predicate(self) -> Predicate:
        return self._predicate
    
    def get_object_map(self) -> ObjectMap:
        return self.__object_map
    
    def to_rdf(self) -> Graph:
        g = Graph('IOMemory')
        g.add((self._id, RDF.type, rml_vocab.PREDICATE_OBJECT_MAP_CLASS))
        g = graph_add_all(g, self._predicate.to_rdf())
        
        g.add((self._id, rml_vocab.OBJECT_MAP, self.__object_map.get_id()))
        
        g = graph_add_all(g, self.__object_map.to_rdf())
        #g += self.__object_map.to_rdf()
        
        return g
    
    def apply(self, df: DataFrame):
        
        df_1 = self.__object_map.apply(df)
        
        predicate = self.get_mapped_entity()
        if isinstance(predicate, ConstantPredicate):
            try:
                df_1 = df_1.apply(lambda x: (predicate.apply(df), x))
            except:
                return None
        elif isinstance(predicate, PredicateMap):
            try:
                df_1 = pd.concat([predicate.apply(df), df_1], axis=1, sort=False)
                df_1 = df_1[[0, 1]].apply(tuple, axis=1)
            except:
                return None
        return df_1
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?pom ?predicate ?om
                WHERE {
                    ?pom rr:objectMap ?om
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
        
        predicate = PredicateBuilder.build(g, row.pom)
        
        object_map = None
        if isinstance(row.om, URIRef):
            if row.om in mapping_dict:
                object_map = mapping_dict.get(row.om)
            else:
                object_map = ObjectMapBuilder.build(g, row.om)
                mapping_dict.add(object_map)
        else:
            object_map = ObjectMapBuilder.build(g, row.om)
            
        if predicate is not None and object_map is not None:
            return PredicateObjectMap(predicate, object_map, row.pom)
        else:
            return None;
    
class ObjectMapBuilder():
    
    @staticmethod
    def build(g: Graph, parent: Union[URIRef, BNode]) -> Set[ObjectMap]:
        
        ret = None
        
        if (parent, rml_vocab.PARENT_TRIPLES_MAP, None) in g:
            ret = ReferencingObjectMap.from_rdf(g, parent)
        else:
            ret = TermObjectMap.from_rdf(g, parent)
        
        '''
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
        '''
        
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
        g = Graph('IOMemory')
        
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
    

class InputFormatNotSupportedError(Exception):
    def __init__(self, format):
        self.message = "The format %s is currently not supported"%(format)


class LogicalSource(AbstractMap):
    def __init__(self, source: Literal, separator: str = None, map_id: URIRef = None, reference_formulation: URIRef = None, iterator: Literal = None):
        super().__init__(map_id, source)
        self.__separator = separator
        if reference_formulation is None:
            self.__reference_formulation = rml_vocab.CSV
        else:
            self.__reference_formulation = reference_formulation
            
        self.__iterator = iterator
        
    def get_source(self) -> Literal:
        return self.get_mapped_entity()
    
    def get_reference_formulation(self) -> URIRef:
        return self.__reference_formulation
    
    def get_separator(self) -> str:
        return self.__separator
    
    def to_rdf(self):
        g = Graph('IOMemory')
        g.add((self._id, RDF.type, rml_vocab.BASE_SOURCE))
        g.add((self._id, rml_vocab.SOURCE, self.get_source()))
        g.add((self._id, rml_vocab.REFERENCE_FORMULATION, self.__reference_formulation))
        if self.__iterator is not None:
            g.add((self._id, rml_vocab.ITERATOR, self.__iterator))
        if self.__separator is not None:
            g.add((self._id, rml_vocab.SEPARATOR, self.__separator))
        
        return g
    
    def apply(self):
    
        loaded_logical_sources = RMLConverter.get_instance().get_loaded_logical_sources()
        
        if self._mapped_entity:
            logical_source_uri = self._mapped_entity.value
            
            if logical_source_uri in loaded_logical_sources:
                return loaded_logical_sources[logical_source_uri]
            
        if self.__separator is None:
            sep = ','
        else:
            sep = self.__separator
            
        if self.__reference_formulation == rml_vocab.JSON_PATH:
            jsonpath_expr = parse(self.__iterator)

            with open(self._mapped_entity.value) as f:
                json_data = json.load(f)
    
                matches = jsonpath_expr.find(json_data)
    
                data = [match.value for match in matches]
                df = pd.json_normalize(data)
            
        elif self.__reference_formulation == rml_vocab.CSV:
            df = pd.read_csv(self._mapped_entity.value, sep=sep, dtype=str)
        else:
            raise InputFormatNotSupportedError(self.__reference_formulation)
            
        loaded_logical_sources.update({logical_source_uri: df})
        return df 
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
            
        sparql = """
            SELECT DISTINCT ?ls ?source ?rf ?sep ?ite
            WHERE {
                ?p rml:logicalSource ?ls .
                ?ls rml:source ?source .
                OPTIONAL {?ls rml:referenceFormulation ?rf}
                OPTIONAL {?ls crml:separator ?sep}
                OPTIONAL {?ls rml:iterator ?ite}
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
            
            ls = LogicalSource(source, row.sep, row.ls, row.rf, row.ite)
            term_maps.add(ls)
            
        return term_maps
    

class GraphMap(AbstractMap):
    
    def __init__(self, mapped_entity: Node, map_id: URIRef = None):
        super().__init__(map_id, mapped_entity)
        
    
    def to_rdf(self):
        g = Graph('IOMemory')
        
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
    def __init__(self, mapped_entity: Node, term_type: Literal, class_: Set[URIRef] = None, graph_map: GraphMap = None, map_id: URIRef = None):
        super().__init__(map_id, mapped_entity)
        self.__class = class_
        self.__graph_map = graph_map
        self.__term_type = term_type
    
    def get_class(self) -> URIRef:
        return self.__class
    
    def get_graph_map(self) -> GraphMap:
        return self.__graph_map
                
    def to_rdf(self):
        g = Graph('IOMemory')
        subject_map = self._id
        
        '''
        if isinstance(self._mapped_entity, Literal):
            g.add((subject_map, rml_vocab.TEMPLATE, self._mapped_entity))
        elif isinstance(self._mapped_entity, URIRef):
            g.add((subject_map, rml_vocab.CONSTANT, self._mapped_entity))
        '''
        
        if self.__term_type == Literal("template"):
            g.add((subject_map, rml_vocab.TEMPLATE, self._mapped_entity))
        elif self.__term_type == Literal("constant"):
            g.add((subject_map, rml_vocab.CONSTANT, self._mapped_entity))
        elif self.__term_type == Literal("reference"):
            g.add((subject_map, rml_vocab.REFERENCE, self._mapped_entity))
            
        if self.__class is not None:
            for c in self.__class:
                g.add((subject_map, rml_vocab.CLASS, self.__class))
            
        if self.__graph_map is not None:
            graph_map_g = self.__graph_map.to_rdf()
             
            g.add((subject_map, rml_vocab.GRAPH_MAP, self.__graph_map.get_id()))
            
            g = graph_add_all(g, graph_map_g)
            #g = g + graph_map_g
            
        
        return g
        
    def __convert(self, row):
        term = None
        if self.__term_type == Literal("template") or self.__term_type == Literal("constant"):
            term = TermUtils.urify(self._mapped_entity, row)
        elif self.__term_type == Literal("reference"):
            term = URIRef(row[self._mapped_entity.value])
            
        return term
        
    
    def apply(self, df: DataFrame):
        
        #l = lambda x: TermUtils.urify(self._mapped_entity, x)
        l = lambda x: self.__convert(x)
        
        #l = lambda x: print(x['id'])  
            
        
        df_1 = df.apply(l, axis=1)
        df_1.replace('', np.nan, inplace=True)
        df_1.dropna(inplace=True)
        return df_1
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
            
        sparql = """
            SELECT DISTINCT ?sm ?map ?termType ?type ?gm ?g
            WHERE {
                ?p rr:subjectMap ?sm .
                { ?sm rr:template ?map
                  BIND("template" AS ?termType)
                }
                UNION
                { ?sm rr:constant ?map
                  BIND("constant" AS ?termType)
                }
                UNION
                { ?sm rml:reference ?map
                  BIND("reference" AS ?termType)
                }
                OPTIONAL {?sm rr:class ?type}
                OPTIONAL {
                    { ?sm rr:graphMap ?gm }
                    UNION
                    { ?sm rr:graph ?g }
                }
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { "rr": rml_vocab.RR, "rml": rml_vocab.RML})
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
                    
        subject_maps = {}
        for row in qres:
            subject_map = None
            if isinstance(row.sm, URIRef) and row.sm in mappings_dict:
                subject_map = mappings_dict.get(row.sm)
                    
                    
            if subject_map is None:
            
                if row.sm.n3() in subject_maps:
                    subject_map = subject_maps[row.sm.n3()]
                    subject_map.__class.add(row.type)
                else:
                    subject_map = SubjectMap.__create(row)
                    subject_maps.update({row.sm.n3(): subject_map})
                    if isinstance(row.sm, URIRef):
                        mappings_dict.add(subject_map)
                        
            else:
                subject_map.__class.add(row.type)
                
        
        return set(subject_maps.values())
    
    @staticmethod
    def __create(row):
        
        graph_map = None
        if row.gm is not None or row.g is not None:
            graph_map = GraphMap.from_rdf(row.g, row.sm).pop()
                
        return SubjectMap(row.map, row.termType, {row.type}, graph_map, row.sm)
    
    
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
        g = Graph('IOMemory')
        g.add((self._id, RDF.type, rml_vocab.TRIPLES_MAP))
        g.add((self._id, rml_vocab.LOGICAL_SOURCE, self.__logical_source.get_id()))
        g.add((self._id, rml_vocab.SUBJECT_MAP, self.__subject_map.get_id()))
        
        if self.__condition is not None:
            g.add((self._id, rml_vocab.CONDITION, self.__condition))
        
        g = graph_add_all(g, self.__logical_source.to_rdf())
        g = graph_add_all(g, self.__subject_map.to_rdf())
        #g += self.__logical_source.to_rdf()
        #g += self.__subject_map.to_rdf()
        
        for key, value in self.__predicate_object_maps.items():
            g.add((self._id, rml_vocab.PREDICATE_OBJECT_MAP, key))
            g = graph_add_all(g, value.to_rdf())
            #g += value.to_rdf()
            
        return g
    
    @staticmethod
    def __triplify_series(entry, entity_types : Set[URIRef], graph : Graph):
        
        try:
            
            graph.add((entry['0_l'], entry['0_r'][0], entry['0_r'][1]))
            if entity_types:
                TripleMappings.__add_types(entry['0_l'], entity_types, graph)
            return graph
        except:
            pass
        
    @staticmethod
    def __add_types(entry, entity_types : Set[URIRef], graph : Graph):
        
        try:
            for entity_type in entity_types:
                graph.add((entry, RDF.type, entity_type))
            
            return graph
        except:
            pass
    
    def apply(self):
        g = Graph('IOMemory')
        
        df = self.__logical_source.apply()
        
        if self.__condition is not None and self.__condition.strip() != '':
            df = df[eval(self.__condition)]
        
        sbj_representation = self.__subject_map.apply(df)
        
                                                             
        if self.__predicate_object_maps is not None:
            
            #triplification = lambda x: TripleMappings.__triplify_series(x, self.__subject_map.get_class(), g)
            
            for pom in self.__predicate_object_maps.values():
                pom_representation = pom.apply(df) 
                
                if pom_representation is not None:
            
                    try:
                        
                        if isinstance(sbj_representation, pd.Series):
                            sbj_representation=sbj_representation.to_frame().reset_index()
                        
                        pom_representation=pom_representation.to_frame().reset_index()
                        
                        results = sbj_representation.merge(pom_representation, how='left', suffixes=("_l", "_r"), left_on="index", right_on="index", sort=False)
                        #results = pd.concat([sbj_representation, pom_representation], axis=1, sort=False)
                    except Exception as e:
                        raise e
                    
                    
                    #results.apply(triplification, axis=1)
                    
                    for k,v in results.iterrows():
                        
                        try:
                            g.add((v['0_l'], v['0_r'][0], v['0_r'][1]))
                            
                            if self.__subject_map.get_class() is not None:
                                for type in self.__subject_map.get_class():
                                    g.add((v['0_l'], RDF.type, type))
                        
                        except:
                            pass
                    
        elif self.__subject_map.get_class() is not None:
            
            #triplification = lambda x: TripleMappings.__add_types(x, self.__subject_map.get_class(), g)
            #sbj_representation.apply(triplification)
            
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
        

        #l = lambda x: TermUtils.urify(self.__parent_triples_map.get_subject_map().get_mapped_entity(), x)
        
                
        if self.__join is not None:
            
            left_on = self.__join.get_child()
            right_on = self.__join.get_parent()
            ptm = RMLConverter.get_instance().get_mapping_dict().get(self.__parent_triples_map.get_id())
            right = ptm.get_logical_source().apply()
            
            df_1 = df.join(right.set_index(right_on.value), how='inner', lsuffix="_l", rsuffix="_r", on=left_on.value, sort=False).rename(columns={left_on.value: right_on.value})
            
        else:
            df_1 = df
        
        #df_1 = df_1.apply(l, axis=1)
        
        #df_1 = self.__parent_triples_map.get_subject_map().apply(df_1)
        
        #df_1.replace('', np.nan, inplace=True)

        #df_1.dropna(inplace=True)
        
        #return df_1
        return self.__parent_triples_map.get_subject_map().apply(df_1)
    
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
    def urify(entity, row):
        if isinstance(entity, Literal):
            s = TermUtils.eval_functions(entity, row, True)
            if s is not None and s.strip() != '':
                return URIRef(s)
            else:
                return float('nan')
        elif isinstance(entity, URIRef):
            return entity
        
    @staticmethod
    def replace_place_holders(value, row, is_iri):
        #p = re.compile('\{(.+)\/?\}')
        p = re.compile('(?<=\{).+?(?=\})')
        
        matches = p.finditer(value)
        
        input_value = value
        s = value
        
        for match in matches:
            column = match.group(0)
            span = match.span(0)
            
            span_start = span[0]-2
            span_end = span[1]+1
            
            column_key = column.strip()
            
            if column_key in row:
                text = "{( )*" + column + "( )*}"
                
                if row[column_key] != row[column_key]:
                    s = re.sub(text, '', s)
                else:
                    if column not in row.index:
                        column += "_l"
                    
                    cell_value = str(row[column_key])
                    
                    if span_start>0 and span_end<len(input_value):
                        if input_value[span_start] == '\'' and input_value[span_end] == '\'':
                            cell_value = cell_value.replace('\'', '\\\\\'')
                        elif input_value[span_start] == '"' and input_value[span_end] == '"':
                            cell_value = cell_value.replace('"', '\\\"')
                    
                    
                    if is_iri:
                        value = TermUtils.irify(cell_value)
                    else:
                        value = cell_value
                    s = re.sub(text, value, s)
            else:
                return None
            #print(str(row[column]))
            
        
        return s
        
    @staticmethod
    def __eval_functions(text, row):
        
        return EvalParser.parse(text, row)
    
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
                out = None
            return out
            
            
        else:
            return text
        
        
        
        
            
    
    @staticmethod
    def eval_functions(value, row, is_iri):
        #p = re.compile('\{(.+)\/?\}')
        
        if value is not None:
            p = re.compile('(?<=\%eval:).+?(?=\%)')
        
            matches = p.finditer(value)
            s = value

            for match in matches:
                function = match.group(0)
                text = "%eval:" + function + "%"
                
                function = TermUtils.replace_place_holders(function, row, False)
                
                result = TermUtils.__eval_functions(function, row)
                
                if result is None:
                    result = ""
                s = s.replace(text, result)
            
            value = TermUtils.replace_place_holders(s, row, is_iri)
            
        return value
        
    @staticmethod
    def irify(string):
    
        '''
        The followint regex pattern allows to check if the input string is provided as a valid URI. E.g. http://dati.isprambiente.it/rmn/Ancona.jpg
        '''
        regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        
        '''
        In case the input sstring is not a valid URI than the function applies the irification (i.e. the transormation aimed at removing characters that prevent
        an IRI to be valid).
        '''
        if re.match(regex, string) is None:
    
            string = unidecode.unidecode(string)
            string = string.lower();
            string = re.sub(r'[\']', '', string)
            string = re.sub(r'[;.,&"???!]', '', string)
            string = re.sub(r'[ \/]', '_', string);
            string = re.sub(r'[\(\)]', '', string);
            string = re.sub(r'\-$', '', string);
            string = re.sub(r'(\-)+', '_', string);
            string = re.sub(r'(\_)+', '_', string);
            
        return string
    
    
class RMLParser():
    
    def parse(source, format="ttl"):
        g = Graph('IOMemory')
        g.parse(source, format=format)
        
        return TripleMappings.from_rdf(g)
        
        '''
        g_2 = Graph('IOMemory')
        for tm in triple_mappings:
            g_2 += tm.apply()
            g1 = tm.to_rdf()
            for l in g1.serialize(format=format).splitlines():
                if l: print(l.decode('ascii'))
        '''

class RMLConverter():
    
    __instance = None
    
    def __init__(self):
        self.__function_registry = dict()
        self.__mapping_dict = MappingsDict()
        self.__loaded_logical_sources = dict()
        RMLConverter.__instance = self
        
    @staticmethod
    def get_instance():
        #if RMLConverter.__instance is None:
        #    RMLConverter.__instance = RMLConverter()
            
        return RMLConverter.__instance
        
    @staticmethod
    def set_instance(instance):
        #if RMLConverter.__instance is None:
        #    RMLConverter.__instance = RMLConverter()
            
        RMLConverter.__instance = instance
        
    
    def convert(self, rml_mapping, multiprocessed=False, template_vars: Dict[str, str] = None) -> Graph:
    
        plugin.register("sparql", Result, "rdflib.plugins.sparql.processor", "SPARQLResult")
        plugin.register("sparql", Processor, "rdflib.plugins.sparql.processor", "SPARQLProcessor")
        
        
        if template_vars is not None:
            file_loader = FileSystemLoader('.')
            env = Environment(loader=file_loader)
            template = env.get_template(rml_mapping)
            rml_mapping_template = template.render(template_vars)
            rml_mapping = StringInputSource(rml_mapping_template.encode('utf-8'))
        
        
        triple_mappings = RMLParser.parse(rml_mapping)
        
        g = Graph('IOMemory')
        
        
        if multiprocessed:
            processes = multiprocessing.cpu_count()
        
            tms = np.array_split(np.array(list(triple_mappings)), processes)
            pool = Pool(initializer=initializer, initargs=(RMLConverter.__instance,), processes=processes)
            graphs = pool.map(pool_map, tms)
            pool.close()
            pool.join()
        
        
            for graph in graphs:
                graph_add_all(g, graph)
        
        else:
            for tm in triple_mappings:
                triples = tm.apply()
                g = graph_add_all(g, triples)
        
        return g
    
    def get_mapping_dict(self):
        return self.__mapping_dict
    
    def register_function(self, name, fun):
        self.__function_registry.update({name: fun})
        
    def unregister_function(self, name):
        del self.__function_registry[name]
        
    def has_registerd_function(self, name):
        return name in self.__function_registry
    
    def get_registerd_function(self, name):
        return self.__function_registry.get(name)
        
    def get_loaded_logical_sources(self):
        return self.__loaded_logical_sources
        
def initializer(rml_converter):
    logger = logging.getLogger("rdflib")
    logger.setLevel(logging.ERROR)
    
    logger.disabled = True
    
    RMLConverter.set_instance(rml_converter)
        
def pool_map(triple_mappings):
    g = Graph()
    for tm in triple_mappings:
        triples = tm.apply()
        graph_add_all(g, triples)
        
    return g
            #g += tm.apply()


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
        return param[0]
    
    def row(self, val):
        return self.__row
    
    def string(self, val):
        return val[0][1:-1]
    
    def number(self, val):
        return val[0]
    
    def const_true(self, val):
        return True
    
    def const_false(self, val):
        return False
    
    def const_none(self, val):
        return None
    
    
class EvalParser():
    
    dirname = os.path.dirname(__file__)
    lark_grammar_file = os.path.join(dirname, 'grammar.lark')
    LARK = Lark.open(lark_grammar_file,parser='lalr')
    
    @staticmethod
    def parse(expr, row):
        #logging.debug("Expr", expr)
        tree = EvalParser.LARK.parse(expr)
        return EvalTransformer(row).transform(tree)
    
    
    
