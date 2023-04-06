from abc import abstractmethod
from io import BytesIO
import json
from pyrml import rml_vocab
import time
from typing import Dict, Union, Set, List, Type, Generator

from SPARQLWrapper import SPARQLWrapper
from jsonpath_ng import parse
from pandas.core.frame import DataFrame
from pyrml.pyrml_api import DataSource, TermMap, AbstractMap, TermUtils, graph_add_all, Expression, FunctionNotRegisteredException, NoneFunctionException, ParameterNotExintingInFunctionException
from rdflib import URIRef, Graph, IdentifiedNode
from rdflib.namespace import RDF
from rdflib.plugins.sparql.processor import prepareQuery
from rdflib.term import Node, BNode, Literal, Identifier

import numpy as np
import pandas as pd
import pyrml.rml_vocab as rml_vocab


__author__ = "Andrea Giovanni Nuzzolese"
__email__ = "andrea.nuzzolese@cnr.it"
__license__ = "Apache 2"
__version__ = "0.2.9"
__status__ = "Alpha"





class ObjectMap(AbstractMap):
    
    def to_rdf(self):
        g = Graph()
        g.add((self._id, RDF.type, rml_vocab.OBJECT_MAP_CLASS))
        return g
    
    
class ConstantObjectMap(ObjectMap):
    def __init__(self, value: Node, map_id: URIRef = None):
        super().__init__(map_id, value)
        self.__value = value
        
    @property
    def value(self):
        return self.__value
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        g.add((self._id, rml_vocab.CONSTANT, self.__value))
        return g
    
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            n_rows = data_source.data.shape[0]
            terms = np.array([self.value for x in range(n_rows)], dtype=URIRef)
            
            AbstractMap.get_rml_converter().mappings[self] = terms
            
            return terms
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        mappings_dict = AbstractMap.get_rml_converter().get_mapping_dict()
        
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
        
        
class TermObjectMap(ObjectMap):
    #def __init__(self, reference: Literal = None, template: Literal = None, constant: Union[Literal, URIRef] = None, term_type : URIRef = rml_vocab.LITERAL, language : 'Language' = None, datatype : URIRef = None, map_id: URIRef = None):
    #    super().__init__(map_id, reference if reference is not None else template)
    #    self._reference = reference
    #    self._template = template
    #    self._constant = constant
    #    self._term_type = term_type
    #    self._language : Language = language
    #    self._datatype = datatype
        
        
    def __init__(self, map_id: URIRef, value: Node, **kwargs):
        #def __init__(self, map: Node, map_type: Literal, term_type : URIRef = rml_vocab.LITERAL, language : 'Language' = None, datatype : URIRef = None, map_id: URIRef = None):
        super().__init__(map_id, value)
        self.__value = value
        self.__map_type = kwargs['map_type'] if 'map_type' in kwargs else None
        self.__term_type = kwargs['term_type'] if 'term_type' in kwargs else rml_vocab.LITERAL
        self.__language : Language = kwargs['language'] if 'language' in kwargs else None
        self.__datatype = kwargs['datatype'] if 'datatype' in kwargs else None
        
    @property
    def value(self):
        return self.__value
    
    @property
    def map_type(self):
        return self.__map_type
    
    @property
    def term_type(self):
        return self.__term_type
    
    @property
    def language(self):
        return self.__language
    
    @property
    def datatype(self):
        return self.__datatype
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        predicate = None
        if self.map_type == Literal("reference"):
            predicate = rml_vocab.REFERENCE
        elif self.map_type == Literal("constant"):
            predicate = rml_vocab.CONSTANT
            g.add((self._id, rml_vocab.CONSTANT, self._reference))
        elif self.map_type == Literal("template"):
            predicate = rml_vocab.TEMPLATE
            g.add((self._id, rml_vocab.TEMPLATE, self._template))
            if self._term_type is not None:
                g.add((self._id, rml_vocab.TERM_TYPE, self._term_type))
        elif self.map_type == Literal("functionmap"):
            predicate = rml_vocab.FUNCTION_VALUE
            
            
        if predicate:     
            g.add((self._id, predicate, self._mapped_entity))
            
        if self._language is not None:
            lang_g = self.language.to_rdf()
            g = graph_add_all(g, lang_g)
        elif self._datatype is not None:
            g.add((self._id, rml_vocab.DATATYPE, self.datatype))
        
        if self._term_type is not None:
            g.add((self._id, rml_vocab.TERM_TYPE, self.term_type))
            
        return g
    
    
    def apply(self, data_source: DataSource) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            if self.map_type == Literal("reference"):
                
                data = data_source.dataframe[self.value.value].values
                
                l = lambda val : TermUtils.irify(val) if self.term_type and self.term_type != rml_vocab.LITERAL else val
                
                terms = [l(term) for term in data]
                
                
            elif self.map_type == Literal("template"):
                
                if self.term_type is None or self.term_type == rml_vocab.LITERAL:
                    terms = self._expression.eval_(data_source, False)
                else:
                    terms = self._expression.eval_(data_source, True)
            elif self.map_type == Literal("constant"):
                n_rows = data_source.data.shape[0]
                
                terms = [self.value for x in range(n_rows)]
                
            elif self.map_type == Literal("functionmap") and self._function_map:
                terms = self._function_map.apply(data_source)
            
            else:
                terms = None
            
            if terms is not None:
                # The term is a literal
                if self.term_type is None or self.term_type == rml_vocab.LITERAL:
                    if terms is None:
                        return None
                    elif self.language is not None:
                        #language = TermUtils.eval_template(self._language.value, row, False)
                        #term = Literal(value, lang=self._language.value)
                        
                        languages = self.language.apply(data_source)
                        
                        terms = np.array([Literal(lit, lang=lang) if lit and not pd.isna(lit) else None for lit, lang in zip(terms, languages)], dtype=Literal)
                        
                    elif self.datatype is not None:
                        
                        l = lambda term: Literal(term, datatype=self.datatype) if term and not pd.isna(term) else None
                        
                        terms = np.array([l(term) for term in terms], dtype=Literal)
                    else:
                        terms = np.array([Literal(term) for term in terms], dtype=Literal)
                else:
                    if self.term_type == rml_vocab.BLANK_NODE:
                        
                        l = lambda term: BNode(term) if term and not pd.isna(term) else term
                        terms = np.array([l(term) for term in terms], dtype=BNode)
                        
                    else:
                        
                        l = lambda term: URIRef(term) if term and not pd.isna(term) else term
                        terms = np.array([l(term) for term in terms], dtype=URIRef)
                        
            
            else:
                return None
            
            
            AbstractMap.get_rml_converter().mappings[self] = terms
            
            return terms
                
            
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['TermObjectMap']:
        term_maps = []
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?map ?mapType ?tt ?datatype
                WHERE {
                    {?p rml:reference ?map . BIND("reference" AS ?mapType)}
                    UNION
                    {?p rr:template ?map. BIND("template" AS ?mapType)}
                    UNION
                    {?p rr:constant ?map . BIND("constant" AS ?mapType)}
                    UNION
                    {?p fnml:functionValue ?map . BIND("functionmap" AS ?mapType)}
                    OPTIONAL{?p rr:termType ?tt}
                    OPTIONAL {?p rr:datatype ?datatype}
            }""",
            initNs = {
                "rr": rml_vocab.RR,
                "rml": rml_vocab.RML,
                "fnml": rml_vocab.FNML
                })

        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
            
        for row in qres:
            
            term_object_map = row.p
            
            language = LanguageBuilder.build(g, term_object_map)
            
            tom = TermObjectMap(term_object_map, row.map, map_type=row.mapType, term_type=row.tt, language=language, datatype=row.datatype)
            if row.mapType == Literal("functionmap"):
                tom._function_map = FunctionMap.from_rdf(g, row.map).pop()
            
            term_maps.append(tom)
           
        return term_maps
        



class Language(AbstractMap):
    def __init__(self, map_id: URIRef = None, mapped_entity: URIRef = None):
        super().__init__(map_id, mapped_entity)
        
    @abstractmethod
    def to_rdf(self) -> Graph:
        pass
    
    @abstractmethod
    def apply(self, row):
        pass
        
    @staticmethod
    @abstractmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        pass
    

class ConstantLanguage(Language):
    
    def __init__(self, constant: URIRef, map_id: URIRef = None):
        super().__init__(map_id, constant)
        self._constant = constant
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        g.add((self._id, rml_vocab.LANGUAGE, self._constant))
            
        return g
    
    
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            n_rows = data_source.data.shape[0]
            terms = np.array([self._constant for x in range(n_rows)])
            
            
            return terms
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        query = prepareQuery(
            """
                SELECT DISTINCT ?p ?language
                WHERE {
                    ?p rr:language ?language
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
            term_maps.add(ConstantLanguage(row.language, row.p))
           
        return term_maps
    

class LanguageMap(Language):
        
    def __init__(self, map_id: IdentifiedNode = None, **kwargs):
        super().__init__(map_id)
        
        self.__value: Literal = kwargs['value'] if 'value' in kwargs else None 
        self.__map_type: Literal = kwargs['map_type'] if 'map_type' in kwargs else None
        
    @property
    def value(self):
        return self.__value
    
    @property
    def map_type(self):
        return self.__map_type
        
    def apply(self, data_source: DataSource) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        
        else:
        
            if self.map_type == Literal("reference"):
                
                index = data_source.columns[self.value.value]
                
                data = data_source.data[index]
                
                l = lambda val : TermUtils.irify(val) if self.term_type != rml_vocab.LITERAL else val
                terms = np.array([l(term) for term in data])
                
                
                
                
            elif self.map_type == Literal("template"):
                
                if self.term_type is None or self.term_type == rml_vocab.LITERAL:
                    terms = self._expression.eval_(data_source, False)
                else:
                    terms = self._expression.eval_(data_source, True)
            elif self.map_type == Literal("constant"):
                n_rows = data_source.data.shape[0]
                terms = np.array([self.value for x in range(n_rows)])
                
            elif self.map_type == Literal("functionmap") and self._function_map:
                terms = self._function_map.apply(data_source)
            
            else:
                terms = None
                
            AbstractMap.get_rml_converter().mappings[self] = terms
            return terms    
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        query = prepareQuery(
            """
                SELECT DISTINCT ?objectMap ?languageMap ?value ?map_type
                WHERE {
                    ?objectMap rml:languageMap ?languageMap
                    OPTIONAL{?languageMap rml:reference ?value BIND('reference' as ?map_type)}
                    OPTIONAL{?languageMap rr:template ?value BIND('template' as ?map_type)}
                    OPTIONAL{?languageMap rr:constant ?value BIND('constant' as ?map_type)}
            }""",
            initNs = {
                "rr": rml_vocab.RR,
                "rml": rml_vocab.RML
                })

        if parent is not None:
            qres = g.query(query, initBindings = { "objectMap": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            term_maps.add(LanguageMap(row.languageMap, value=row.value, map_type=row.map_type))
           
        return term_maps
    
class LanguageBuilder():
    
    @staticmethod
    def build(g: Graph, parent: Union[URIRef, BNode]) -> Set[Language]:
        
        ret = None
        if (parent, rml_vocab.LANGUAGE, None) in g:
            ret = ConstantLanguage.from_rdf(g, parent)
        elif (parent, rml_vocab.LANGUAGE_MAP, None) in g:
            ret = LanguageMap.from_rdf(g, parent)
        else:
            return None
        
        if ret is None or len(ret) == 0:
            return None
        else:
            return ret.pop()
    
class Predicate(AbstractMap):
    def __init__(self, map_id: URIRef = None, mapped_entity: URIRef = None):
        super().__init__(map_id, mapped_entity)

class ConstantPredicate(Predicate):
    
    def __init__(self, constant: URIRef, map_id: URIRef = None):
        super().__init__(map_id, constant)
        self._constant = constant
        
    def to_rdf(self) -> Graph:
        g = super().to_rdf()
        
        g.add((self._id, rml_vocab.PREDICATE, self._constant))
            
        return g
    
    
    def apply(self, data_source: DataSource) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            n_rows = data_source.data.shape[0]
            terms = np.array([URIRef(self._constant) if self._constant else None for x in range(n_rows)], dtype=URIRef)
            
            AbstractMap.get_rml_converter().mappings[self] = terms 
            
            return terms
        
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
    
    #def __init__(self, triple_mapping : Union[BNode, URIRef], reference: Literal = None, template: Literal = None, constant: URIRef = None, map_id: URIRef = None):
    #    super().__init__(map_id, reference if reference is not None else template if template is not None else constant)
    #    self._reference = reference
    #    self._template = template
    #    self._constant = constant
    #    
    #    self._triple_mapping = triple_mapping
        
    def __init__(self, map_id: IdentifiedNode = None, **kwargs):
        super().__init__(map_id)
        
        self.__predicate_expression: Literal = kwargs['predicate_expression'] if 'predicate_expression' in kwargs else None 
        self.__predicate_expression_type: Literal = kwargs['predicate_expression_type'] if 'predicate_expression_type' in kwargs else None
        
        
    @property
    def predicate_expression(self):
        return self.__predicate_expression
    
    @property
    def predicate_expression_type(self):
        return self.__predicate_expression_type
    
    
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            if self._predicate_ref_type == Literal("functionmap") and self.function_map:
                terms = self.function_map.apply(data_source, rdf_term_type=URIRef)
                
            else:
                if self.predicate_expression_type == Literal("template"):
                    terms = Expression.create(self.predicate_expression).eval_(data_source, True)
                elif self.predicate_ref_type == Literal("constant"):
                    n_rows = data_source.data.shape[0]
                    
                    terms = np.array([URIRef(self.predicate_expression) if self.predicate_expression else None for x in range(n_rows)], dtype=URIRef)
                    
                    
                elif self.predicate_ref_type == Literal("reference"):
                    
                    
                    index = data_source.columns[self.value.value]
                
                    data = data_source.data[index]
                
                    l = lambda val : URIRef(val) if val else None
                
                    terms = np.array([l(term) for term in data], dtype=URIRef)
                
        
            AbstractMap.get_rml_converter().mappings[self] = terms
        
            return terms
        
    
        
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> Set[TermMap]:
        term_maps = []
        query = prepareQuery(
            """
                SELECT DISTINCT ?map ?termType
                WHERE {
                    {
                        ?predicateMap rml:reference ?map
                        BIND("reference" AS ?termType)
                    }
                    UNION
                    {
                        ?predicateMap rr:template ?map
                        BIND("template" AS ?termType)
                        
                    }
                    UNION
                    {
                        ?predicateMap rr:constant ?map
                        BIND("constant" AS ?termType)
                    }
                    UNION
                    {
                        ?predicateMap fnml:functionValue ?map
                        BIND("functionmap" AS ?termType)
                    }
                    
            }""",
            initNs = {
                "rr": rml_vocab.RR,
                "rml": rml_vocab.RML,
                "fnml": rml_vocab.FNML
                })

        if parent is not None:
            qres = g.query(query, initBindings = { "predicateMap": parent})
        else:
            qres = g.query(query)
        
        for row in qres:
            #pm = PredicateMap(row.tripleMap, row.reference, row.template, row.constant, row.predicateMap)
            pm = PredicateMap(row.tripleMap, row.map, row.termType, row.predicateMap)
            
            if row.termType == Literal("functionmap"):
                pm._function_map = FunctionMap.from_rdf(g, row.map).pop()
            
            term_maps.add(pm)
           
        return term_maps
    
    
class PredicateBuilder():
    
    @staticmethod
    def build(g: Graph, pom: IdentifiedNode) -> List[Predicate]:
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?pred ?predtype
                WHERE {
                    { 
                        ?pom rr:predicate ?pred
                        BIND('shortconstant' as ?predtype) 
                    }
                    UNION
                    { 
                        ?pom rr:predicateMap ?pred
                        BIND('map' as ?predtype) 
                    }
            }""", 
            initNs = { "rr": rml_vocab.RR})

        qres = g.query(query, initBindings = { "pom": pom})
        
        predicates = [] 
        for res in qres:
            predtype = res.predtype
            predicate_ref = res.pred
            if predtype.value == 'shortconstant':
                predicates += [ConstantPredicate(predicate_ref, pom)]
            else:
                predicates += PredicateMap.from_rdf(g, predicate_ref)
                
            
        return predicates
        

class PredicateObjectMap(AbstractMap):
    
    
    def __init__(self, map_id: IdentifiedNode, **kwargs):
        
        super().__init__(map_id)
        self._predicates: List[Predicate] = kwargs['predicates'] if 'predicates' in kwargs else None
        self.__object_maps: List[ObjectMap] = kwargs['object_maps'] if 'object_maps' in kwargs else None
    
    @property
    def predicates(self) -> List[Predicate]:
        return self._predicates
    
    @property
    def object_maps(self) -> List[ObjectMap]:
        return self.__object_maps
    
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        
        else:
            preds = [pm.apply(data_source) for pm in self._predicates]
            objs = [om.apply(data_source) for om in self.__object_maps]
            
            
            init = True
            preds_objs = None
            for predicates in preds:
                predicates = np.array(predicates).reshape(len(predicates), 1)
                
                for objects in objs:
                    objects = np.array(objects).reshape(len(objects), 1)
                    
                    p_o = np.concatenate([predicates, objects], axis=1)
                    
                    if init:
                        preds_objs = p_o
                        init = False
                    else:
                        preds_objs = np.concatenate([preds_objs, p_o], axis=0)
                
            
            #preds_objs = np.array([[pred, obj] for pred in predicates for obj in objects])
            
            
            
            #shp = preds_objs.shape
            
            #preds_objs.reshape(shp[1], shp[0])
            
            AbstractMap.get_rml_converter().mappings[self] = preds_objs
            return preds_objs
    
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None, parent_var: str = None) -> List['PredicateObjectMap']:
        query = prepareQuery(
            """
                SELECT DISTINCT ?pom
                WHERE {
                    ?tm rr:predicateObjectMap ?pom .
                    ?pom rr:predicate|rr:predicateMap ?pred; 
                        rr:object|rr:objectMap ?om
            }""", 
            initNs = { "rr": rml_vocab.RR})

        if parent is not None:
            if not parent_var or parent_var != 'pom': 
                qres = g.query(query, initBindings = { "tm": parent})
            else:
                qres = g.query(query, initBindings = { "pom": parent})
        else:
            qres = g.query(query)
            
        lmbd = lambda graph : lambda row :  PredicateObjectMap.__build(graph, row)
        return list(map(lmbd(g), qres))
        
    
    @staticmethod
    def __build(g, row):
        
        predicates = PredicateBuilder.build(g, row.pom)
        
        object_maps = ObjectMapBuilder.build(g, row.pom)
        
        if predicates is not None and object_maps is not None:
            return PredicateObjectMap(row.pom, predicates=predicates, object_maps=object_maps)
        else:
            return None;
    
class ObjectMapBuilder():
    
    @staticmethod
    def build(g: Graph, parent: IdentifiedNode) -> List[ObjectMap]:
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?om ?omtype
                WHERE {
                    { 
                        ?pom rr:object ?om
                        BIND('shortconstant' as ?omtype) 
                    }
                    UNION
                    { 
                        ?pom rr:objectMap ?om
                        BIND('map' as ?omtype) 
                    }
            }""", 
            initNs = { "rr": rml_vocab.RR})

        qres = g.query(query, initBindings = { "pom": parent})
        
        object_maps = [] 
        for res in qres:
            omtype = res.omtype
            om = res.om
            if omtype.value == 'shortconstant':
                object_maps += [ConstantObjectMap(om, None)]
            else:
                if (om, rml_vocab.PARENT_TRIPLES_MAP, None) in g:
                    object_maps += ReferencingObjectMap.from_rdf(g, om)
                else:
                    object_maps += TermObjectMap.from_rdf(g, om)
                
        return object_maps


class Join(AbstractMap):
    def __init__(self, map_id: URIRef = None, **kwargs):
        super().__init__(map_id)
        self.__child: Literal = kwargs['child'] if 'child' in kwargs else None
        self.__parent: Literal = kwargs['parent'] if 'parent' in kwargs else None
        
    @property
    def child(self) -> Literal:
        return self.__child
    
    @property
    def parent(self) -> Literal:
        return self.__parent
    
    def apply(self, data_source: DataSource = None) -> np.array:
        return self
    
    def to_rdf(self) -> Graph:
        g = Graph()
        
        if self.__child is not None and self.__parent is not None:
            join = self._id
            g.add((join, rml_vocab.CHILD, self.__child))
            g.add((join, rml_vocab.PARENT, self.__parent))
            
        return g
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['Join']:
        
        query = prepareQuery(
            """
                SELECT DISTINCT ?join ?child ?parent
                WHERE {
                    ?p rr:child ?child ;
                        rr:parent ?parent
            }""", 
            initNs = { "rr": rml_vocab.RR})

        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
            
        return [Join(row.join, child=row.child, parent=row.parent) for row in qres]
        
    

class InputFormatNotSupportedError(Exception):
    def __init__(self, format):
        self.message = "The format %s is currently not supported"%(format)


class LogicalSource(AbstractMap):
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        self.__separator : str = kwargs['separator'] if 'separator' in kwargs else ',' 
        self.__query : str = kwargs['query'] if 'query' in kwargs else None
        self.__reference_formulation : URIRef = kwargs['reference_formulation'] if '__reference_formulation' in kwargs else rml_vocab.CSV
        self.__iterator: URIRef = kwargs['iterator'] if 'iterator' in kwargs else None
        self.__sources : List[Source] = kwargs['sources'] if 'sources' in kwargs else None
        
    @property
    def sources(self) -> List['Source']:
        return self.__sources
    
    @property    
    def reference_formulation(self) -> URIRef:
        return self.__reference_formulation
    
    @property
    def separator(self) -> str:
        return self.__separator
    
    @property
    def query(self) -> str:
        return self.__query
    
    @property
    def iterator(self) -> URIRef:
        return self.__iterator
    
    def apply(self, row: pd.Series = None) -> Generator:
        if self in AbstractMap.get_rml_converter().logical_sources:
            dfs = AbstractMap.get_rml_converter().logical_sources[self]
        else: 
            if self.__separator is None:
                sep = ','
            else:
                sep = self.__separator
            
            dfs = []
            for source in self.sources:
                if isinstance(source, BaseSource):
                    if self.__reference_formulation == rml_vocab.JSON_PATH and self.__iterator:
                        json_data = json.load(source._mapped_entity)
                        
                        jsonpath_expr = parse(self.__iterator)
                        matches = jsonpath_expr.find(json_data)
                
                        data = [match.value for match in matches]
                        
                        df = pd.json_normalize(data)
                        
                    else:
                        df = pd.read_csv(source._mapped_entity, sep=sep, dtype=str)
                elif isinstance(source, CSVSource):
                    df = pd.read_csv(source.url, sep=source.delimiter, dtype=str)
                elif isinstance(source, SPARQLSource) and self.__query:
                    
                    sparql = SPARQLWrapper(source.endpoint)
                    sparql.setQuery(self.__query)
            
                    sparql.setReturnFormat(source.result_format)
                    rs = sparql.queryAndConvert()
            
                    if source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_CSV'):
                        df = pd.read_csv(BytesIO(rs), sep=',', dtype=str)
                    elif source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_TSV'):
                        df = pd.read_csv(BytesIO(rs), sep='\t', dtype=str)
                    elif source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_JSON') and self.__iterator:
                
                        jsonpath_expr = parse(self.__iterator)
                        matches = jsonpath_expr.find(rs)
        
                        data = [match.value for match in matches]
                        df = pd.json_normalize(data)
            
                    elif self.__result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_XML') and self.__iterator:
                        df = pd.read_xml(BytesIO(rs), xpath=self.__iterator, dtype=str)
                        
                    else:
                        df = None
                else:
                    df = None
                
                
                #df.columns = df.columns.str.replace(r' ', '_')
                
                dfs.append(df)
                
                AbstractMap.get_rml_converter().logical_sources[self] = dfs
                
        return dfs
            

        
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List[TermMap]:
        term_maps = []
            
        sparql = """
            SELECT DISTINCT ?ls ?rf ?sep ?ite ?query
            WHERE {
                ?tm rml:logicalSource ?ls
                OPTIONAL {?ls rml:referenceFormulation ?rf}
                OPTIONAL {?ls crml:separator ?sep}
                OPTIONAL {?ls rml:iterator ?ite}
                OPTIONAL {?ls rml:query ?query}
            }"""
            #OPTIONAL {?ls crml:separator ?sep}
        
        query = prepareQuery(sparql, 
                initNs = { 
                    "rml": rml_vocab.RML,
                    "crml": rml_vocab.CRML,
                    "csvw": rml_vocab.CSVW,
                    "sd": rml_vocab.SD,
                    "ql": rml_vocab.QL

                })
        
        if parent is not None:
            qres = g.query(query, initBindings = { "tm": parent})
        else:
            qres = g.query(query)
                    
        for row in qres:
            
            '''
            if row.sourcetype.value == 'uri' and ref_formulation == rml_vocab.CSV:
                source = SourceBuilder.build_source(g, row.source, FileTableSource)
            elif row.sourcetype.value == 'uri' and ref_formulation == rml_vocab.JSON_PATH:
                source = SourceBuilder.build_source(g, row.source, FileJSONSource)
            elif row.sourcetype.value == 'sparql':
                source = SourceBuilder.build_source(g, row.source, SPARQLServiceSource)
            '''
            
            sources = Source.from_rdf(g, row.ls)
            
            separator = row.sep.value if row.sep else ','
            query = row.query.value if row.query else None
            iterator = row.ite if row.ite else None
            rf = row.rf if row.rf else rml_vocab.CSV
            
            ls = LogicalSource(row.ls, None, sources=sources, separator=separator, query=query, iterator=iterator, reference_formalation=rf)
            term_maps.append(ls)
            
        return term_maps
    

class GraphMap(AbstractMap):
    
    def __init__(self, map_id: IdentifiedNode, value: Node = None, **kwargs):
        super().__init__(map_id, value)
        self.__term_type = kwargs['term_type'] if 'term_type' in kwargs else None
        
    @property
    def term_type(self):
        return self.__term_type
        
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            
            if self.term_type == Literal("reference"):
                
                index = data_source.columns[self._mapped_entity.value]
                
                data = data_source.data[index]
                
                l = lambda val : URIRef(TermUtils.irify(val))
                
                terms = np.array([l(term) for term in data], dtype=URIRef)
                
                
            elif self.term_type == Literal("template"):
                
                terms = self._expression.eval_(data_source, True)
            elif self.term_type == Literal("constant"):
                n_rows = data_source.data.shape[0]
                terms = np.array([self._mapped_entity for x in range(n_rows)])
                
            elif self.term_type == Literal("functionmap") and self._function_map:
                terms = self._function_map.apply(data_source)
            
            else:
                terms = None
        
        
            AbstractMap.get_rml_converter().mappings[self] = terms
            return terms 
        

    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['GrapMap']:
        
        sparql = """
            SELECT DISTINCT ?gm ?g ?mode
            WHERE {
                { ?p rr:graphMap ?gm . 
                  ?gm rr:constant ?g
                  BIND("constant" AS ?mode)
                }
                UNION
                { ?p rr:graph ?g
                  BIND("constant" AS ?mode)
                  BIND(BNODE() AS ?gm)
                }
                UNION
                { ?p rr:graphMap ?gm . 
                  ?gm rr:template ?g
                  BIND("template" AS ?mode)
                }
                UNION
                { ?p rr:graphMap ?gm .
                  ?gm rml:reference ?g
                  BIND("reference" AS ?mode)
                }
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { "rr": rml_vocab.RR, "rml": rml_vocab.RML })
        
        if parent is not None:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
                    
        
        return [GraphMap(row.gm, row.g, term_type=row.mode) for row in qres]
    
class SubjectMap(AbstractMap):
    def __init__(self, map_id: IdentifiedNode, value: Node = None, **kwargs):
        
        # , term_type: Literal, class_: Set[URIRef] = None, graph_map: GraphMap = None, map_id: URIRef = None
        super().__init__(map_id, value)
        self.__value = value
        self.__classes: List[URIRef] = kwargs['_classes'] if '_classes' in kwargs else None
        self.__graph_maps: List[GraphMap] = kwargs['graph_maps'] if 'graph_maps' in kwargs else None
        self.__term_type: Literal = kwargs['term_type'] if 'term_type' in kwargs else None
        self._function_map: FunctionMap = kwargs['function_map'] if 'function_map' in kwargs else None
    
    @property
    def value(self) -> Literal:
        return self.__value
    
    @property
    def _classes(self) -> URIRef:
        return self.__classes
    
    @property
    def graph_maps(self) -> List[GraphMap]:
        return self.__graph_maps
    
    @property
    def term_type(self) -> Literal:
        return self.__term_type
                
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            
            if self.term_type == Literal("functionmap") and self.function_map:
                terms = self.function_map.apply(data_source, rdf_term_type=URIRef)
            else:
                if self.term_type == Literal("template"):
                    terms  = Expression.create(self.value).eval_(data_source, True)
                   
                elif self.term_type == Literal("reference"):
                    index = data_source.columns[self.value.value]
                
                    data = data_source.data[index]
                
                    l = lambda val : URIRef(TermUtils.irify(val)) if val else None
                
                    terms = np.array([l(term) for term in data], dtype=URIRef)
                     
                    
                elif self.term_type == Literal("constant"):
                    
                    n_rows = data_source.data.shape[0]
                    l = lambda val : URIRef(TermUtils.irify(val)) if val else None
                    terms = np.array([l(self.value.value) for x in range(n_rows)], dtype=URIRef)
                
                
            AbstractMap.get_rml_converter().mappings[self] = terms
            return terms
    
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        
        sparql = """
            SELECT DISTINCT ?map ?sm ?termType
            WHERE {
                ?tm rr:subjectMap ?sm
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
                UNION
                {
                 ?sm fnml:functionValue ?map
                 BIND("functionmap" as ?termType)
                }
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { "rr": rml_vocab.RR, "rml": rml_vocab.RML, "fnml": rml_vocab.FNML})
        
        if parent is not None:
            qres = g.query(query, initBindings = { "tm": parent})
        else:
            qres = g.query(query)
                    
        x = [SubjectMap.__create(parent, g, row) for row in qres]
        
        return x
        
    
    @staticmethod
    def __create(_id: IdentifiedNode, graph: Graph, row):
        
        classes = graph.objects(row.sm, rml_vocab.CLASS, True)
        
        graph_maps = GraphMap.from_rdf(graph, row.sm)
        
        function_map = None   
        if row.termType == Literal("functionmap"):
            function_map = FunctionMap.from_rdf(graph, row.map)[0]
        
        return SubjectMap(_id, row.map, _classes=classes, graph_maps=graph_maps, term_type=row.termType, function_map=function_map)
        
    

class FunctionMap(AbstractMap):
    
    
    def __init__(self, funct_map: Node, poms: Set[PredicateObjectMap]):
        super().__init__(funct_map, None)
        self._poms = poms
        
    def to_rdf(self):
        g = Graph()
        
        for pom in self._poms:
            graph_add_all(g, pom.to_rdf())
        
    def apply(self, data_source: DataSource = None, **kwargs) -> np.array:
        
        
        class Function:

            def __init__(self, arr: np.array):
                
                self.__args = dict()
                for i in range(0, len(arr), 2):
                    key = str(arr[i])
                    value = arr[i+1]
                    
                    if key not in self.__args:
                        self.__args[key] = value
                    else:
                        if isinstance(self.__args[key], list):
                            self.__args[key].append(value)
                        else:
                            self.__args[key] = [self.__args[key], value]
                
                self.__function_ref = self.__args['https://w3id.org/function/ontology#executes'].value if 'https://w3id.org/function/ontology#executes' in self.__args else None
                del(self.__args['https://w3id.org/function/ontology#executes'])
                
                
            def evaluate(self):
                if self.__function_ref:
                    if AbstractMap.get_rml_converter().has_registerd_function(self.__function_ref):
                        fun = AbstractMap.get_rml_converter().get_registerd_function(self.__function_ref)
                        
                        if 'rdf_term_type' in kwargs:
                            try:
                                out = kwargs['rdf_term_type'](fun.evaluate(self.__args))
                            except Exception as e:
                                out = None
                            
                            return out
                            
                        else:
                            return fun.evaluate(self.__args)
                            
                    
                    else:
                        raise FunctionNotRegisteredException(function_ref)
                else:
                    raise NoneFunctionException()
            
            
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
        
            function_ref = None
            
            #if 'rdf_term_type' in kwargs:
            
            pom_matrix : np.array = None
            
            pom_applied = [pom.apply(data_source) for pom in self._poms]
            try:
                pom_matrix = np.concatenate((pom_applied), axis=1)    
            except Exception as e:
                pass
            
            return np.array([Function(row).evaluate() for row in pom_matrix], dtype=Function) 
            
            
            
     
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['FunctionMap']:
        sparql = """
            SELECT DISTINCT ?funVal ?pom ?logicalSource
            WHERE {
                ?funVal rr:predicateObjectMap ?pom   
                OPTIONAL{?funVal rml:logicalSource ?logicalSource}
            }"""
        
        query = prepareQuery(sparql, 
                initNs = { "rr": rml_vocab.RR, "rml": rml_vocab.RML, "fnml": rml_vocab.FNML})
        
        if parent is not None:
            qres = g.query(query, initBindings = { "funVal": parent})
        else:
            qres = g.query(query)
                    
        poms = []
        for row in qres:
            pom_uri = row.pom
            
            pom = PredicateObjectMap.from_rdf(g, pom_uri, 'pom')
            
            for pom_occurrence in pom:
                poms.append(pom_occurrence)
        
        return [FunctionMap(parent, poms)]


class TripleMapping():
    def __init__(self, _id: IdentifiedNode, **kwargs):
        self.__id = _id
        self.__logical_source : LogicalSource = kwargs['source'] if 'source' in kwargs else None 
        self.__subject_map : List[SubjectMap] = kwargs['subject_map'] if 'subject_map' in kwargs else None
        self.__predicate_map : Union[Predicate, PredicateMap] = kwargs['predicate_map'] if 'predicate_map' in kwargs else None
        self.__object_map : ObjectMap = kwargs['object_map'] if 'object_map' in kwargs else None
        self.__condition : str = kwargs['condition'] if 'condition' in kwargs else None
        
        
    @property
    def id(self):
        return self.__id
    
    @property
    def logical_source(self) -> LogicalSource:
        return self.__logical_source
    
    @property    
    def subject_map(self) -> SubjectMap:
        return self.__subject_map
    
    @property
    def predicate_map(self) -> Union[Predicate, PredicateMap]:
        return self.__predicate_map
    
    @property
    def object_map(self) -> ObjectMap:
        return self.__object_map
    
    @property
    def condition(self):
        return self.__condition
         
    def apply(self, row: tuple) -> Generator:
        
        
        try:
            if self.subject_map:
                
                sbj_representation = self.subject_map.apply(row)
                
                if sbj_representation:
                                                             
                    if self.predicate_map:
                        pred_representation = self.predicate_map.apply(row)
                        
                        if self.object_map:
                            obj_representation = self.object_map.apply(row)
                            
                            for sbj in sbj_representation:
                                for pred in pred_representation:
                                    for obj in obj_representation:
                                    
                                        if self.subject_map.graph_maps:
                                            for graph_map in self.subject_map.graph_maps: 
                                                _graph_ctxs: URIRef = graph_map.apply()
                                                for _graph_ctx in _graph_ctxs:
                                                    yield sbj, pred, obj, _graph_ctx
                                        else:
                                            yield sbj, pred, obj
                        
        except Exception as e:
            raise e
        
        
class TripleMappings(AbstractMap):
    '''
    logical_sources: List[LogicalSource],
    subject_maps: List[SbjectMap],
    predicate_object_maps: List[PredicateObjectMap] = None, 
    iri: URIRef = None,
    condition: str = None
    '''
    def __init__(self,
                 mapping: IdentifiedNode,
                 value: Node,
                 **kwargs):
        super().__init__(mapping, value)
        self.__logical_sources : List[LogicalSource] = kwargs['sources'] if 'sources' in kwargs else None 
        self.__subject_maps : List[SubjectMap] = kwargs['subject_maps'] if 'subject_maps' in kwargs else None
        self.__predicate_object_maps : List[PredicateObjectMap] = kwargs['predicate_object_maps'] if 'predicate_object_maps' in kwargs else None
        self.__condition : str = kwargs['condition'] if 'condition' in kwargs else None
        
    @property
    def logical_sources(self) -> LogicalSource:
        return self.__logical_sources
    
    @property    
    def subject_maps(self) -> List[SubjectMap]:
        return self.__subject_maps
    
    @property
    def predicate_object_maps(self) -> List[PredicateObjectMap]:
        return self.__predicate_object_maps
    
    @property
    def condition(self):
        return self.__condition
    
    
    
    def apply(self, data_source: DataSource = None) -> np.array:
        start_time = time.time()
        msg = "\t TripleMapping %s" % self._id
        #print(msg)
        
        triples : DataFrame = pd.DataFrame(columns=['s', 'p', 'o'], dtype=object)
        for logical_source in self.logical_sources:
            
            for df in logical_source.apply():
                
                data_source = DataSource(df)
                
                sbj_maps = [subject_map.apply(data_source) for subject_map in self.subject_maps]
                
                grfs = [graph for graph in [graphs for graphs in [subject_map.graph_maps for subject_map in self.subject_maps]]]
                
                graph_maps = [graph_map.apply(data_source) for subject_map in self.subject_maps for graph_map in subject_map.graph_maps]
                
                
                for sbj_representation in sbj_maps:
                    if self.predicate_object_maps is not None:
                            
                        results = None
                        
                        for pom in self.predicate_object_maps:
                            try:
                                for object_map in pom.object_maps:
                                    
                                    df_join = None
                                    if isinstance(object_map, ReferencingObjectMap) and object_map.join_conditions:
                                        df_left = df
                                        df_left["__pyrml_sbj_representation__"] = sbj_representation
                                        parent_triple_mappings = object_map.parent_triples_maps
                                        
                                        
                                        results = pd.DataFrame(columns=['0_l', '0_r'])
                                        
                                        for parent_triple_mapping in parent_triple_mappings:
                                            for parent_logical_source in parent_triple_mapping.logical_sources:
                                                
                                                for df_right in parent_logical_source.apply():
                                                    
                                                    pandas_condition = parent_triple_mapping.condition
                                                    if pandas_condition:
                                                        df_right = df_right[eval(pandas_condition)]
                                                        
                                                    join_conditions = object_map.join_conditions
                                                    
                                                    left_ons = []
                                                    right_ons = []
                                                    
                                                    for join_condition in join_conditions:
                                                        left_ons.append(join_condition.child.value)
                                                        right_ons.append(join_condition.parent.value)
                                                    
                                                    if not df_left.empty and not df_right.empty:
                                                        
                                                        df_join = df_left.merge(df_right, how='inner', suffixes=(None, "_r"), left_on=left_ons, right_on=right_ons, sort=False)
                                                    
                                                        pom_representation = pom.apply(DataSource(df_join))
                                                        
                                    else:
                                
                                        
                                        '''
                                        if isinstance(object_map, ReferencingObjectMap):
                                            for ptm in object_map.parent_triples_maps:
                                                
                                                pandas_condition = ptm.condition
                                                if pandas_condition:
                                                    df_pom = df[eval(pandas_condition)]
                                                    
                                                    pom_representation = pom.apply(DataSource(df_pom))
                                                    
                                                        
                                        
                                        else:
                                            
                                            
                                        ''' 
                                        pom_representation = pom.apply(data_source)   
                                            
                                    if pom_representation is not None:
                                        #results: pd.DataFrame = pd.concat([sbj_representation, pom_representation], axis=1, sort=False)
                                        
                                        #triples_representation = [[s, p, o] for s in sbj_representation for p, o in pom_representation]
                                        
                                        end = len(pom_representation)
                                        if df_join is not None:
                                            # if referencing object map with joins
                                            sbjs = df_join["__pyrml_sbj_representation__"]
                                        else:
                                            sbjs = sbj_representation
                                        
                                        chunk_size = len(sbjs)
                                        
                                        for i in range(0, end, chunk_size):
                                            triples_representation = [[s, p_o[0], p_o[1]] for s, p_o in zip(sbjs, pom_representation[i:i+chunk_size])]
                                            
                                            
                                            if len(graph_maps) > 0:
                                                #triples_representation = [[s, p, o, ctx] for ctx in graph_maps for s, p, o in triples_representation]
                                                
                                                for gmaps in graph_maps:
                                                    g_end = len(gmaps)
                                                    
                                                    for k in range(0, g_end, chunk_size):
                                                        triples_representation = [[t[0], t[1], t[2], g] for t, g in zip(triples_representation, gmaps[k:k+chunk_size])]
                                                        
                                                        triples_df = pd.DataFrame(triples_representation, columns=['s', 'p', 'o', 'ctx'])
                                                        triples_df.dropna(inplace=True)
                                                    
                                                        triples = pd.concat([triples, triples_df], axis=0)
                                                    
                                                
                                            else:
                                            
                                                triples_df = pd.DataFrame(triples_representation, columns=['s', 'p', 'o'])
                                                triples_df.dropna(inplace=True)
                                                
                                                triples = pd.concat([triples, triples_df], axis=0)
                                            
                            
                            except Exception as e:
                                raise e
                
        
        elapsed_time_secs = time.time() - start_time
        
        #msg = "\t Triples Mapping %s: %s secs" % (self._id, elapsed_time_secs)
        msg = "\t\t done in %s secs" % (elapsed_time_secs)
        return triples.to_numpy(dtype=Node)  
    
         
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List[TermMap]:
        
        sparql = """
            SELECT DISTINCT ?tm
            WHERE {
                %PLACE_HOLDER%
                ?tm rml:logicalSource ?source ;
                    rr:subjectMap ?sm
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
        
        if parent:
            qres = g.query(query, initBindings = { "p": parent})
        else:
            qres = g.query(query)
        
        
        return set([TripleMappings.__build(g, row) for row in qres])
    
    @staticmethod
    def __build(g: Graph, row) -> 'TripleMappings':
        sources = LogicalSource.from_rdf(g, row.tm)
                        
        subject_maps: List[SubjectMap] = SubjectMap.from_rdf(g, row.tm)
        
        predicate_object_maps = PredicateObjectMap.from_rdf(g, row.tm)
        '''
        if row.pom:
            predicate_object_maps = PredicateObjectMap.from_rdf(g, row.tm)
        else:
            predicate_object_maps = []
        ''' 
            
        '''
        rr:class statements are used for generating PredicateObjectMap
        of the form (ConstantPredicate(RDF.type), ConstantObjectMap(_class)).
        These PredicateObjectMaps will generate triples/quads for typing individuals.
        '''
            
        for subject_map in subject_maps:
            for _class in subject_map._classes:
                pom = PredicateObjectMap(BNode(str(_class)), predicates=[ConstantPredicate(RDF.type)], object_maps=[ConstantObjectMap(_class)])
                predicate_object_maps.append(pom)
            
                     
        return TripleMappings(row.tm, None, sources=sources, subject_maps=subject_maps, predicate_object_maps=predicate_object_maps)
        
    
    
        
    
class ReferencingObjectMap(ObjectMap):
    def __init__(self, map_id: URIRef = None, **kwargs):
        super().__init__(map_id)
        self.__parent_triples_maps: List[TripleMappings] = kwargs['parent_triples_maps'] if 'parent_triples_maps' in kwargs else None 
        self.__joins: List[Join] = kwargs['joins'] if 'joins' in kwargs else None
        
    @property
    def parent_triples_maps(self) -> List[TripleMappings]:
        return self.__parent_triples_maps
    
    @property
    def join_conditions(self) -> List[Join]:
        return self.__joins
    
    def apply(self, data_source: DataSource = None) -> np.array:
        
        if self in AbstractMap.get_rml_converter().mappings:
            return AbstractMap.get_rml_converter().mappings[self]
        else:
            ref = np.array([subject_map.apply(data_source) for tm in self.parent_triples_maps for subject_map in tm.subject_maps], dtype=URIRef)
            ret = np.concatenate(ref, axis=0, dtype=URIRef)
            AbstractMap.get_rml_converter().mappings[self] = ret
            return ret
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List[TermMap]:
        term_maps = []
        
        
        query_join = prepareQuery(
            """
            SELECT DISTINCT ?join
            WHERE {
                ?p rr:joinCondition ?join
            }""", 
            initNs = { "rr": rml_vocab.RR})
    
        join_qres = g.query(query_join, initBindings = { "p": parent})
                
        joins: List[Join] = []
        for row_join in join_qres:
            
            joins += Join.from_rdf(g, row_join.join)
            
        parent_triples = TripleMappings.from_rdf(g, parent)
        if parent_triples:
            rmo = ReferencingObjectMap(parent, joins=joins, parent_triples_maps=parent_triples)
            term_maps.append(rmo)
           
        return term_maps
    
class Source(AbstractMap):
    
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode) -> 'Source':
        sparql = '''
            SELECT DISTINCT ?source ?sourcetype
            WHERE {
                {
                    ?ls rml:source ?source
                    FILTER(isLiteral(?source))
                    BIND('plain' AS ?sourcetype)
                }
                UNION
                {
                    ?ls rml:source ?source .
                    ?source csvw:url ?url
                    OPTIONAL {?s csvw:dialect/csvw:delimiter ?sep}
                    BIND('table' AS ?sourcetype)  
                }
                UNION
                {
                    ?ls rml:source ?source .
                    ?source sd:endpoint ?s
                    BIND('sparql' AS ?sourcetype)  
                }
            }'''
            
        query = prepareQuery(sparql, 
            initNs = { 
                "rml": rml_vocab.RML,
                "crml": rml_vocab.CRML,
                "csvw": rml_vocab.CSVW,
                "sd": rml_vocab.SD,
                "ql": rml_vocab.QL

            })
    
        if parent is not None:
            qres = g.query(query, initBindings = { "ls": parent})
        else:
            qres = g.query(query)
              
        return [Source.__build(g, row) for row in qres]
    
    @staticmethod         
    def __build(g, row):
        
        sourcetype = row.sourcetype.value
        if sourcetype == 'plain':
            return BaseSource.from_rdf(g, row.source)
        elif sourcetype == 'table':
            return CSVSource.from_rdf(g, row.source)
        elif sourcetype == 'sparql':
            return SPARQLSource.from_rdf(g, row.source)
        else:
            return None
            
    def apply(self, row: pd.Series = None) -> Generator:
        return self


class BaseSource(Source):
    
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode) -> 'BaseSource':
        
        return BaseSource(parent, parent.value)
        

class CSVSource(Source):
    
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        
        self.__delimiter = kwargs['delimiter'] if 'delimiter' in kwargs else ','
        self.__encoding = kwargs['encoding'] if 'encoding' in kwargs else 'UTF-8'
        self.__url = value
        
    
    @property
    def delimiter(self):
        return self.__delimiter
    
    @property
    def encoding(self):
        return self.__encoding
    
    @property
    def url(self):
        return self.__url
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode) -> 'TableSource':
        
        csvw = 'http://www.w3.org/ns/csvw#'
        
        dialect = URIRef(csvw+'dialect')
        
        urls = g.objects(parent, URIRef(csvw+'url'), True)
        for url in urls:
            url = url.value
            
            delimiters = g.objects(parent, (dialect/URIRef(csvw+'delimiter')), True)
            if delimiters:
                delimiter = next(delimiters, None)
                if delimiter: 
                    delimiter = delimiter.value
            else:
                delimiter = ','
                
            encodings = g.objects(parent, (dialect/URIRef(csvw+'encoding')), True)
            if encodings:
                encoding = next(encodings, None)
                if encoding:
                    encoding = encoding.value
            else:
                encoding = 'UTF-8'
                
            return CSVSource(parent, url, delimiter=delimiter, encoding=encoding)
                
        else:
            return None
        
class SPARQLSource(Source):
    
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        
        self.__endpoint : URIRef = kwargs['endpoint'] if 'endpoint' in kwargs else None
        self.__supported_language : URIRef = kwargs['supported_language'] if 'supported_language' in kwargs else URIRef('http://www.w3.org/ns/sparql-service-description#SPARQL11Query')
        self.__result_format : URIRef = kwargs['result_format'] if 'result_format' in kwargs else URIRef('http://www.w3.org/ns/formats/SPARQL_Results_JSON')
        
    @property
    def endpoint(self):
        return self.__endpoint
    
    @property
    def supported_language(self):
        return self.__supported_language
    
    @property
    def result_format(self):
        return self.__result_format
        
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode) -> 'TableSource':
        
        csvw = 'http://www.w3.org/ns/csvw#'
        
        dialect = URIRef(csvw+'dialect')
        
        urls = g.objects(parent, URIRef(csvw+'url'), True)
        if urls and len(urls) > 0:
            url = urls[0].value
            
            delimiters = g.objects(parent, (dialect/URIRef(csvw+'delimiter')), True)
            if delimiters and len(delimiters) > 0:
                delimiter = delimiters[0].value
            else:
                delimiter = ','
                
            encodings = g.objects(parent, (dialect/URIRef(csvw+'encoding')), True)
            if encodings and len(encodings) > 0:
                encoding = encodings[0].value
            else:
                encoding = 'UTF-8'
                
            return CSVSource(parent, url, delimiter=delimiter, encoding=encoding)
                
        else:
            return None
    
        
        
class SourceBuilder:
    
    @staticmethod
    def build_source(g: Graph, source: Identifier, c: Type[Source], *args) -> Source:
        return c(g, source)

class RMLFunction():
    
    def __init__(self, fun_id, function, **kwargs):
        self.__fun_id = fun_id
        self.__function = function
        self.__params = {}
        
        for k,v in kwargs.items():
            self.__params[v] = k
        
    @property
    def fun_id(self):
        return self.__fun_id
    
    @property
    def function(self):
        return self.__function
    
    def get_param(self, param_id):
        return self.__params[param_id]
    
    def has_param(self, param_id):
        return param_id in self.__params
    
    def get_params(self):
        return self.__params
    
    def evaluate(self, params: Dict[str, str]):
        input_values = {}
        for param in params:
            
            if param in self.__params:
                input_values[self.__params[param]] = params[param] 
            else:
                raise ParameterNotExintingInFunctionException(self.__function, param)
            
        try:
            out = self.__function(**input_values)
            return out
        except Exception as e:
            pass
            #print(e)
                
            #g += tm.apply()
    
        
