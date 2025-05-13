
from abc import abstractmethod
from io import BytesIO
import json
from pyrml import rml_vocab
import time
from typing import Dict, Union, Set, List, Type, Generator

from SPARQLWrapper import SPARQLWrapper, CSV, JSON, XML, TSV
from jsonpath_ng import parse
from pandas.core.frame import DataFrame
from pyrml.pyrml_api import PyRML, DataSource, TermMap, AbstractMap, TermUtils, graph_add_all, Expression, FunctionNotRegisteredException, NoneFunctionException, ParameterNotExintingInFunctionException, RMLModelException
from rdflib import URIRef, Graph, IdentifiedNode
from rdflib.namespace import RDF, Namespace, XSD
from rdflib.plugins.sparql.processor import prepareQuery
from rdflib.term import Node, BNode, Literal, Identifier, URIRef, _is_valid_langtag, _castPythonToLiteral, _castLexicalToPython

import numpy as np
import pandas as pd
import pyrml.rml_vocab as rml_vocab
import xml.etree.ElementTree as ET
import sqlalchemy as sa


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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            n_rows = data_source.data.shape[0]
            terms = np.array([self.value for x in range(n_rows)], dtype=URIRef)
            
            PyRML.get_mapper().mappings[self] = terms
            
            return terms
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        term_maps = set()
        mappings_dict = PyRML.get_mapper().get_mapping_dict()
        
        g.tr
        
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
    
    
    def apply(self, data_source: DataSource, **kwargs) -> np.array:
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            if self.map_type == Literal("reference"):
                
                try:
                    column_name = self.value.value
                    if column_name not in data_source.dataframe.columns:
                        column_name = column_name.lower()
                        if column_name not in data_source.dataframe.columns:
                            column_name = column_name.upper()
                            
                    data = data_source.dataframe[column_name].values                    
                except KeyError as e:
                    print(e)
                    data = []
                
                def l(val):
                    if self.term_type and self.term_type == rml_vocab.IRI:
                        val = TermUtils.irify(val)  
                    elif self.term_type and self.term_type == rml_vocab.BLANK_NODE:
                        val = BNode(val) 
                    return val
                        
                
                #l = lambda val : TermUtils.irify(val) if self.term_type and self.term_type == rml_vocab.IRI else BNode(val) if self.term_type and self.term_type == rml_vocab.BLANK_NODE else val
                
                terms = [l(term) for term in data]
                
                
            elif self.map_type == Literal("template"):
                
                if self.term_type == rml_vocab.LITERAL or self.language or self.datatype:
                    terms = self._expression.eval_(data_source, False)
                else:
                    terms = self._expression.eval_(data_source, True)
                    self.__term_type = URIRef(rml_vocab.RR + 'IRI')
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
                        
                        def l(term, lang):
                            if not TermUtils.is_valid_language_tag(lang):
                                raise RMLModelException(f'The language tag {lang} is not a valid IETF BCP 47 language tag.')
                                
                            if isinstance(term, list) or isinstance(term, np.ndarray):
                                return np.array([Literal(lit, lang=lang) if lit and not pd.isna(lit) else lit for lit in term], dtype=Literal)
                            else:
                                return Literal(term, lang=lang) if term and not pd.isna(term) else None
                        
                        languages = self.language.apply(data_source)
                        
                        #terms = np.array([Literal(lit, lang=lang) if lit and not pd.isna(lit) else None for lit, lang in zip(terms, languages)], dtype=Literal)
                        terms = np.array([l(lit, lang) for lit, lang in zip(terms, languages)], dtype=Literal)
                        
                    elif self.datatype is not None:
                        
                        def l(term):
                            if isinstance(term, list) or isinstance(term, np.ndarray):
                                return np.array([Literal(_castPythonToLiteral(_castLexicalToPython(str(lit), self.datatype), self.datatype)[0], datatype=self.datatype) if lit is not None and not pd.isna(lit) else lit for lit in term], dtype=Literal)
                            else:
                                _term = Literal(_castPythonToLiteral(_castLexicalToPython(str(term), self.datatype), self.datatype)[0], datatype=self.datatype) if term is not None and not pd.isna(term) else None
                                return _term
                        
                        terms = np.array([l(term) for term in terms], dtype=Literal)
                    else:
                        
                        def l(term):
                            
                            
                            def get_item(_item):
                                if not PyRML.INFER_LITERAL_DATATYPES:
                                    return Literal(str(_item))
                                
                                if isinstance(_item, np.datetime64):
                                    return Literal(_item, datatype=XSD.dateTime)
                                else:
                                    try:
                                        return Literal(_item.item())
                                    except Exception:
                                        return Literal(_item)
                            
                            if isinstance(term, list) or isinstance(term, np.ndarray):
                                return np.array([get_item(lit) if lit or not pd.isna(lit) else lit for lit in term], dtype=Literal)
                            else:
                                return get_item(term) if term or not pd.isna(term) else None
                        
                        
                        terms = np.array([l(term) for term in terms], dtype=Literal)
                else:
                    if self.term_type == rml_vocab.BLANK_NODE:
                        
                        l = lambda term: BNode(term) if term and not pd.isna(term) else term
                        terms = np.array([l(term) for term in terms], dtype=BNode)
                        
                    else:
                        def l(term):
                            if isinstance(term, list) or isinstance(term, np.ndarray):
                                return np.array([URIRef(TermUtils.irify(t)) if t and not pd.isna(t) else t for t in term], dtype=URIRef)
                            else:
                                return URIRef(TermUtils.irify(term)) if term and not pd.isna(term) else term
                            
                        terms = np.array([l(term) for term in terms], dtype=URIRef)
                        
                        
            
            else:
                return None
            
            
            PyRML.get_mapper().mappings[self] = terms
            
            return terms
                
            
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['TermObjectMap']:
        term_maps = []
        
        tt = g.value(parent, rml_vocab.RR_NS.termType)
        datatype = g.value(parent, rml_vocab.RR_NS.datatype)
        
        maps = [(m, Literal('reference')) for m in g.objects(parent, rml_vocab.RML_NS.reference, True)]
        maps += [(m, Literal('template')) for m in g.objects(parent, rml_vocab.RR_NS.template, True)]
        maps += [(m, Literal('constant')) for m in g.objects(parent, rml_vocab.RR_NS.constant, True)]
        maps += [(m, Literal('functionmap')) for m in g.objects(parent, rml_vocab.FNML_NS.functionValue, True)]
        
        for map in maps:
            term_object_map = parent
            
            language = LanguageBuilder.build(g, term_object_map)
            
            object_map = map[0]
            map_type=map[1]
            tom = TermObjectMap(term_object_map, object_map, map_type=map_type, term_type=tt, language=language, datatype=datatype)
            if map_type == Literal("functionmap"):
                tom._function_map = FunctionMap.from_rdf(g, object_map).pop()
            
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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            n_rows = data_source.data.shape[0]
            terms = np.array([self._constant for x in range(n_rows)])
            
            
            return terms
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        langs = g.objects(parent, rml_vocab.RR_NS.language, True)
        
        return [ConstantLanguage(language, parent) for language in langs]
        
    

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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        
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
                
            PyRML.get_mapper().mappings[self] = terms
            return terms    
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        
        term_maps = []
        
        language_maps = g.objects(parent, rml_vocab.RML_NS.languageMap, True)
        for language_map in language_maps:
            preds = [(rml_vocab.RML_NS.reference, Literal('reference')), (rml_vocab.RR_NS.template, Literal('template')), (rml_vocab.RR_NS.constant, Literal('constant'))]
            
            for pred in preds:
                l = g.value(language_map, pred[0])
                if l:
                    term_maps.append(LanguageMap(language_map, pred[0], pred[1]))
        
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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            n_rows = data_source.data.shape[0]
            terms = np.array([URIRef(self._constant) if self._constant else None for x in range(n_rows)], dtype=URIRef)
            
            PyRML.get_mapper().mappings[self] = terms 
            
            return terms
        
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        predicates = g.objects(parent, rml_vocab.RR_NS.predicate, True)
        
        return [ConstantPredicate(predicate, parent) for predicate in predicates]
        


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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            if self.predicate_expression_type == Literal("functionmap") and self.function_map:
                terms = self.function_map.apply(data_source, rdf_term_type=URIRef)
                
            else:
                if self.predicate_expression_type == Literal("template"):
                    terms = Expression.create(self.predicate_expression).eval_(data_source, True)
                elif self.predicate_expression_type == Literal("constant"):
                    n_rows = data_source.data.shape[0]
                    
                    terms = np.array([URIRef(self.predicate_expression) if self.predicate_expression else None for x in range(n_rows)], dtype=URIRef)
                    
                    
                elif self.predicate_expression_type == Literal("reference"):
                    
                    
                    index = data_source.columns[self.value.value]
                
                    data = data_source.data[index]
                
                    l = lambda val : URIRef(val) if val else None
                
                    terms = np.array([l(term) for term in data], dtype=URIRef)
                
        
            PyRML.get_mapper().mappings[self] = terms
        
            return terms
        
    
        
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> Set[TermMap]:
        term_maps = []
        
        predicate_maps = [(predicate_map, Literal('reference')) for predicate_map in g.objects(parent, rml_vocab.RML_NS.reference, True)]
        predicate_maps += [(predicate_map, Literal('template')) for predicate_map in g.objects(parent, rml_vocab.RR_NS.template, True)]
        predicate_maps += [(predicate_map, Literal('constant')) for predicate_map in g.objects(parent, rml_vocab.RR_NS.constant, True)]
        predicate_maps += [(predicate_map, Literal('functionmap')) for predicate_map in g.objects(parent, rml_vocab.FNML_NS.functionValue, True)]
        
        for p_map in predicate_maps:
            
            predicate_expression = p_map[0]
            predicate_expression_type = p_map[1]
            pm = PredicateMap(parent, predicate_expression=predicate_expression, predicate_expression_type=predicate_expression_type)
            
            if predicate_expression == Literal("functionmap"):
                pm.function_map = FunctionMap.from_rdf(g, predicate_expression).pop()
                
            term_maps.append(pm)
        
        return term_maps
    
    
class PredicateBuilder():
    
    @staticmethod
    def build(g: Graph, pom: IdentifiedNode) -> List[Predicate]:
        
        
        predicates = [ConstantPredicate(pred, pred) for pred in g.objects(pom, rml_vocab.RR_NS.predicate, True)]
        predicates += [PredicateMap.from_rdf(pred, pom) for pred in g.objects(pom, rml_vocab.RR_NS.predicateMap, True)]
        
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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        
        else:
            preds = [pm.apply(data_source) for pm in self._predicates]
            objs = [om.apply(data_source) for om in self.__object_maps]
            init = True
            preds_objs = None
            for predicates in preds:
                predicates = np.array(predicates).reshape(len(predicates), 1)
                
                for multi_objects in objs:
                    object_lens = [
                        len(obj)
                        if isinstance(obj, list) or isinstance(obj, np.ndarray)
                        else 1
                        for obj in multi_objects
                    ]
                    
                    if object_lens:
                        for obj_index in range(max(object_lens)):
                            objects = np.array([
                                (
                                    (
                                        obj[obj_index]
                                        if obj_index < len(obj)
                                        else None
                                    )
                                    if (
                                        isinstance(obj, list) or
                                        isinstance(obj, np.ndarray)
                                    ) else (
                                        obj
                                        if obj_index == 0
                                        else None
                                    ) 
                                )
                                for obj in multi_objects
                            ], dtype=object).reshape(-1, 1)
                                                
                            p_o = np.concatenate([
                                predicates, objects
                            ], axis=1)
                            
                            if init:
                                preds_objs = p_o
                                init = False
                            else:
                                preds_objs = np.concatenate([preds_objs, p_o], axis=0)
                                
                            
            #preds_objs = np.array([[pred, obj] for pred in predicates for obj in objects])
            
            
            
            #shp = preds_objs.shape
            
            #preds_objs.reshape(shp[1], shp[0])
            
            PyRML.get_mapper().mappings[self] = preds_objs
            
            return preds_objs
    
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None, parent_var: str = None) -> List['PredicateObjectMap']:
        
        term_maps = []
        tm = None
        pom = None
        if not parent_var or parent_var != 'pom':
            tm = parent
        else:
            pom = parent
        
        return [PredicateObjectMap.__build(g, pom) for tm, pred, pom in g.triples((tm, rml_vocab.RR_NS.predicateObjectMap, pom))]
        
        
    
    @staticmethod
    def __build(g, pom):
        
        predicates = PredicateBuilder.build(g, pom)
        
        object_maps = ObjectMapBuilder.build(g, pom)
        
        if predicates is not None and object_maps is not None:
            return PredicateObjectMap(pom, predicates=predicates, object_maps=object_maps)
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
        
        term_maps = []
        
        j_child = g.value(parent, rml_vocab.RR_NS.child, None)
        j_parent = g.value(parent, rml_vocab.RR_NS.parent, None)
        
        if j_child and j_parent:
            term_maps.append(Join(parent, child=j_child, parent=j_parent))
            
        return term_maps
        
    

class InputFormatNotSupportedError(Exception):
    def __init__(self, format):
        self.message = "The format %s is currently not supported"%(format)


class LogicalSource(AbstractMap):
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        self.__separator : str = kwargs['separator'] if 'separator' in kwargs else ',' 
        self.__query : str = kwargs['query'] if 'query' in kwargs else None
        self.__table_name : str = kwargs['table_name'] if 'table_name' in kwargs else None
        self.__reference_formulation : URIRef = kwargs['reference_formulation'] if 'reference_formulation' in kwargs else rml_vocab.CSV
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
    def table_name(self) -> str:
        return self.__table_name
    
    @property
    def iterator(self) -> URIRef:
        return self.__iterator
    
    @staticmethod
    def xml_namespaces(xml) -> Dict[str, str]: 
        events = "start", "start-ns"
        root = None
        ns = {}
        for event, elem in ET.iterparse(xml, events):
            if event == "start-ns":
                if elem[0] in ns and ns[elem[0]] != elem[1]:
                    # NOTE: It is perfectly valid to have the same prefix refer
                    #     to different URI namespaces in different parts of the
                    #     document. This exception serves as a reminder that this
                    #     solution is not robust.    Use at your own peril.
                    raise KeyError("Duplicate prefix with different URI found.")
                ns[elem[0]] = "{%s}" % elem[1]
            elif event == "start":
                if root is None:
                    root = elem
        return ns
    
    def apply(self, row: pd.Series = None) -> DataFrame:
        if self.id in PyRML.get_mapper().logical_sources:
            dfs = PyRML.get_mapper().logical_sources[self.id]
        else: 
            if self.__separator is None:
                sep = ','
            else:
                sep = self.__separator
            
            dfs = []
            for source in self.sources:
                if isinstance(source, BaseSource):
                    if self.__reference_formulation == rml_vocab.JSON_PATH and self.__iterator:
                        json_data = json.load(open(source._mapped_entity,mode='r',encoding='utf-8'))
                        
                        jsonpath_expr = parse(self.__iterator)
                        matches = jsonpath_expr.find(json_data)
                
                        data = [match.value for match in matches]
                        
                        df = pd.json_normalize(data)
                        
                    elif (self.__reference_formulation == rml_vocab.XML or self.__reference_formulation == rml_vocab.XPAPTH) and self.__iterator:
                        
                        _namespaces = LogicalSource.xml_namespaces(source._mapped_entity)
                        
                        df = pd.read_xml(source._mapped_entity, namespaces=_namespaces, xpath=self.__iterator, dtype=str)
                        
                    else:
                        df = pd.read_csv(source._mapped_entity, sep=sep, dtype=str)
                elif isinstance(source, CSVSource):
                    df = pd.read_csv(source.url, sep=source.delimiter, dtype=str)
                elif isinstance(source, SPARQLSource) and self.__query:
                    
                    sparql = SPARQLWrapper(source.endpoint)
                    sparql.setQuery(self.__query)
                    
                    if source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_CSV'):
                        sparql.setReturnFormat(CSV)
                    elif source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_TSV'):
                        sparql.setReturnFormat(TSV)
                    elif source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_XML'):
                        sparql.setReturnFormat(XML)
                    else:
                        sparql.setReturnFormat(JSON)
                    
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
                        
                    elif source.result_format == URIRef('http://www.w3.org/ns/formats/SPARQL_Results_XML') and self.__iterator:
                        df = pd.read_xml(BytesIO(rs), xpath=self.__iterator, dtype=str)
                        
                    else:
                        df = None
                elif isinstance(source, SQLSource) and source.valid() and (self.__query or self.__table_name):
                    
                    protocol_delimiter = source.dsn.find(':')
                    if protocol_delimiter >= 0:
                    
                        db_protocol = source.dsn[:protocol_delimiter]
                        db_server = source.dsn[protocol_delimiter+3:]
                        if db_protocol:
                            
                            #engine = sa.create_engine(f'{db_protocol}://{source.username}:{source.password}@?dsn={source.dsn}')
                            engine = sa.create_engine(f'{db_protocol}://{source.username}:{source.password}@{db_server}')
    
                            
                            query = self.__query if self.__query else f'SELECT * FROM {self.__table_name}'
                            try:
                                df = pd.read_sql(query, engine, parse_dates=['EntranceDate'])
                            except Exception as e:
                                print(e)
                                df = pd.DataFrame()
                            engine.dispose()
                            
                        else:
                            raise Exception(f'Database {source.driver} not supported.')
                    else:
                        raise Exception(f'No protocol found from DSN string {source.dsn}.')
                    
                else:
                    df = None
                
                
                #df.columns = df.columns.str.replace(r' ', '_')
                
                dfs.append(df)
                
                PyRML.get_mapper().logical_sources[self.id] = dfs
                
        return dfs
            

        
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List[TermMap]:
        term_maps = []
        
        rml = Namespace(rml_vocab.RML)
        rr = Namespace(rml_vocab.RR)
        crml = Namespace(rml_vocab.CRML)
        
        tps = g.triples((parent, rml.logicalSource|rr.logicalTable, None))
        for s,p,o in tps:
            ls = o
            rf = g.value(ls, rml.referenceFormulation, None)
            sep = g.value(ls, crml.separator, None)
            ite = g.value(ls, rml.iterator, None)
            query = g.value(ls, rml.query|rr.sqlQuery, None)
            table_name = g.value(ls, rr.tableName, None)
            
            sources = Source.from_rdf(g, ls)
            
            rf = rf if rf else rml_vocab.CSV
            
            ls = LogicalSource(ls, None, sources=sources, separator=sep, query=query, table_name= table_name, iterator=ite, reference_formulation=rf)
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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
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
        
        
            PyRML.get_mapper().mappings[self] = terms
            return terms 
        

    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['GrapMap']:
        
        rr = Namespace(rml_vocab.RR)
        rml = Namespace(rml_vocab.RML)
        
        gmps = [GraphMap(gm, g.value(gm, rr.constant, None), term_type=Literal('constant')) for gm in g.objects(parent, rr.graphMap)]
        
        gmps += [GraphMap(BNode(), graph, term_type=Literal('constant')) for graph in g.objects(parent, rr.graph)]
        
        for gm in g.objects(parent, rr.graphMap):
            v = g.value(gm, rr.template, None)
            if v:
                gmps.append(GraphMap(gm, v, term_type=Literal('template')))
            
            v = g.value(gm, rml.reference, None)
            if v:
                gmps.append(GraphMap(gm, v, term_type=Literal('reference')))
        
        return gmps
    
class SubjectMap(AbstractMap):
    def __init__(self, map_id: IdentifiedNode, value: Node = None, **kwargs):
        
        # , term_type: Literal, class_: Set[URIRef] = None, graph_map: GraphMap = None, map_id: URIRef = None
        super().__init__(map_id, value)
        self.__value = value
        self.__classes: List[URIRef] = kwargs['_classes'] if '_classes' in kwargs else None
        self.__graph_maps: List[GraphMap] = kwargs['graph_maps'] if 'graph_maps' in kwargs else None
        self.__term_type: Literal = kwargs['term_type'] if 'term_type' in kwargs else None
        self.__tt: IdentifiedNode = kwargs['tt'] if 'tt' in kwargs else rml_vocab.IRI
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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            
            if self.term_type == Literal("functionmap") and self.function_map:
                terms = self.function_map.apply(data_source)
            else:
                if self.term_type == Literal("template"):
                    terms  = Expression.create(self.value).eval_(data_source, True)
                   
                elif self.term_type == Literal("reference"):
                    
                    terms = data_source.dataframe[self.value.value]
                
                    #l = lambda val : URIRef(TermUtils.irify(val)) if val and isinstance(val, str) else None
                
                    #terms = np.array([l(term) for term in data], dtype=URIRef)
                     
                    
                elif self.term_type == Literal("constant"):
                    
                    n_rows = data_source.data.shape[0]
                    terms = [self.value for x in range(n_rows)]
                    
                    #l = lambda val : URIRef(TermUtils.irify(val)) if val else None
                    #terms = np.array([l(self.value.value) for x in range(n_rows)], dtype=URIRef)
            
            
            def l(term):
                
                if self.__tt == rml_vocab.BLANK_NODE:
                    tt = BNode
                else:
                    tt = URIRef
                
                if isinstance(term, list) or isinstance(term, np.ndarray):
                    return np.array([tt(TermUtils.irify(t)) if t and not pd.isna(t) and not isinstance(t, URIRef) else t for t in term], dtype=IdentifiedNode)
                else:
                    return tt(TermUtils.irify(term)) if term and not pd.isna(term) else term
                        
            terms = np.array([l(term) for term in terms], dtype=URIRef)
                
            PyRML.get_mapper().mappings[self] = terms
            return terms
    
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        
        term_maps = []
        
        preds = [(rml_vocab.RR_NS.template, Literal('template')),
                 (rml_vocab.RR_NS.constant, Literal('constant')),
                 (rml_vocab.RML_NS.reference, Literal('reference')),
                 (rml_vocab.FNML_NS.functionValue, Literal('functionmap'))]
        
        sbj_maps = g.objects(parent, rml_vocab.RR_NS.subjectMap, True)
        for sbj_map in sbj_maps:
            
            tt = g.value(sbj_map, rml_vocab.RR_NS.termType, None)
            
            for pred in preds:
                _map = g.value(sbj_map, pred[0], None)
                if _map:
                    d = {'map': _map,
                         'sm': sbj_map,
                         'termType': pred[1],
                         'tt': tt
                        }
                    obj = type('obj', (object,), d)
                    
                    term_maps.append(SubjectMap.__create(parent, g, obj))
        
        return term_maps
        
    
    @staticmethod
    def __create(_id: IdentifiedNode, graph: Graph, row):
        
        classes = graph.objects(row.sm, rml_vocab.CLASS, True)
        
        graph_maps = GraphMap.from_rdf(graph, row.sm)
        
        function_map = None   
        if row.termType == Literal("functionmap"):
            function_map = FunctionMap.from_rdf(graph, row.map)[0]
        
        return SubjectMap(_id, row.map, _classes=classes, graph_maps=graph_maps, term_type=row.termType, tt=row.tt, function_map=function_map)
        
    

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
                    if PyRML.has_registerd_function(self.__function_ref):
                        fun = PyRML.get_registerd_function(self.__function_ref)
                        
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
            
            
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
        
            function_ref = None
            
            #if 'rdf_term_type' in kwargs:
            
            pom_matrix : np.array = None
            
            pom_applied = [pom.apply(data_source) for pom in self._poms]
            try:
                pom_matrix = np.concatenate((pom_applied), axis=1)    
            except Exception as e:
                pass
            
            
            try:
                return np.array([Function(row).evaluate() for row in pom_matrix], dtype=Function)    
            except Exception as e:
                return 
            
             
            
            
            
     
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List['FunctionMap']:
        
        poms = [pom for pom_uri in g.objects(parent, rml_vocab.RR_NS.predicateObjectMap, True) for pom in PredicateObjectMap.from_rdf(g, pom_uri, 'pom')]
        
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
        self.__base : str = kwargs['base'] if 'base' in kwargs and kwargs['base'] else ''
        
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
    
    @property
    def base(self):
        return self.__base
    
    
    
    def apply(self, data_source: DataSource = None) -> np.array:
        if PyRML.RML_STRICT and (len(self.subject_maps) > 1 or len(self.logical_sources) > 1):
            raise RMLModelException(f'The RML descriptor declares a TripleMapping with {len(self.subject_maps)} subject maps. Exactly 1 subject map must be declared.')
        
        triples : DataFrame = pd.DataFrame(columns=['s', 'p', 'o'], dtype=object)
        for logical_source in self.logical_sources:
            
            for df in logical_source.apply():
                
                if self.condition:
                    df = df[eval(self.condition)]
                
                data_source = DataSource(df)
                
                sbj_maps = [subject_map.apply(data_source) for subject_map in self.subject_maps]
                
                graph_maps = [graph_map.apply(data_source) for subject_map in self.subject_maps for graph_map in subject_map.graph_maps]
                
                
                for sbj_representation in sbj_maps:
                    if self.predicate_object_maps is not None:
                            
                        for pom in self.predicate_object_maps:
                            try:
                                for object_map in pom.object_maps:
                                    
                                    df_join = None
                                    if isinstance(object_map, ReferencingObjectMap) and object_map.join_conditions:
                                        df_left = df.copy()
                                        df_left['__pyrml_sbj_representation__'] = sbj_representation
                                        
                                        parent_triple_mappings = object_map.parent_triples_maps
                                        
                                        
                                        for parent_triple_mapping in parent_triple_mappings:
                                            for parent_logical_source in parent_triple_mapping.logical_sources:
                                                
                                                for df_right in parent_logical_source.apply():
                                                    
                                                    df_tmp = df
                                                    df = df_right
                                                    pandas_condition = parent_triple_mapping.condition
                                                    if pandas_condition:
                                                        df = df[eval(pandas_condition)]
                                                        
                                                    join_conditions = object_map.join_conditions
                                                    
                                                    left_ons = []
                                                    right_ons = []
                                                    
                                                    for join_condition in join_conditions:
                                                        left_ons.append(join_condition.child.value)
                                                        right_ons.append(join_condition.parent.value)
                                                    
                                                    if not df_left.empty and not df.empty:
                                                        try: 
                                                            df_join = df_left.merge(df, how='inner', suffixes=("_s", "_r"), left_on=left_ons, right_on=right_ons, sort=False)
                                                        except ValueError:
                                                            df_join = pd.concat([df_left, df], axis=1, join='inner', sort=False)
                                                    
                                                        pom_representation = pom.apply(DataSource(df_join))
                                                        
                                                    df = df_tmp
                                                        
                                    else:
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
                                        
                                        if chunk_size > 0:
                                        
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
                            
                
        return triples.to_numpy(dtype=Node)  
    
         
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List[TermMap]:
        
        tm = None
        if parent:
            tm = g.value(parent, rml_vocab.RR_NS.parentTriplesMap)
        
        tps = g.triples((tm, rml_vocab.RML_NS.logicalSource|rml_vocab.RR_NS.logicalTable, None))
        
        triple_mappings = []
        for tm,p,o in tps:
            if g.value(tm, rml_vocab.RR_NS.subjectMap):
                triple_mappings.append(TripleMappings.__build(g, tm))
        return triple_mappings
        
    
    @staticmethod
    def __build(g: Graph, tm) -> 'TripleMappings':
        sources = LogicalSource.from_rdf(g, tm)
                        
        subject_maps: List[SubjectMap] = SubjectMap.from_rdf(g, tm)
        
        predicate_object_maps = PredicateObjectMap.from_rdf(g, tm)
        
        condition = g.value(tm, URIRef(rml_vocab.CRML+'condition'), None)
        
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
            
                     
        return TripleMappings(tm, None, sources=sources, subject_maps=subject_maps, predicate_object_maps=predicate_object_maps, condition=condition, base=g.base)
        
    
    
        
    
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
        
        if self in PyRML.get_mapper().mappings:
            return PyRML.get_mapper().mappings[self]
        else:
            
            if data_source:
                ref = np.array([subject_map.apply(data_source) for tm in self.parent_triples_maps for subject_map in tm.subject_maps], dtype=URIRef)
            else:
                ref = np.array([subject_map.apply(DataSource(source)) for tm in self.parent_triples_maps for subject_map in tm.subject_maps for logical_source in tm.logical_sources for source in logical_source.apply()], dtype=URIRef)
            ret = np.concatenate(ref, axis=0, dtype=URIRef)
            PyRML.get_mapper().mappings[self] = ret
            return ret
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode = None) -> List[TermMap]:
        term_maps = []
        
        join_conds = g.objects(parent, rml_vocab.RR_NS.joinCondition, True)
        
        joins: List[Join] = []
        
        for join_cond in join_conds:
            joins += Join.from_rdf(g, join_cond)
        
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
        term_maps = []
        sources = g.objects(parent, rml_vocab.RML_NS.source, True)
        
        for source in sources:
            
            sourcetype = None
            
            if isinstance(source, Literal):
                sourcetype = Literal('plain')
            elif g.value(source, rml_vocab.CSVW_NS.url):
                sourcetype = Literal('table')
            elif g.value(source, rml_vocab.SD_NS.endpoint):
                sourcetype = Literal('sparql')
            elif g.value(source, rml_vocab.D2RQ_NS.jdbcDSN):
                sourcetype = Literal('sql')
            else:
                db = g.value(None, RDF.type, rml_vocab.D2RQ_NS.Database, True)
                if db:
                    return [SQLSource.from_rdf(g, db)]
            
            if sourcetype:
                term_maps.append(Source.__build(g, source, sourcetype))
                
        return term_maps
        
        
    @staticmethod         
    def __build(g, source, sourcetype):
        
        sourcetype = sourcetype.value
        if sourcetype == 'plain':
            return BaseSource.from_rdf(g, source)
        elif sourcetype == 'table':
            return CSVSource.from_rdf(g, source)
        elif sourcetype == 'sparql':
            return SPARQLSource.from_rdf(g, source)
        elif sourcetype == 'sql':
            return SQLSource.from_rdf(g, source)
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
    def from_rdf(g: Graph, parent: IdentifiedNode) -> Source:
        
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
    def from_rdf(g: Graph, parent: IdentifiedNode) -> Source:
        
        sd = Namespace('http://www.w3.org/ns/sparql-service-description#')
        w3_formats = Namespace('http://www.w3.org/ns/formats/')
        
        endpoint = g.value(parent, sd.endpoint, None, True)
        if endpoint:
            
            supported_language = g.value(parent, sd.supportedLanguage, None, True)
            supported_language = supported_language if supported_language else sd.SPARQL11Query
                
            result_format = g.value(parent, sd.resultFormat, None, True)
            result_format = result_format if result_format else w3_formats.SPARQL_Results_JSON
                
            return SPARQLSource(parent, endpoint, endpoint=endpoint, supported_language=supported_language, result_format=result_format)
                
        else:
            return None
        
class SQLSource(Source):
    
    def __init__(self, map_id: IdentifiedNode, value: Node, **kwargs):
        super().__init__(map_id, value)
        
        self.__dsn : Literal = kwargs['dsn'] if 'dsn' in kwargs else None
        self.__driver : Literal = kwargs['driver'] if 'driver' in kwargs else None
        self.__usermane: Literal = kwargs['username'] if 'username' in kwargs else None
        self.__password: Literal = kwargs['password'] if 'password' in kwargs else None
        self.__result_size_limit = kwargs['result_size_limit'] if 'result_size_limit' in kwargs else None
        self.__fetch_size = kwargs['fetch_size'] if 'fetch_size' in kwargs else None
        
    @property
    def dsn(self):
        return self.__dsn
    
    @property
    def driver(self):
        return self.__driver
    
    @property
    def username(self):
        return self.__usermane
    
    @property
    def password(self):
        return self.__password
    
    @property
    def result_size_limit(self):
        return self.__result_size_limit
    
    @property
    def fetch_size(self):
        return self.__fetch_size
    
    def valid(self):
        
        return True if self.__dsn and self.__usermane and self.__password else False
        
    @staticmethod
    def from_rdf(g: Graph, parent: IdentifiedNode) -> Source:
        
        d2rq = Namespace('http://www.wiwiss.fu-berlin.de/suhl/bizer/D2RQ/0.1#')
        
        dsn = g.value(parent, d2rq.jdbcDSN, None, True)
        username = g.value(parent, d2rq.username, None, True)
        password = g.value(parent, d2rq.password, None, True)
        if dsn and username and password:
            
            driver = g.value(parent, d2rq.jdbcDriver, None, True)
            result_size_limit = g.value(parent, d2rq.resultSizeLimit, None, True)
            fetch_size = g.value(parent, d2rq.fetchSize, None, True)
                
            return SQLSource(parent, dsn, dsn=dsn, driver=driver, username=username, password=password, result_size_limit=result_size_limit, fetch_size=fetch_size)
                
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
            #pass
            print(e)
                
            #g += tm.apply()
    
        
