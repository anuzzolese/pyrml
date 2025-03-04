__author__ = "Andrea Giovanni Nuzzolese"
__email__ = "andrea.nuzzolese@cnr.it"
__license__ = "Apache 2"
__version__ = "0.2.9"
__status__ = "Alpha"

import logging
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import os
import time
from typing import Dict, Generator, Union, List

from jinja2 import Environment, FileSystemLoader
from pyrml.pyrml_api import Mapper, MappingsDict, graph_add_all
from pyrml.pyrml_core import TripleMappings, \
    TripleMapping, LogicalSource
from rdflib import Graph, Namespace, plugin, ConjunctiveGraph, URIRef
from rdflib.term import Node, IdentifiedNode, BNode
from rdflib.parser import StringInputSource
from rdflib.query import Processor, Result

import numpy as np
import pandas as pd
import pyrml.rml_vocab as rml_vocab


class RMLParser():
    
    @staticmethod
    def parse(source, format="ttl"):
        
        g = Graph()
        
        g.bind('csvw', Namespace(rml_vocab.CSVW))
        g.bind('rr', Namespace(rml_vocab.RR))
        g.bind('rml', Namespace(rml_vocab.RML))
        g.bind('ql', Namespace(rml_vocab.QL))
        g.bind('crml', Namespace(rml_vocab.CRML))
        g.bind('fnml', Namespace(rml_vocab.FNML))
        g.bind('fno', Namespace(rml_vocab.FNO))
        
        
        g.parse(source, format=format)
        
        
        return TripleMappings.from_rdf(g)
        
        '''
        g_2 = Graph()
        for tm in triple_mappings:
            g_2 += tm.apply()
            g1 = tm.to_rdf()
            for l in g1.serialize(format=format).splitlines():
                if l: print(l.decode('ascii'))
        '''
    
class RMLConverter(Mapper):
    
    #__instance = None
    
    def __init__(self):
        self.__function_registry = dict()
        self.__mapping_dict = MappingsDict()
        self.__loaded_logical_sources = dict()
        self.__mappings = dict()
        
        self.subject_map_representations = dict()
        
#        global __instance
#        __instance = self
        
    def reset(self):
        del(self.__function_registry)
        del(self.__mapping_dict)
        del(self.__loaded_logical_sources)
        del(self.__mappings)
        del(self.subject_map_representations)
        
        self.__function_registry = dict()
        self.__mapping_dict = MappingsDict()
        self.__loaded_logical_sources = dict()
        self.__mappings = dict()
        self.subject_map_representations = dict()
        
        
    @property
    def mappings(self):
        return self.__mappings
        
    @property
    def logical_sources(self):
        return self.__loaded_logical_sources
    
    
    @property
    def function_registry(self):
        return self.__function_registry
    
    '''    
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = RMLConverter()
        
        return cls.__instance
       
    @staticmethod
    def set_instance(instance):
        #if RMLConverter.__instance is None:
        #    RMLConverter.__instance = RMLConverter()
            
        RMLConverter.__instance = instance
    '''    
    
    def convert(self, rml_mapping, multiprocessed=False, base=None, template_vars: Dict[str, str] = None) -> Graph:
    
        plugin.register("sparql", Result, "rdflib.plugins.sparql.processor", "SPARQLResult")
        plugin.register("sparql", Processor, "rdflib.plugins.sparql.processor", "SPARQLProcessor")
        
        if template_vars is not None:
            
            
            if os.path.isabs(rml_mapping):
                templates_searchpath = "/"
            else:
                templates_searchpath = "."
            file_loader = FileSystemLoader(templates_searchpath)
            env = Environment(loader=file_loader)
            template = env.get_template(rml_mapping)
            rml_mapping_template = template.render(template_vars)
            
            rml_mapping = StringInputSource(rml_mapping_template.encode('utf-8'))
        
        triple_mappings = RMLParser.parse(rml_mapping)
        
        if base:
            g = ConjunctiveGraph(default_graph_base=base)
        else:
            g = ConjunctiveGraph()
        
        
        print(f'The RML mapping contains {len(triple_mappings)} triple mappings.')
        start_time = time.time()
        if multiprocessed:
            start_time = time.time()
            processes = cpu_count()
        
            tms = np.array_split(np.array(list(triple_mappings)), processes)
            pool = ThreadPool(initializer=initializer, initargs=(RMLConverter.__instance,), processes=processes)
            tuples_collection = pool.map(pool_map, tms)
            pool.close()
            pool.join()
        
        
            for tuples in tuples_collection:
                for _tuple in tuples:
                    g.add(_tuple)
        
        else:
            for tm in triple_mappings:
                for _tuple in tm.apply():
                    
                    def normalize_iri(iri):
                        if isinstance(iri, BNode):
                            return iri
                        elif isinstance(iri, URIRef):
                            if str(iri).find(':') > 0:
                                return iri
                        
                        return URIRef(tm.base + str(iri))

                        
                    
                    #print(f'TUPLE {_tuple}')
                    _sub = self.__generate(_tuple[0])
                    _pred = self.__generate(_tuple[1])
                    _obj = self.__generate(_tuple[2])
                    _graph = self.__generate(_tuple[3]) if len(_tuple) == 4 else None
                    
                    _sub = np.array([_sub], dtype=IdentifiedNode) if not isinstance(_sub, np.ndarray) else _sub
                    _pred = np.array([_pred], dtype=URIRef) if not isinstance(_pred, np.ndarray) else _pred
                    _obj = np.array([_obj], dtype=Node) if not isinstance(_obj, np.ndarray) else _obj
                    
                    if _graph:
                        _graph = np.array([_graph], dtype=Node) if not isinstance(_graph, np.ndarray) else _graph
                        _graph = [URIRef(_g) if not isinstance(_g, URIRef) else _g for _g in _graph]
                    
                    #_sub = [URIRef(_s) if not isinstance(_s, IdentifiedNode) else _s for _s in _sub]
                    #_pred = [URIRef(_p) if not isinstance(_p, URIRef) else _p for _p in _pred]
                    _sub = [normalize_iri(_s) for _s in _sub]
                    _pred = [normalize_iri(_p) for _p in _pred]
                    
                    
                    
                    
                    try:
                        for _s in _sub:
                            for _p in _pred:
                                for _o in _obj:
                                    if _s and _p and _o:
                                        if _graph:
                                            for _g in _graph:
                                                g.add((_s, _p, _o, _g))
                                        else:
                                            g.add((_s, _p, _o))
                    except Exception as e:
                        print(f'{_sub}, {_pred}, {_obj}')
                        print(f'{_sub} as type {type(_sub)}')
                        print(type(_obj))
                        print(_tuple)
                        raise e
            
        elapsed_time_secs = time.time() - start_time
        print(f'Mapping computed in {elapsed_time_secs} secs producing {len(g)} triples.')
        return g
    
    
    def convert_(self, rml_mapping, multiprocessed=False, template_vars: Dict[str, str] = None) -> Graph:
    
        plugin.register("sparql", Result, "rdflib.plugins.sparql.processor", "SPARQLResult")
        plugin.register("sparql", Processor, "rdflib.plugins.sparql.processor", "SPARQLProcessor")
        
        if template_vars is not None:
            
            
            if os.path.isabs(rml_mapping):
                templates_searchpath = "/"
            else:
                templates_searchpath = "."
            file_loader = FileSystemLoader(templates_searchpath)
            env = Environment(loader=file_loader)
            template = env.get_template(rml_mapping)
            rml_mapping_template = template.render(template_vars)
            rml_mapping = StringInputSource(rml_mapping_template.encode('utf-8'))
        
        triple_mappings = RMLParser.parse(rml_mapping)
        
        g = ConjunctiveGraph()
        
        if multiprocessed:
            processes = cpu_count()
        
            tms = np.array_split(np.array(list(triple_mappings)), processes)
            pool = ThreadPool(initializer=initializer, initargs=(RMLConverter.__instance,), processes=processes)
            tuples_collection = pool.map(pool_map, tms)
            pool.close()
            pool.join()
        
        
            for tuples in tuples_collection:
                for _tuple in tuples:
                    g.add(_tuple)
        
        else:
            print(f'The RML mapping contains {len(triple_mappings)} triple mappings.')
            start_time = time.time()
            
            for tm in triple_mappings:
                for _tuple in tm.apply():
                    try:
                        g.add(tuple(_tuple))
                    except Exception as e:
                        print(_tuple)
                        raise e
            
        elapsed_time_secs = time.time() - start_time
        print(f'Mapping computed in {elapsed_time_secs} secs')
        return g
    
    
    def __itertuples(self, triple_maps: List[TripleMapping], df: pd.DataFrame, g: Graph):
        
        
        
        rows = [row for row in df.itertuples()]
        print(f'Applying {len(triple_maps)} to {len(rows)} rows.')
        for row in rows:
            for triple_map in triple_maps:
                for _tuple in triple_map.apply(row):
                    _sub = self.__generate(_tuple[0])
                    _pred = self.__generate(_tuple[1])
                    _obj = self.__generate(_tuple[2])
                    
                    if _sub and _pred and _obj: 
                        if len(_tuple) == 4:
                            _ctx = self.__generate(_tuple[3])
                            _t = (_sub, _pred, _obj, _ctx)
                        else:
                            _t = (_sub, _pred, _obj)
                        
                        try:
                            g.add(_t)
                        except Exception as e:
                            print(f'{_sub}, {_pred}, {_obj}')
                            print(f'{_sub} as type {type(_sub)}')
                            print(type(_obj))
                            raise e
                    
    def __itertuples_(self, row, triple_maps: List[TripleMapping], g: Graph):
        for triple_map in triple_maps:
            for _tuple in triple_map.apply(row):
                _sub = self.__generate(_tuple[0])
                _pred = self.__generate(_tuple[1])
                _obj = self.__generate(_tuple[2]) 
                if len(_tuple) == 4:
                    _ctx = self.__generate(_tuple[3])
                    _t = (_sub, _pred, _obj, _ctx)
                else:
                    _t = (_sub, _pred, _obj)
                
                try:
                    g.add(_t)
                except Exception as e:
                    print(f'{_sub}, {_pred}, {_obj}')
                    print(f'{_sub} as type {type(_sub)}')
                    print(type(_obj))
                    raise e
    
    
    def __generate(self, value):
        '''
        while value and isinstance(value, Generator):
            value = next(value, None)
        ''' 
        return value
    
    def get_mapping_dict(self):
        return self.__mapping_dict
    
    def get_loaded_logical_sources(self):
        return self.__loaded_logical_sources
    
    def get_function_registry(self):
        return self.__function_registry
    
class LogicalSourceIndex():
    
    def __init__(self, ls: LogicalSource, triple_maps: List[TripleMapping] = None, parent_ls: LogicalSource = None, left_joins:List[str]=None, right_joins:List[str]=None):
        self.__ls = ls
        self.__triple_maps = triple_maps if triple_maps else []
        self.__parent_ls = parent_ls
        self.__left_joins = left_joins
        self.__right_joins = right_joins 
        
    @property
    def logical_source(self):
        return self.__ls
    
    @property
    def triple_maps(self):
        return self.__triple_maps
    
    @property
    def parent_logical_source(self):
        return self.__parent_ls
    
    @property
    def left_joins(self):
        return self.__left_joins
    
    @property
    def rigth_joins(self):
        return self.__right_joins
    
    def add_triple_map(self, tm: TripleMapping):
        self.__triple_maps.append(tm)
    
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
