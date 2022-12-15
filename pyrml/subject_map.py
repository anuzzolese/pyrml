from api import *
from rdflib import Literal, URIRef
from pandas.core.frame import DataFrame

from rdflib.plugins.sparql.processor import prepareQuery,

import numpy as np

import rml_vocab

class SubjectMapImpl(SubjectMap):
    def __init__(self, mapped_entity: Node, term_type: Literal, class_: Set[URIRef] = None, graph_map: GraphMap = None, map_id: URIRef = None):
        super().__init__(mapped_entity, term_type, class_, graph_map, map_id)
    
    def to_rdf(self):
        g = Graph()
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
        
        start_time = time.time()

        #l = lambda x: TermUtils.urify(self._mapped_entity, x)
        l = lambda x: self.__convert(x)
        
        #l = lambda x: print(x['id'])  
            
        
        df_1 = df.apply(l, axis=1)
        df_1.replace('', np.nan, inplace=True)
        df_1.dropna(inplace=True)
        
        elapsed_time_secs = time.time() - start_time
        
        msg = "Subject Map: %s secs" % elapsed_time_secs
        print(msg)  
        return df_1
    
    def apply_(self, row):
        
        term = None
        if self.__term_type == Literal("template") or self.__term_type == Literal("constant"):
            #term = TermUtils.eval_template(self._expression, row, True)
            term = self._expression.eval(row, True)
        elif self.__term_type == Literal("reference"):
            term = URIRef(row[self._mapped_entity.value])
        
        
        if self.__graph_map:
            out = (self.__graph_map.apply_(row), term)
        else:
            out = (None, term)
            
        return out
        
    
    @staticmethod
    def from_rdf(g: Graph, parent: Union[BNode, URIRef] = None) -> Set[TermMap]:
        mappings_dict = RMLConverter.get_instance().get_mapping_dict()
            
        sparql = """
            SELECT DISTINCT ?sm ?map ?termType ?type ?gm
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
                    { ?sm rr:graph ?gm }
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
                    subject_map = SubjectMap.__create(g, row)
                    subject_maps.update({row.sm.n3(): subject_map})
                    if isinstance(row.sm, URIRef):
                        mappings_dict.add(subject_map)
                        
            else:
                subject_map.__class.add(row.type)
                
        
        return set(subject_maps.values())
    
    @staticmethod
    def __create(graph: Graph, row):
        
        graph_map = None
        if row.gm:
            graph_map = GraphMap.from_rdf(graph, row.sm).pop()
                
        return SubjectMap(row.map, row.termType, {row.type}, graph_map, row.sm)