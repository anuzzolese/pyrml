from rdflib.plugins.parsers.notation3 import TurtleParser, RDFSink, SinkParser
from rdflib.exceptions import ParserError
from rdflib import Graph

from typing import Optional

class MyTurtleParser(TurtleParser):

    def parse(
        self,
        source: "InputSource",
        graph: Graph,
        encoding: Optional[str] = "utf-8",
        turtle: bool = True,
    ):
        if encoding not in [None, "utf-8"]:
            raise ParserError(
                "N3/Turtle files are always utf-8 encoded, I was passed: %s" % encoding
            )

        sink = RDFSink(graph)

        baseURI = graph.absolutize(source.getPublicId() or source.getSystemId() or "")
        p = SinkParser(sink, baseURI=baseURI, turtle=turtle)
        # N3 parser prefers str stream
        stream = source.getCharacterStream()
        if not stream:
            stream = source.getByteStream()
        p.loadStream(stream)

        for prefix, namespace in p._bindings.items():
            graph.bind(prefix, namespace)
            
        if p._baseURI:
            graph.base = p._baseURI