#!/usr/bin/env python


class Triple:
    s = None
    p = None
    o = None

    def __init__(self, s, p, o):
        self.s = s
        self.p = p
        self.o = o

    def ntriple(self):
        t = ""
        if type(self.s) is URIRef:
            t += '<' + self.s.URI + "> "
        else:  # BNode
            t += "_:" + self.s.identifier + ' '

        t += '<' + self.p.URI + "> "

        if type(self.o) is URIRef:
            t += '<' + self.o.URI + "> "
        else:
            t += '"' + self.o.value + '"'
            if self.o.dtype is not None:
                t += "^^<" + self.o.dtype + "> "
            elif self.o.lang is not None:
                t += '@' + self.o.lang

        return t + '.'

    def __str__(self):
        return '(' + ', '.join([str(self.s),
                                str(self.p),
                                str(self.o)]) + ');'

class URIRef:
    URI = None

    def __init__(self, URI):
        self.URI = URI

    def __str__(self):
        return self.URI

class Literal:
    value = None
    dtype = None
    lang = None

    def __init__(self, value, dtype=None, lang=None):
        self.value = value
        self.dtype = dtype
        self.lang = lang

    def __str__(self):
        suffix = None
        if self.dtype is not None:
            suffix = ' [' + self.dtype + ']'
        elif self.lang is not None:
            suffix = ' (' + self.lang + ')'

        return repr(self.value) if suffix is None\
               else repr(self.value) + suffix

class BNode:
    identifier = None

    def __init__(self, identifier):
        self.identifier = identifier

    def __str__(self):
        return self.identifier
