from collections.abc import Sequence

from sympy import diff

from devito.tools import GenericVisitor, flatten
from devito.symbolics import retrieve_function_carriers, indexify
from devito.types import Eq

class ExpressionSet(Sequence):
    def __init__(self):
        self._generated_expressions = None
        self._generated = False

    @property
    def collection(self):
        if self._generated_expressions is not None:
            return self._generated_expressions
        self._generated_expressions = self.generate_expressions()
        self._generated = True
        return self._generated_expressions

    def __getitem__(self, key):
        return self.collection[key]

    def __len__(self):
        return len(self.collection)


class Adjoint(ExpressionSet):
    def __init__(self, expressions, substitutions=None, ignores=None):
        if substitutions is None:
            substitutions = {}
        if ignores is None:
            ignores = []
        self.expressions = expressions
        self.substitutions = substitutions
        self.ignores = ignores
        super().__init__()

    def generate_expressions(self):
        return Differentiator(self.substitutions, self.ignores).visit(self.expressions)


class Differentiator(GenericVisitor):
    def __init__(self, substitutions, ignores):
        self._substitutions = substitutions
        self._ignores = ignores
        super().__init__()

    def visit_object(self, node):
        raise ValueError("Can not differentiate %s. Need a substituting Function" % str(node))
        
    def visit_list(self, nodes):
        return flatten([self.visit(x) for x in nodes])

    def visit_Indexed(self, node):
        return self.visit(node.function).indexed[node.indices]

    def visit_Function(self, node):
        if (node.function in self._substitutions.keys()):
            return self._substitutions[node.function]
        elif node.function in self._ignores:
            return node.function
        else:
            raise ValueError("Can not differentiate %s. Need a substituting Function" % str(node))

    def visit_Eq(self, e):
        e = indexify(e)  # sympy.diff() seems to not like u() like functions. wants u[] instead
        indexeds = retrieve_function_carriers(e.rhs, mode='all')
        adjoint_lhs = self.visit(e.lhs)
        
        differentiated_expressions = []
        for i in indexeds:
            if i.function in self._ignores:
                continue
            i_d = self.visit(i)
            differentiated_expressions.append(Eq(i_d, diff(e.rhs, i) * adjoint_lhs))
        return differentiated_expressions
            
            
        
        
