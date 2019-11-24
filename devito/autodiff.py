from collections.abc import Sequence

from devito.tools import visitor


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
    def __init__(self, expressions, substitutions):
        self.original_expressions = expressions
        self.substitutions = substitutions

    def generate_expressions(self):
        return Differentiator(self.substitutions).visit(self.expressions)


class Differentiator(GenericVisitor):
    def __init__(self, substitutions):
        self._substitutions = substitutions
        super(GenericVisitor, self).__init__()
        
    def visit_Sequence(self, nodes):
        return [self.visit(x) for x in nodes]

    def visit_Indexed(self, node):
        return self.visit(node.function)[node.indices]

    def visit_Function(self, node):
        assert(node in self.substitutions.keys())
    
        return self.substitutions[node]

    def visit_Eq(self, e):
        indexeds = retrieve_indexed(e.rhs, mode='all', deep=True)
        adjoint_lhs = self.visit(e.lhs)

        differentiated_expressions = []
        for i in indexeds:
            i_d = self.visit(i)
            differentiated_expressions.append(Eq(i_d, diff(e.rhs, i) * adjoint_lhs))
        return differentiated_expressions
            
            
        
        
