from devito.dle.backends import BasicRewriter, dle_pass


class YaskRewriter(BasicRewriter):

    def _pipeline(self, state):
        super(YaskRewriter, self)._pipeline(state)
