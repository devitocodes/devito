class NamedObject(object):
    def getName(self):
        return self.__class__.__name__
