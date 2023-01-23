from collections import namedtuple


class CanonicalEntity(namedtuple('CanonicalEntity', ['value', 'type'])):
    __slots__ = ()

    def __str__(self):
        return '[%s]' % str(self.value)


class Entity(namedtuple('Entity', ['surface', 'canonical'])):
    __slots__ = ()

    @classmethod
    def from_elements(cls, surface=None, value=None, type=None):
        if value is None:
            value = surface
        return super(cls, Entity).__new__(cls, surface, CanonicalEntity(value, type))

    def __str__(self):
        return '[%s, %s]' % (str(self.surface), str(self.canonical.value))


def is_entity(x):
    return isinstance(x, Entity) or isinstance(x, CanonicalEntity)


