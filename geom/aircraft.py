from part import Part


__all__ = ['Aircraft']


class Aircraft(Part, dict):

    ''' An Aircraft is a customized Python dict that acts as root for
    all the Parts it is composed of.

    Intended usage
    --------------
    >>> ac = Aircraft(fuselage=fus, wing=wi)
    >>> ac.items()
    {'wing': <geom.wing.Wing at <hex(id(wi))>,
     'fuselage': <geom.fuselage.Fuselage at <hex(id(fus))>}
    >>> del ac['wing']
    >>> ac['tail'] = tail

    '''

    def __init__(self, *args, **kwargs):

        ''' Form an Aircraft from Part components.

        Parameters
        ----------
        args, kwargs = the Parts to constitute the Aircraft with

        '''

        dict.__init__(self, *args, **kwargs)
        if not all([isinstance(p, Part) for p in self.values()]):
            raise NonPartComponentDetected()

    def __repr__(self):
        ''' Override the object's internal representation. '''
        return ('<{}.{} at {}>'
                .format(self.__module__,
                        self.__class__.__name__,
                        hex(id(self))))

    def __setitem__(self, k, v):
        ''' Make sure v is a Part. '''
        if not isinstance(v, Part):
            raise NonPartComponentDetected()
        super(Aircraft, self).__setitem__(k, v)

    def __delitem__(self, k):
        ''' Unglue the Part before deleting it. '''
        self[k].unglue()
        super(Aircraft, self).__delitem__(k)

    def blowup(self):
        ''' Blow up the Aircraft's Parts in the current namespace. '''
        IP = get_ipython()
        for k, v in self.items():
            IP.user_ns[k] = v
        return self.items()

    def _glue(self):
        ''' See Part._glue. '''
        return [o for P in self.values() for o in P._glue(self)]

    def _draw(self):
        ''' See Part._draw. '''
        if hasattr(self, 'draw'):
            return self.draw
        return [P for P in self.values()]


# EXCEPTIONS


class AircraftException(Exception):
    pass

class NonPartComponentDetected(AircraftException):
    pass
