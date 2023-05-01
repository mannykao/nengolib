import numpy as np

__all__ = ['get_activities']


def get_activities(model, ens, eval_points):
    """nengo.builder.ensemble.get_activities from < 2.3.0."""
    x = np.dot(eval_points, model.params[ens].encoders.T / ens.radius)
    return ens.neuron_type.rates(
        x, model.params[ens].gain, model.params[ens].bias)

def with_metaclass(meta, *bases):   #mck: copied from nengo.utils.compat of 2.8.0
    """Function for creating a class with a metaclass.

    The syntax for this changed between Python 2 and 3.
    Code snippet from Armin Ronacher:
    http://lucumr.pocoo.org/2013/5/21/porting-to-python-3-redux/
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})
