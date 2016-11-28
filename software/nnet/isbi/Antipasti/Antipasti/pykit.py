__author__ = 'nasimrahaman'

__doc__ = """ Functions to help with Python """

import itertools as it
import random
import numpy as np

# Python's equivalent of MATLAB's unique (legacy)
def unique(items):
    """
    Python's equivalent of MATLAB's unique (legacy)
    :type items: list
    :param items: List to operate on
    :return: list
    """
    found = set([])
    keep = []

    for item in items:
        if item not in found:
            found.add(item)
            keep.append(item)

    return keep


# Add two lists elementwise
def addelems(list1, list2):
    """
    Adds list1 and list2 element wise. Summands may contain None, which are ignored.
    :type list1: list
    :param list1: First summand
    :type list2: list
    :param list2: Second summand
    :return: list
    """
    # Make sure the lists are of the same length
    assert len(list1) == len(list2), "Summands must have the same length."

    # Add
    return [(item1 + item2 if not (item1 is None or item2 is None) else None) for item1, item2 in zip(list1, list2)]


# Convert a tuple or a non iterable to a list, simultaneously
def obj2list(obj):
    # Try-except clause may not work here because layertrain is an iterator and can be converted to list
    if isinstance(obj, (list, tuple, np.ndarray)):
        return list(obj)
    else:
        return [obj]


# Try to convert an object to int
def try2int(obj):
    try:
        return int(obj)
    except:
        return obj


# Convert a list of one element to element
def delist(l):
    if len(l) == 1 and isinstance(l, (list, tuple)):
        return l[0]
    else:
        return l


# Smart len function that doesn't break when input is not a list/tuple
def smartlen(l):
    if isinstance(l, (list, tuple)):
        return len(l)
    else:
        return 1


# Function to remove singleton sublists
def removesingletonsublists(l):
    return [elem[0] if isinstance(elem, (list, tuple)) and len(elem) == 1 else elem for elem in l]


# Function to convert a list to a list of list if it isn't one already, i.e. [l] --> [[l]] but [[l]] = [[l]].
def list2listoflists(l):
    if islistoflists(l):
        return l
    else:
        return [l]


# Function to chain lists (concatenate lists in a list of lists)
def chain(l):
    return list(it.chain.from_iterable(l))


# Function to flatten a list of list (of list of list of li...) to a list
flatten = lambda *args: (result for mid in args for result in (flatten(*mid)
                                                               if isinstance(mid, (tuple, list)) else (mid,)))


def delistlistoflists(l):
    newlist = []
    for elem in l:
        if not islistoflists(elem):
            newlist.append(elem)
        else:
            newlist += elem
    return newlist


def islistoflists(l):
    return all([isinstance(elem, (list, tuple))for elem in l])


# Function to update a list (list1) with another list (list2) (similar to dict.update, but with lists)
def updatelist(list1, list2):
    return list1 + [elem for elem in list2 if elem not in list1]


def updatedictlist(list1, list2):
    dict1 = dict(list1)
    dict1.update(dict(list2))
    return dict1.items()


# Function to migrate attributes from one instance of a class to another. This was written to be used for weight
# sharing.
def migrateattributes(source, target, attributes):
    """
    Function to migrate attributes from one instance (source) of a class to another (target). This function does no
    checks, so please don't act like 10 year olds with chainsaws.

    :type source: object
    :param source: Source object

    :type target: object
    :param target: Target object

    :type attributes: list of str or tuple of str
    :param attributes: List/tuple of attribute names.
    """
    for attribute in attributes:
        target.__setattr__(attribute, source.__getattribute__(attribute))

    return target


# TODO Test
# Shuffle a python generator
def shufflegenerator(gen, buffersize=20, rngseed=None):
    # Seed RNG
    if rngseed is not None:
        random.seed(rngseed)

    # Flag to check if generator is exhausted
    genexhausted = False
    # Buffer
    buf = []
    while True:
        # Check if stopping condition is fulfilled
        if genexhausted and len(buf) == 0:
            raise StopIteration

        # Fill up buffer if generator is not exhausted
        if not genexhausted:
            for _ in range(0, buffersize - len(buf)):
                try:
                    buf.append(gen.next())
                except StopIteration:
                    genexhausted = True

        # Pop a random element from buffer random number of times
        numpops = random.randint(0, len(buf))
        for _ in range(numpops):
            popindex = random.randint(0, len(buf) - 1)
            yield buf.pop(popindex)

if __name__ == "__main__":
    print(list(shufflegenerator(iter('abcdefghijk'), buffersize=3, rngseed=42)))
    pass