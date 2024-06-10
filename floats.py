from numbers import Number
from math import sqrt


class MyFloat:
    def __init__(self, n: Number) -> None:
        self.n = n

    def __add__(self, other):
        return MyFloat(other.n + self.n)

    def __sub__(self, other):
        return MyFloat(self.n - other.n)

    def __mul__(self, other):
        return MyFloat(self.n * other.n)

    def __div__(self, other):
        return MyFloat(self.n/other.n)

    def __abs__(self):
        return sqrt(self.n**2)

    def __neg__(self):
        return MyFloat(-self.n)

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '%g' % (self.n)

    def __repr__(self):
        return 'MyFloat' + str(self)

    def __pow__(self, power):
        return MyFloat(self.n ** power)