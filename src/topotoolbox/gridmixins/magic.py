import numpy as np
import copy


class MagicMixin():
    def __eq__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] == other.z[x][y]

        return dem

    def __ne__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] != other.z[x][y]

        return dem

    def __gt__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] > other.z[x][y]

        return dem

    def __lt__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] < other.z[x][y]

        return dem

    def __ge__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] >= other.z[x][y]

        return dem

    def __le__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] <= other.z[x][y]

        return dem

    def __add__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z + other.z
            return dem

        else:
            dem.z = self.z + other
            return dem

    def __sub__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z - other.z
            return dem

        else:
            dem.z = self.z - other
            return dem

    def __mul__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z * other.z
            return dem

        else:
            dem.z = self.z * other
            return dem

    def __div__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z / other.z
            return dem

        else:
            dem.z = self.z / other
            return dem

    def __and__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'and' can only compare True (1) and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) & int(other.z[x][y]))

        return dem

    def __or__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'or' can only compare True (1) and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) | int(other.z[x][y]))

        return dem

    def __xor__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'xor' can only compare True (1) and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) ^ int(other.z[x][y]))

        return dem

    def __len__(self):
        return len(self.z)

    def __iter__(self):
        return iter(self.z)

    def __getitem__(self, index):
        return self.z[index]

    def __setitem__(self, index, value):
        try:
            value = np.float32(value)
        except:
            raise TypeError(
                value, " not can't be converted to float32.") from None

        self.z[index] = value

    def __array__(self):
        return self.z
