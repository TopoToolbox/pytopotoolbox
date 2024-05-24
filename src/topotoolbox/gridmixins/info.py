"""This module contains the Mixin class InfoMixin for the GridObject. 
"""


class InfoMixin():
    """This class is a Mixin for the GridObject class.
    It contains the info() function.
    """

    def info(self):
        """Prints all variables of a GridObject.
        """
        print("path: "+str(self.path))
        print("rows: "+str(self.rows))
        print("cols: "+str(self.columns))
        print("cellsize: "+str(self.cellsize))
