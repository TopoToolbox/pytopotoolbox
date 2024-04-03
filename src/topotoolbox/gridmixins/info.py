class InfoMixin():
    def info(self):
        print("rows: " + str(self.rows))
        print("cols: " + str(self.columns))
        print("cellsize: " + str(self.cellsize))