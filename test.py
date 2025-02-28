import topotoolbox as tt3
dem = tt3.load_dem('perfectworld')
p, index = dem.prominence(100.0)
