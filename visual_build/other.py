from map.shapefile import build_shapefile
from map.visualize import visualize

def dim_func(put_do_papki):
    #здесь твоя функция
    pass

def den_func(put_do_papki):
    #получить по .npy шейпфайл
    build_shapefile(put_do_papki, output_filename="paths.shp")
    visualize("paths.shp", "map.html")
