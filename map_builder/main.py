from shapefile import build_shapefile
from visualize import visualize
import timeit


def main():
    start = timeit.default_timer()

    build_shapefile(r'D:\data\dl_masks8', output_filename="output/paths.shp")

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
    # visualize("output/paths.shp", "map.html")


if __name__ == "__main__":
    main()
