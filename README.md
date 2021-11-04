# Lawn paths recognition system


### Application

~~~
python visual_build/main.py
~~~


### Docker

~~~
docker cp /dir_with_tif_tfw kadmus/app_no_interface:/code/test
~~~


### Preliminary Markup

Запуск из командной строки:  
~~~
python tagging/pipeline.py get_mask img.tif
~~~
Возвращает маску, где красным цветом на белом фоне выделены области, которые нашла нейросеть. Маска сохраняется в img_mask.tif.  
Остается стереть лишние красные метки, затем сконвертировать маску в .npy; команда для конвертации:
~~~
python tagging/pipeline.py get_npy img_mask.tif
~~~
Красный цвет заменяет на белый, все остальное - на черный.  
Команда чтобы посмотреть результат наложения маски и изображения:
~~~
python tagging/pipeline.py blend image.tif mask.tif  
~~~


### Build shapefile

How to use map_builder:  
1. Put all .NPY and .TFW files to separate directory
2. run map_builder/main.py  

output: paths.shp and map.html

~~~python
def build_shapefile(dataset_directory, file_list=None,
                    output_filename: str = 'paths.shp',
                    crs: str = 'epsg:32637',
                    max_path_distance_cm: float = 100,
                    max_path_width_cm: float = 60,
                    min_bbox_size_m: float = 1,
                    max_bbox_size_m: float = 200,
                    p_epsilon: float = 0.3,
                    c_epsilon: float = 0.3)
~~~
Build shapefile containing paths of all given images.

__dataset_directory__: directory where .NPY mask files and .TFW world files are contained  
__file_list__: list of filenames to be processed (without extensions)  
__output_filename__: name of the output file (should be .SHP)  
__crs__: initial coordinate reference system  
__max_path_distance_cm__: max distance between paths for them to be connected in cm  
__max_path_width_cm__: max path width  
__min_bbox_size_m__: min size of path's bounding box in meters  
__max_bbox_size_m__: maxsize of path's bounding box in meters  
__p_epsilon__: RDP parameter to smooth path polygons  
__c_epsilon__: RDP parameter to smooth their centerlines  


### Visualize

~~~python
def visualize(filename: str, output_file: str) -> None
~~~
Build an interactive map visualizing data in a shapefile. (matplotlib==3.3.2 required)
