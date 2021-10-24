from other import den_func
from other import dim_func
import eel

@eel.expose
def test(name):
    print(name)
    #start
    dim_func(name)
    #get masks
    den_func(name)
    #get shapefile
    print("finished")
    eel.my_javascript_function('Finished ', 'job!')
