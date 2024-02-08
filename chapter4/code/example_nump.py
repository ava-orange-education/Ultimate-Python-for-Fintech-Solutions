
import numpy as nump


def sort_array(array):
    print("array = {}".format(array))
    index = nump.where(array == 32)
    print("index = {}".format(index))
    sorted_array = nump.sort(array, axis = None)       
    print("\nsorted array is \n", sorted_array)
    sorted_array_indices = nump.argsort(array)
    print('Sorted indices of the original array is', sorted_array_indices)

if __name__ == "__main__":

   array = nump.array([11, 22, 45, 56, 41, 32, 76, 19])

   sort_array(array)
