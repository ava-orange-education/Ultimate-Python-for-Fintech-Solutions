from scipy.interpolate import interp1d as interpol
import numpy as nump



def interpolate(xarr,yarr):

  	interp_func = interpol(xarr, yarr)

  	newarr = interp_func(nump.arange(3.1, 4, 0.2))

  	print("interpolated array between 3 and 4 with 0.2 increments is ",newarr)



if __name__ == "__main__":

   xarr = nump.arange(12)
   yarr = 2*xarr + 1
   
   print("xarr is ",xarr)
   
   print("yarr is ",yarr)

   interpolate(xarr,yarr)