import numpy as nump 

def createVector():
    vector = [[2], [4], [6],[10]]  
    
    nump_vector = nump.array(vector) 
    
    print("Nump Vector from a  list:",nump_vector) 
  
    vector2 = [11,12,13,14,15]
  
    vector3 = [21,22,23,24,25]
  
    nump_vector2 = nump.array(vector2) 
  
    nump_vector3 = nump.array(vector3)
  
    sum_vector = nump_vector2 + nump_vector3
  
    print("Nump sum Vector :",sum_vector) 

    diff_vector = nump_vector3 - nump_vector2
	
    print("Nump subtraction Vector :",diff_vector) 
    
    prod_vector = nump_vector2 * nump_vector3
    
    print("Nump product Vector :",prod_vector) 
    
    
    div_vector = nump_vector3 * nump_vector2
    
    print("Nump division Vector :",div_vector) 
    
    dot_vector = nump_vector3.dot(nump_vector2)
    
    print("Nump dot Vector :",dot_vector) 
  
  

if __name__ == "__main__":
  
   createVector()
  