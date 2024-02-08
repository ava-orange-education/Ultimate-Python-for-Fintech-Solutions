def create3DArray():
    
    threed_array = [[ ['1' for col in range(3)] for col in range(3)] for row in range(3)]
    
    print("3darray initialized : ",threed_array)

    insert_row = ['0','2','3']
    threed_array.insert(2,insert_row)
    
    print("updated 3d array is:",threed_array)
  
    threed_array.pop()
    
    print("updated 3darray after pop",threed_array)

if __name__ == "__main__":
  
  create3DArray()
  