def searchElement(list,element):
    for i in range (len(list)):
        if list[i] == int(element):
           print("found",i)
           return i
    return -1

if __name__ == "__main__":
  
   list = [2,6,8,0,11,9,12,14]
   
   element = 11
   found_index = searchElement(list,element)
  