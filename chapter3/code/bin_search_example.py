def binarySearchElement(list,element,low,high):
    if high >= low:

        mid = low + (high - low)//2

        if list[mid] == element:
            return mid

        elif list[mid] > element:
            return binarySearchElement(list, element, low, mid-1)

        else:
            return binarySearchElement(list, element, mid + 1, high)

    else:
        return -1
if __name__ == "__main__":
  
   list = [2,6,8,0,11,9,12,14]
   
   list.sort()
   
   print("sorted list",list)
   
   element = 11
   found_index = binarySearchElement(list,element,0,len(list)-1)
   
   print("found",found_index)