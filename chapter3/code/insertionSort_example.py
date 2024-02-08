def insertionSortAlgo(to_be_sorted):
	n = len(to_be_sorted) 
	
	if n <= 1:
		return 

	for i in range(1, n): 
		key = to_be_sorted[i] 
		j = i-1
		while j >= 0 and key < to_be_sorted[j]: 
			to_be_sorted[j+1] = to_be_sorted[j] 
			j -= 1
		to_be_sorted[j+1] = key 


if __name__ == "__main__":
  
   to_be_sorted = [ 11, 5, 3, 9,14 ]
   len_list = len(to_be_sorted)
   insertionSortAlgo(to_be_sorted)

   print('to_be_sorted after Insertion Sort in Ascending Order is :',to_be_sorted)
   
