def selectionSortAlgo(to_be_sorted, length):
	
	for s in range(length):
		min_idx = s
		
		for i in range(s + 1, length):
			if to_be_sorted[i] < to_be_sorted[min_idx]:
				min_idx = i
		(to_be_sorted[s], to_be_sorted[min_idx]) = (to_be_sorted[min_idx], to_be_sorted[s])

if __name__ == "__main__":
  
   to_be_sorted = [ 11, 5, 3, 9,14 ]
   len_list = len(to_be_sorted)
   selectionSortAlgo(to_be_sorted, len_list)

   print('to_be_sorted after Selection Sort in Ascending Order is :',to_be_sorted)
