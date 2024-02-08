def filterList(to_be_filtered):
	result = filter(lambda x: x % 2 != 0, to_be_filtered)
	
	return list(result)

if __name__ == "__main__":
  
   to_be_filtered =  [0, 3, 2, 7, 9, 8, 13,15,17,21]
   
   result_filtered = filterList(to_be_filtered)
   print('filtered odd numbers in the list :',result_filtered)