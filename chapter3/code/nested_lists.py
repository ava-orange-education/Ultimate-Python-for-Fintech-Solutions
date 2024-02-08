


def createNestedList():
    nested_list = []

    for i in range(7):
	

    	nested_list.append([])
	
    	for j in range(7):

    		nested_list[i].append(j)
		
    print("nested list is ",nested_list)



if __name__ == "__main__":
  
  createNestedList()
  
    