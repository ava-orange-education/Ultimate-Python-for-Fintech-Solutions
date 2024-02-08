def write_file():
	new_file = open("newfile.txt", "w")
	new_file.write("new file first sentence")
	new_file.close()
	
	
def read_file():
    new_file = open("newfile.txt", "r")	
	read_file = new_file.read()
	print("file content",read_content)
	new_file.close()

if __name__ == "__main__":
  
   write_file()
   
   read_file()