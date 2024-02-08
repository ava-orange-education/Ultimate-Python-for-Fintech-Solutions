

def read_file(path,pos):

	read_file = open(path, "r+")
	read_content = read_file.read(pos)
	print("content at the position ",pos," is: ", read_content)


	current_pos = read_file.tell()
	print("The position of the file reader is : ", current_pos)

	position = read_file.seek(0, 0);
	read_data = read_file.read(pos)
	print("The data read again is:", read_data)
	read_file.close()


if __name__ == "__main__":
   path = "newfile.txt"
   read_file(path,5)