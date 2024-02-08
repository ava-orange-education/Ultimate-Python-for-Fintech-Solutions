import os

def write_file(path):
	  new_file = open(path, "w")
	  new_file.write("new file first sentence")
	  new_file.close()
  
def execute_file_operations(path):
    
    os.rename(path, "renamed_file.txt")
    write_file("to_be_deleted.txt")
    
    os.remove("to_be_deleted.txt")
    
def execute_dir_operations():
    dir_path = "new_directory"
    os.mkdir(dir_path)
    
    os.chdir(dir_path)
    
    pwd = os.getcwd()
    
    print("present working directory : ",pwd)
    
    dir_path = "sub_directory"
    
    os.mkdir(dir_path)
    
    os.rmdir(dir_path)


if __name__ == "__main__":
   path = "oldfile.txt"
   write_file(path)
   execute_file_operations(path)
   execute_dir_operations()