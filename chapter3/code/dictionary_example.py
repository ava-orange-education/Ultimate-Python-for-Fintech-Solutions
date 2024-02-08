def createDictionary():
	userDict = {
	  "username": "hong",
	  "country": "china",
	  "age": 30,
	  "employed": True
	}
	print("dictioary created is ",userDict)
	
	print("dictionary - username is", userDict["username"])
	
	print("length of dictionary is ",len(userDict))



if __name__ == "__main__":
  
  createDictionary()
  
  user_dict = dict(username = "Thomas", age = 56, country = "USA")
  
  print("dictioary created using constructor is ",user_dict)

  print("dictionary created using constructor - username is", user_dict["username"])

  print("length of dictionary created using constructor is ",len(user_dict))
  