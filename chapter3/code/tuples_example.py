def showTuples():
    
    fruit_tuples = ("pomegranate", "orange", "grape", "guava", "orange")
    print("fruit tuples",fruit_tuples)

    print("length of tuple is ",len(fruit_tuples))
    
    print("fruit_tuple type is ",type(fruit_tuples))
    
    mixed_types_tuples = ("str", 21, False, 60.0, "str2")
    
    print("mixed typed tuples are ",mixed_types_tuples)
    
    created_tuple = tuple(("lime", "beans", "okra"))
    
    print("tuple created through constructor",created_tuple)

if __name__ == "__main__":
  
   showTuples()
  