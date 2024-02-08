def create2DArray():
    vehicles = ["Auto", "Truck", "Bus"]
    
    vehicle0 = vehicles[0]
    
    print("vehicle index 0",vehicle0)
    
    vehicles_length = len(vehicles)
    
    for vehicle in vehicles:
        print("vehicle ",vehicle)
    
    vehicles.append("Car")

    vehicles.pop(1)
  
    print("vehicles after poping index 1",vehicles)

    vehicles.remove("Bus")
    
    print("vehicles after removing Bus",vehicles)

if __name__ == "__main__":
  
  create2DArray()
  