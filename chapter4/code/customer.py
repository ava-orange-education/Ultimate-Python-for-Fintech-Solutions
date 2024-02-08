import logging

class Customer:
  def __init__(self, firstname,lastname,middlename, ssn):
      self.first_name = firstname
      self.last_name = lastname
      self.middle_name = middlename
      self.national_id = ssn
      
def main():
  logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
  logging.getLogger("finance").setLevel(logging.INFO)
  customer = Customer("Michael","Smith","Thomas", 14322226)
  print("customer name",customer.first_name," ",customer.middle_name," ",customer.last_name," ",customer.national_id)
  print("customer national id",customer.national_id)
  
  
if __name__ == "__main__":
   main()
