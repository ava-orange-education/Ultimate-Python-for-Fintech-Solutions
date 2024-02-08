import zope.interface


class ExampleInterface(zope.interface.Interface):
	val = zope.interface.Attribute("prop1")
	def executeTask(self, val):
		pass
	def printOutput(self):
		pass

@zope.interface.implementer(ExampleInterface)
class ExampleClass():
    	val = zope.interface.Attribute("prop1")
    	def executeTask(self, val):
          print("executing Task and prop1 value is",val)
    	def printOutput(self):
          print("printing output")
	
print(type(ExampleInterface))
print(ExampleInterface.__module__)
print(ExampleInterface.__name__)

prop_value = ExampleInterface['val']
print(prop_value)
print(type(prop_value))

exampleObj = ExampleClass()
exampleObj.executeTask(4)
exampleObj.printOutput()
