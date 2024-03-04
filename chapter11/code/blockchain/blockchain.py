import hashlib as hasher
import datetime as date

class FinBlockChain:


    def __init__(self,Name):
        self.name = Name
        self.chain = []
        
    def append(self,block):
        self.chain.append(block) 
    
    def next_block(self,Name,last_block):
        this_index = last_block.index + 1
        this_timestamp = date.datetime.now()
        this_data = Name + str(this_index)
        this_hash = last_block.hash
        return FinBlock(Name,this_index, this_timestamp, this_data, this_hash)    
        
class FinBlock:        
        
    def __init__(self,Name,index, timestamp, data, previous_hash):
        self.name = Name
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        sha = hasher.sha256()
        sha.update(str(self.index).encode('utf-8') + str(self.timestamp).encode('utf-8') + str(self.data).encode('utf-8') + str(self.previous_hash).encode('utf-8'))
        return sha.hexdigest()
    
    
        

fin_blockchain = FinBlockChain("Fintech BlockChain")

fin_block = FinBlock("Block1",0, date.datetime.now(), "Block1".encode('utf-8'), "0".encode('utf-8'));
print("Blockchain name is ",fin_blockchain.name);

fin_previous_block = fin_block

fin_num_of_blocks = 30


for i in range(0, fin_num_of_blocks):
  fin_block_to_add = fin_blockchain.next_block("num"+str(i),fin_previous_block)
  fin_blockchain.append(fin_block_to_add)
  fin_previous_block = fin_block_to_add
  print("The Block #{} is added to the blockchain!".format(fin_block_to_add.index))
  print("The Hash: {}\n".format(fin_block_to_add.hash))







