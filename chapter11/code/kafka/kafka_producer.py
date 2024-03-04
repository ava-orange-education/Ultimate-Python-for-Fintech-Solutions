from time import sleep
from json import dumps
from kafka import KafkaProducer



    
    

def main():
    
    kafka_producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))
                         
    print("sending the message")               
                         
    for number in range(300):
        message_data = {'Message id' : 'Message code '+str(number)}
        kafka_producer.send('kafka_topic', value=message_data)
        sleep(1)  
          
if __name__ == '__main__':
   main()