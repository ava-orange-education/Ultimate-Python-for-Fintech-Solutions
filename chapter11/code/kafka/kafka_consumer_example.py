
from time import sleep
from json import dumps
from kafka import KafkaConsumer


def main():
    
    kafka_consumer = KafkaConsumer(
        'kafka_topic',
         bootstrap_servers=['localhost:9092'],
         auto_offset_reset='earliest',
         enable_auto_commit=True,
         group_id='kafka-group')


    for rec_message in kafka_consumer:
        val_message = rec_message.value
        print(' message',val_message)
    
if __name__ == '__main__':
   main()