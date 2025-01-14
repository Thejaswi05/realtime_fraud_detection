from kafka import KafkaConsumer
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_consumer():
    try:
        # Initialize consumer
        consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers=['kafka:29092'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='fraud_detection_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info("Kafka consumer started successfully")
        
        # Process messages
        for message in consumer:
            try:
                transaction = message.value
                logger.info(f"Received transaction: {transaction}")
                # Add processing logic here
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error starting consumer: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Kafka consumer...")
    start_consumer()
