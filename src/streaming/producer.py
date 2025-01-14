from kafka import KafkaProducer
import json
import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProducer:
    def __init__(self, bootstrap_servers=['kafka:29092']):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        logger.info("Kafka producer initialized")

    def send_transaction(self, transaction):
        try:
            self.producer.send('transactions', transaction)
            self.producer.flush()
            logger.info(f"Sent transaction: {transaction}")
        except Exception as e:
            logger.error(f"Error sending transaction: {str(e)}")

def test_producer():
    producer = TransactionProducer()
    # Test transaction
    transaction = {
        "amount": 100.0,
        "time": 3600
    }
    producer.send_transaction(transaction)

if __name__ == "__main__":
    logger.info("Starting test producer...")
    test_producer()
