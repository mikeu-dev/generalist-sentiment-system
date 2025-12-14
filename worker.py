
import os
import redis
from rq import Worker, Queue, Connection
from app import app
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

listen = ['default']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    logger.info(f"Starting RQ Worker listening on {listen}...")
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
