from conf.mongodb.mongodb_conf import MONGO_URL

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def create_mongodb_connection():
  mongodb_client_connection = MongoClient(MONGO_URL)
  return mongodb_client_connection

def close_mongodb_connection(mongodb_client_connection):
  mongodb_client_connection.close()