import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL_1") + os.getenv("MONGODB_PASSWORD") + os.getenv("MONGODB_URL_2")