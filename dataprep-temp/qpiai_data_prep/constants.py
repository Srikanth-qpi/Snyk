import os
home_directory = os.path.expanduser('~')
MODEL_SUBMIT_DIR = os.environ.get("DATASET_MOUNT",home_directory) 

# MONGODB_USER = os.environ.get('MONGODB_USER','qpiai')
# MONGODB_PASS = os.environ.get('MONGODB_PASS','qpiai')
# MONGODB_HOST = os.environ.get('MONGODB_HOST','localhost')
# MONGODB_NAME = os.environ.get('MONGODB_NAME','qpiai')


MONGODB_USER = os.getenv("MONGODB_USER", "qpiai")
MONGODB_PASS = os.getenv("MONGODB_PASS", "qpiai")
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = os.getenv("MONGODB_PORT", "27017")
MONGODB_NAME = os.getenv("MONGODB_NAME", "admin")

MONGODB_CONN_ID = f"mongodb://{MONGODB_USER}:{MONGODB_PASS}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_NAME}"

