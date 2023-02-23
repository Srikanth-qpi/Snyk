import json
from qpiai_data_prep.config import DB_LOG, RequestModel
from qpiai_utils.database import QpiDatabase

with open('temp/db_config.json', 'r') as f:
    inp = json.load(f)

req = RequestModel(**inp)
db = QpiDatabase(request_id=req.request_id,
user_id=req.user_id, feature_type=req.feature_type, input_args=req.input, no_db = False)

while True:
    red_db = db.redis_db
    pubsub = red_db.pubsub()
    pubsub.subscribe([inp['request_id'].encode("utf-8")])
    for item in pubsub.listen():
        print(item)
