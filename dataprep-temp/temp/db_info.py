import json

from qpiai_utils.database import QpiDatabase

# process.py writes to db_config.json
with open('temp/db_config.json', 'r') as f:
    db_config = json.load(f)

# NOTE: db_config['input_args'] is an empty dict here
# only use this variable to update progress
# don't use it for anything which requires input_args (ex. update_process_state)
db = QpiDatabase(**db_config)