from qpiai_data_prep.tab.tab_dataprep import tab_dataprep
from temp.db_info import db
 

BASE_DICT = {"request_id": "None", "user_id": "None"}



class pipeline:
    def __init__(self, dataset, target_device, num_device, **kwargs):
        self.dataset = dataset
        self.target_device = target_device
        self.num_device = num_device
        self.list_commands = kwargs.get("cmd_info", {})
        self.base_commands = {k: kwargs.get(k, None) for k in BASE_DICT.keys()}
        self.data_dict = {}
        self.pipeline = True
        self.datatype = kwargs.get("datatype", {})
        if self.base_commands.values() is None:
            return {"error_message": "Please provide valid inputs"}

    def run(self, **kwargs):
        db.update_progress(progress=20)
        count = len(self.list_commands)
        for k, v in self.list_commands.items():
            count -=1
            data_prepcmd = k
            db.update_progress(progress=int(80/(count+1)))
            checkpoint = tab_dataprep(
                self.dataset,
                self.target_device,
                self.num_device,
                data_prepcmd=data_prepcmd,
                pipeline = self.pipeline,
                **self.base_commands,
                **v
            )
            self.data_dict[data_prepcmd] = checkpoint
            data = checkpoint.get("dataPrepOutput")
            data = data[0]
            data = data.split("/")
            dataset = data[-1]
            self.dataset = dataset
        return self.data_dict
