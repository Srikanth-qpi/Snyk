from qpiai_data_prep.process import process_main
from _pytest.monkeypatch import MonkeyPatch
from base_dataprep import *


def test_addition(monkeypatch):
    monkeypatch.setenv("DB_LOG", "false")
    input = {
        "request_id": "507f1f77bcf86cd799439011",
        "user_id": "test_user1",
        "feature_type": "qpiai-microservice-template",
        "input": {"pipeline": "add", "num1": 2, "num2": 3},
    }
    response = process_main(input)
    assert response == {"output": 5}



def test_dataprep(list_of_curls, monkeypatch=MonkeyPatch()):
    monkeypatch.setenv("DB_LOG", "false")
    for i in list_of_curls:

        input = {
            "request_id": "507f1f77bcf86cd799439011",
            "user_id": "test_user1",
            "feature_type": "qpiai-microservice-template",
            "input": {**i}
        }
        response = process_main(input)
    return response 


test_dataprep(list_of_curls, monkeypatch=MonkeyPatch())
