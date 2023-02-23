import time

from qpiai_utils.database import QpiDatabase
from qpiai_utils.errors import QpiException


def divide(db: QpiDatabase, a, b):
    for i in range(1, 11):
        db.update_progress(progress=i * 10)
        if b == 0:
            raise QpiException(message="Cannot divide by zero")
        time.sleep(2)
    return a / b


def addition(db: QpiDatabase, a, b):
    try:
        for i in range(1, 11):
            db.update_progress(progress=i * 10)
            time.sleep(2)
        return a + b
    except ValueError:
        raise QpiException(message="Incorrect Value")
