import logging
from datetime import datetime
import pandas as pd
import os

LOG_DIR = "application_logs"

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
level = logging.INFO,
format = '[%(asctime)s]|[%(module)s] | %(levelname)s | %(message)s',
datefmt='%d-%m-%Y %H:%M:%S'
)

def get_log_dataframe(file_path):
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("|"))

    log_df = pd.DataFrame(data)
    columns=["Time stamp","Module","Level","Message"]
    log_df.columns=columns
    
    log_df["log_message"] = log_df['Time stamp'].astype(str) +":$"+ log_df["Message"]

    return log_df[["log_message"]]