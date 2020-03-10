# This program combines 10 Hz data from both spacecraft and 
# preserves only the data taken at the same time.

import pandas as pd

import detect_daily
import dirs

class CombineData():
    def __init__(self, ):
        dirs.AC6_MERGED_DATA_PATH
        return