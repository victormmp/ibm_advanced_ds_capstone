import csv
import pandas as pd
import numpy as np
import os
import sys

root_dir = os.path.dirname(__file__)
sys.path.insert(0, root_dir + '/../..')

class ETL:
    def __init__(self):
        self.data = None

    def load_data(self, path):
        self.data = pd.read_csv(path)

        return self
    
    