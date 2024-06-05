import os
import numpy as np
import pandas as pd
import tensorflow as tf

import string

DATA_DIR = "./data"

filenames = [DATA_DIR + filename for filename in os.listdir(DATA_DIR)]
print(filenames)