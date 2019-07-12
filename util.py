import yaml
import json
import subprocess
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
from tqdm import tqdm
import os
from functools import reduce, partial

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
import sqlite3

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    ROOT = '.'
    companyList = ['AAPL', 'AMZN', 'BIDU', 'GOOG', 'MSFT', 'NFLX']
    date_range = {
        'train': ['2018-07-04', '2018-12-31'],
        'test': ['2019-01-01', '2019-03-31']
    }

    ks = [5, 1]
    taus = ['std']
    models = ['HAN']
    targets = ['isTrend', 'Trend']