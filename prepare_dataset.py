from .util import *
from .han_dataset import *

for firm in tqdm(companyList, total=len(companyList)):
    ds = HANDataset(firm, date_range=(date_range['train'][0], date_range['test'][1]), 
                    l_max=40, doc2vec_dim=200)
    ds.prepare()