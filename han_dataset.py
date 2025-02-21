from .util import *

from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0) 
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
        
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

class HANDataset(Dataset):
    def __init__(self, ROOT, firm, date_range=None, metric='Trend', k=5, tau=[-3, 3], n=11, 
                 l_max=40, doc2vec_dim=200):
        self.root = ROOT
        self.data_dir = os.path.join(ROOT, 'saved', firm)
        self.firm = firm
        self.date_range = date_range
        self.metric = metric
        self.k = k
        self.tau = tau       
        self.n = n
        self.l_max = l_max
        self.doc2vec_dim = doc2vec_dim
        self.classLabels = ['flat', 'rise', 'fall']  

        self.init()
        
    def __getitem__(self, i):
        base = datetime.strptime(self.dateList[i], '%Y-%m-%d')
        dateList = [str(base - timedelta(days=_))[:10] for _ in range(self.n)]
        
        ''' Build a 3 dimensions array
        # n, 1st dimension is the window, number of pasted days used to make the prediction
        # i, 2nd dimension is the number of articles that we've got on the corresponding day
        # j, 3rd dimension is the vector representing the article. In 200 dimensions'''
        vectors = [np.load(os.path.join(self.data_dir, '{}.npy'.format(date))) for date in dateList]
        feature = np.vstack([x[np.newaxis, :] for x in vectors]).astype(float)
        label = self.labels[i]
        
        return feature, label

    def __len__(self):
        return len(self.labels)
    
    def _get_label(self, LPC):
        if LPC < 0 and LPC < self.tau[0]:
            return self.classLabels.index('fall')
        elif LPC > 0 and LPC > self.tau[1]:
            return self.classLabels.index('rise')
        else:
            return self.classLabels.index('flat')

    def init(self):
        # Load Stock price time series
        info = pd.read_csv('{}/num_data/{}.csv'.format(self.root, self.firm))

        info['LargestPosChange'] = self.calc_largest_change(self.k, 'positive', info)
        info['LargestNegChange'] = self.calc_largest_change(self.k, 'negative', info)
        info['LargestPosPercentageChange'] = [100.*LC/base for LC, base in zip(info['LargestPosChange'], 
                                                                               info['Close'])]
        info['LargestNegPercentageChange'] = [100.*LC/base for LC, base in zip(info['LargestNegChange'], 
                                                                               info['Close'])]
        info['LargestPercentageChange'] = [p if p > (-n) else n for p,n in zip(info['LargestPosPercentageChange'], 
                                                                               info['LargestNegPercentageChange'])]
        info['LargestAbsPercentageChange'] = [abs(x) for x in info['LargestPercentageChange']]
            
        # Trend classes 
        info['Trend'] = [self._get_label(LPC) for LPC in info['LargestPercentageChange']]
        info['isTrend'] = [int(not math.isnan(i) and self.classLabels[int(i)] != 'flat') for i in info['Trend']] 

        self.info = info
        self.dateList = [x for x in self.info['Date'] if ((x >= self.date_range[0]) & (x <= self.date_range[1]))]
        self.indexList = [self.info['Date'].tolist().index(x) for x in self.dateList]
        self.labels = [self.info[self.metric].tolist()[i] for i in self.indexList]
                          
        assert sum(np.isnan(self.labels))==0

    def get_docs_by_date(self, model, text_data_info, d):
        ''' Build a 2 dimensions array
            # i, 1st dimension is the number of articles that we've got on the corresponding day
            # j, 2nd dimension is the vector representing the article. In 200 dimensions'''
        if d not in text_data_info:
            X = np.zeros((self.l_max, self.doc2vec_dim))
        else:
            vectors = []
            for i in range(min(text_data_info[d], self.l_max)):
                vectors.append(model.docvecs['{}_{}'.format(d, i)][np.newaxis, :])
            for i in range(text_data_info[d], self.l_max):
                vectors.append(np.zeros((1, self.doc2vec_dim)))
            X = np.vstack(vectors)
        return X
    
    def prepare(self):  
        ensure_dir(self.data_dir) 
        model = Doc2Vec.load(os.path.join(self.root, 'saved', 'd2v.model'))
        
        with open('{}/text_data/{}.json'.format(self.root, self.firm), 'r') as f:  
            text_data_info = json.load(f)
        
        start = datetime.strptime(self.dateList[0], '%Y-%m-%d') - timedelta(days=self.n)
        end = datetime.strptime(self.dateList[-1], '%Y-%m-%d')
        d_range = pd.date_range(start=start, end=end)
        for d in d_range:        
            d = str(d)[:10]
            X = self.get_docs_by_date(model, text_data_info, d)
            np.save(os.path.join(self.data_dir, '{}.npy'.format(d)), X)

    """
    @Param step: (int) we compare price on day (t) and day (t+step)
    """
    def calc_largest_change(self, k=5, sign='positive', info=None):
        PCs = map(partial(self.calc_change, info=info), [step+1 for step in range(k)])
        PCs = list(map(list, zip(*PCs)))
        
        if sign == 'positive':
            LCs = [0 if max(x)<0 else max(x) for x in PCs]
        elif sign == 'negative':
            LCs = [0 if min(x)>0 else min(x) for x in PCs]
        else:
            return None
        
        return list(LCs)[:-k] + [None]*k
    
    """
    @Param step: (int) we compare price on day (t) and day (t+step)
    """
    def calc_change(self, step=1, info=None, col='Close'):
        if step >= 0:
            prices = info[col].values.tolist()
            prices_comp = prices[step:] + prices[-1:]*step

            PC = [(p2 - p1) for p1, p2 in zip(prices, prices_comp)]
            return PC
        else:
            prices = info[col].values.tolist()
            prices_comp = prices[:1]*(-step) + prices[:step]

            PC = [(p2 - p1) for p1, p2 in zip(prices_comp, prices)]
            return PC
    
class HANDataLoader(BaseDataLoader):
    def __init__(self, ROOT, firm, date_range, metric, k, tau, n=11,
                 batch_size=8, shuffle=False, validation_split=0.0, num_workers=4):
        self.dataset = HANDataset(ROOT, firm, date_range=date_range, metric=metric, k=k, tau=tau, n=n)
        super(HANDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)