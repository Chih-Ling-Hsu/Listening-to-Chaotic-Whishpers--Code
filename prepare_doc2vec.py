from .util import *

def prepare_data(ROOT, companyList):
    data = []
    tag = []
    i = 0

    for firm in companyList:
        folder_name = '{}/text_data/{}'.format(ROOT, firm)
        for fname in os.listdir(folder_name):
            df = pd.read_csv(os.path.join(folder_name, fname), index_col=None)
            data += df['body'].tolist()
            tag += ['{}_{}'.format(fname[:-4], idx) for idx in range(len(df))]
            i += len(df)

    return [TaggedDocument(words=word_tokenize(str(_d.lower())), tags=[_t]) for _d, _t in zip(data, tag)]


def train_doc2vec_model(tagged_data, save_path, max_epochs=15, vec_size=200, alpha=0.025):
    model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.025, min_count=5, dm=1, workers=16)
    model.build_vocab(tagged_data)
    
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        # decrease learning rate
        model.alpha -= 0.0002
        # and reinitialize it
        model.min_alpha = model.alpha

    model.save(save_path)

if __name__ == '__main__':
    tagged_data = prepare_data(ROOT, companyList)
    train_doc2vec_model(tagged_data, '{}/saved/d2v.model'.format(ROOT), max_epochs=15, vec_size=200, alpha=0.025)