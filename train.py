from util import *

def train(firm, date_range, metric, k, tau, n_epochs=60):
    enc = OneHotEncoder(handle_unknown='ignore')
    if metric == 'isTrend':
        n_class = 2
        enc.fit(np.array([[0], [1]]))
    else:
        n_class = 3
        enc.fit(np.array([[0], [1], [2]]))
    
    meta_info = 'k_{}.tau_{}.{}'.format(k, tau, metric)
    data_loader = HANDataLoader(firm=firm, date_range=date_range, metric=metric, k=k, tau=tau, 
                                batch_size=8, shuffle=True, validation_split=0, num_workers=8)
    
    model = han(input_dim=(11, 40, 200), output_dim=n_class)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            target = enc.transform(target.numpy()[:, np.newaxis]).toarray()
            model.train_on_batch(np.asarray(data), np.asarray(target))
    
    mdl_pth = '{}/saved/models/han.{}.{}.hdf5'.format(ROOT, meta_info, firm)
    model.save(mdl_pth) 
    return mdl_pth

for model_name, k, tau, target, firm in gen_meta_info_params(models, ks, taus, targets, companyList):
    train(firm, date_range['train'], target, k, tau, n_epochs=60)
