from util import *

def test(mdl_pth, firm, date_range, metric, k, tau):  
    data_loader = HANDataLoader(firm=firm, date_range=date_range, metric=metric, k=k, tau=tau, 
                                batch_size=8, shuffle=True, validation_split=0, num_workers=8)
    
    model = load_model(mdl_pth)
    
    Y_test, Y_pred = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        outputs = model.predict(np.asarray(data))
        Y_test.extend(target.tolist())
        Y_pred.extend([np.argmax(output) for output in outputs])
        
    return Y_test, Y_pred

experiment_results = []
for model_name, k, tau, target, firm in gen_meta_info_params(models, ks, taus, targets, companyList):
    meta_info = 'k_{}.tau_{}.{}'.format(k, tau, target)
    
    mdl_pth = '{}/saved/models/han.{}.{}.hdf5'.format(ROOT, meta_info, firm)
    Y_test, Y_pred = test(mdl_pth, firm, date_range['test'], target, k, tau)
    
    result = {'k': k, 'tau': tau, 'target':target, 'firm':firm, 'model': model_name}
    result.update(eval_results(Y_test, Y_pred))
    experiment_results.append(result)

df.to_csv('{}/experiment_results.csv'.format(ROOT), index=None)