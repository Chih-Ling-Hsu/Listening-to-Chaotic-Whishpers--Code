from .util import *


def get_connection(db_name):
    conn = sqlite3.connect('data/{}.db'.format(db_name))
    conn.text_factory = str
    return conn

def query(firm, cmd):
    with get_connection(firm) as conn:
        cur = conn.cursor()
        cur = cur.execute(cmd)
        
        field_names = [i[0] for i in cur.description]
        results = cur.fetchall()
        
        try:
            df = pd.DataFrame([x for x in results])
            df.columns = field_names
        except:
            #print(cmd)
            df = pd.DataFrame([])
        
        return df

def _fetch_posts(firm, TABLE_NAME, _min, _max, do_filter=False):
    #print(firm, TABLE_NAME, _min, _max)
    if TABLE_NAME=='tweets':    
        with get_connection(firm) as conn:        
            df = query(firm, '''
                    SELECT time, id, user, body, symbols, urls, mentioned_users, source, hashtags, 
                    in_reply_to_status_id_str, in_reply_to_user_id_str, retweeted
                    FROM {}
                    WHERE time >= "{}" AND time < "{}"
                '''.format(TABLE_NAME, _min, _max)) 
    elif TABLE_NAME=='twits':  
        with get_connection(firm) as conn:        
            df = query(firm, '''
                    SELECT time, id, user, body, symbols, urls, mentioned_users, source, 
                    liked_users, sentiment
                    FROM {}
                    WHERE time >= "{}" AND time < "{}"
                '''.format(TABLE_NAME, _min, _max)) 
    else:
        return None  
    return df

# trading hours: UTC 14:30~21:00 (Local 9:30~16:00)
def fetchPostGroup(firm, TABLE_NAME='tweets', date='2018-08-10', do_filter=False):
    _min = datetime.strptime(date, '%Y-%m-%d')
    _max = _min + relativedelta(days=1)
    
    timezone = pytz.timezone("America/New_York")
    _min = timezone.localize(datetime(_min.year, _min.month, _min.day, 16, 0, 0))
    _max = timezone.localize(datetime(_max.year, _max.month, _max.day, 9, 30, 0))
    
    _min = _min.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    _max = _max.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    
    df = _fetch_posts(firm, TABLE_NAME, _min, _max, do_filter)    
    return df

def get_text_info_per_day(firm, start='2018-07-04', end='2018-11-30'):
    text_data_info = {}
    
    dr = [str(d.date()) for d in pd.date_range(start=start, end=end)]
    for date in tqdm(dr, total=len(dr), desc=firm):
        data =[]
        for TABLE_NAME in ['tweets', 'twits']:
            df = fetchPostGroup(firm, TABLE_NAME, date=date, do_filter=False)
            if len(df) > 0:
                df['body'] = [_.replace('\r', '\n') for _ in df['body']]
                data.append(df[['time', 'body']])
        data = pd.concat(data)
        
        if len(data)!=0:
            text_data_info[date] = len(data)
            ensure_dir('{}/text_data/{}'.format(ROOT, firm))
            data.sort_values('time').to_csv('{}/text_data/{}/{}.csv'.format(ROOT, firm, date), index=False)
            
    with open('{}/text_data/{}.json'.format(ROOT, firm), 'w') as f:  
        json.dump(text_data_info, f)

for firm in companyList:
    get_text_info_per_day(firm, start=date_range['train'][0], end=date_range['test'][1])