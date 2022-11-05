import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6 
    
    
def process_context_data(users, books, ratings1, ratings2): # label-encoding 
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title']], on='isbn', how='left')
    
    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)
    
    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    #year2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}
    title2idx = {v:k for k,v in enumerate(context_df['book_title'].unique())}
    
    
    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['book_title'] = train_df['book_title'].map(title2idx)
    #train_df['year_of_publication'] = train_df['year_of_publication'].map(year2idx)
    
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['book_title'] = test_df['book_title'].map(title2idx)
    #test_df['year_of_publication'] = test_df['year_of_publication'].map(year2idx)

    train_df['age'] = train_df['age'].fillna(30)
    train_df['age'] = train_df['age'].apply(age_map)

    test_df['age'] = test_df['age'].fillna(30)
    test_df['age'] = test_df['age'].apply(age_map)

    
    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "title2idx":title2idx,
        #"year_of_publication":year2idx
    }
   
    return idx, train_df, test_df


def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users_f_location_1.1.csv')
    books = pd.read_csv(args.DATA_PATH + 'books_1.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    users['location_country'] = users['location_country'].fillna('usa')
    def get_core(x):
        if x in l:
            return x
        else:
            return 'others'
        
    l = list(books.groupby('category')['category'].count().sort_values(ascending=False).head(90).index) # 70
    books['category'] = books['category'].apply(get_core)
    l = list(books.groupby('language')['language'].count().sort_values(ascending=False).head(9).index)
    books['language'] = books['language'].apply(get_core)
    l = list(books.groupby('publisher')['publisher'].count().sort_values(ascending=False).head(250).index) # 200 
    books['publisher'] = books['publisher'].apply(get_core)
    # l = list(books.groupby('book_title')['book_title'].count().sort_values(ascending=False).head(100).index)
    # books['book_title'] = books['book_title'].apply(get_core)
    l = list(users.groupby('location_country')['location_country'].count().sort_values(ascending=False).head(32).index)
    users['location_country'] = users['location_country'].apply(get_core)
    # l = list(users.groupby('location_state')['location_state'].count().sort_values(ascending=False).head(100).index)
    # users['location_state'] = users['location_state'].apply(get_core)
    # l = list(users.groupby('location_city')['location_city'].count().sort_values(ascending=False).head(600).index)
    # users['location_city'] = users['location_city'].apply(get_core)
    l = list(books.groupby('book_author')['book_author'].count().sort_values(ascending=False).head(450).index) # 400
    books['book_author'] = books['book_author'].apply(get_core)
    
    # books['category'] = 'nan'
    # books['publisher'] = 'nan'
    # #books['language'] = 'nan'
    # books['book_author'] = 'nan'
    books['book_title'] = 'nan'
    users['location_city'] = 'nan'
    users['location_state'] = 'nan'
    # users['location_country'] = 'nan'
    # users['age'] = 10
    # users['user_id'] = 1
    
    ids = pd.concat([train['user_id'], sub['user_id']]).unique() # 전체 유저
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique() # 전체 item

    idx2user = {idx:id for idx, id in enumerate(ids)} 
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()} #label encoding
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()} #label encoding

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            1, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']),len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']), len(idx['title2idx'])], dtype=np.uint32)

    
    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
