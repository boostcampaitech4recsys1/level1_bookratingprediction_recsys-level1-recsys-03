import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel, _HighOrderFactorizationMachineModel, _DeepFactorizationMachineModel
from ._models import rmse, RMSELoss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F

class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        if args.FOLD == False:
            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims'] #array([ 68069, 149570, 6, 11990, 1323, 269, 413, 1523, 27, 62059]

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.batch_size = args.BATCH_SIZE
        self.device = args.DEVICE
        self.seed = args.SEED
        #self.data_shuffle = args.DATA_SHUFFLE
        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
        self.f4 = True if args.MODEL == 'F4' else False 
    def train(self):
        rmse_score = 0
        rmse = []
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100     
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                
                fields, target = fields.to(self.device), target.to(self.device)
                
                y = self.model(fields)
                loss = self.criterion(y, target.float())

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0
                # if i%450 == 0 and epoch > 1:
                #     rmse_score = self.predict_train()
                #     rmse.append(rmse_score)
                #     print('epoch:', epoch, 'validation: rmse:', rmse_score)

            rmse_score = self.predict_train()
            rmse.append(rmse_score)
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
        print(min(rmse))
        return min(rmse)



    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        if self.f4:
            pd.DataFrame({'targets':targets, 'predicts':predicts}).to_csv('fm.csv',index=False)
            print('save_fm')
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        
        return predicts
    
    def train_fold(self, data):
        n=5
        str_kf = StratifiedKFold(n_splits = n, shuffle = True, random_state = self.seed)
        #kFold = KFold(n_splits=n, shuffle=True, random_state=self.seed)
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predicts = []
        rmse = []
        X, y = data['train'].drop(['rating'], axis=1), data['train']['rating']

        for i, (train_index, test_index) in enumerate(str_kf.split(X, y)):
            X_train, X_valid = X.loc[train_index], X.loc[test_index]
            y_train, y_valid = y.loc[train_index], y.loc[test_index]

            data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

            train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
            valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

            data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']

            self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
            print('')
            print(f'--------------- FM TRAIN FOLD{i} ---------------')
            rmse.append(self.train())
            predicts.append(self.predict(test_dataloader))

            
        print('')
        print(f'--------------- FM 5-FOLD SCORE : {sum(rmse)/n} ---------------')

        return np.sum(np.array(predicts), axis=0)/n


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        if args.FOLD == False:
            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.seed = args.SEED
        self.batch_size = args.BATCH_SIZE
        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
        self.f4 = True if args.MODEL == 'F4' else False 

    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        rmse_score = 0
        rmse = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
            rmse.append(rmse_score)
        print(min(rmse))
        return min(rmse)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        if self.f4:
            pd.DataFrame({'targets':targets, 'predicts':predicts}).to_csv('ffm.csv',index=False)
            print('save_ffm')
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts
    
    def train_fold(self, data):
        n=5
        str_kf = StratifiedKFold(n_splits = n, shuffle = True, random_state = self.seed)
        #kFold = KFold(n_splits=n, shuffle=True, random_state=self.seed)
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predicts = []
        rmse = []
        X, y = data['train'].drop(['rating'], axis=1), data['train']['rating']

        for i, (train_index, test_index) in enumerate(str_kf.split(X, y)):
            X_train, X_valid = X.loc[train_index], X.loc[test_index]
            y_train, y_valid = y.loc[train_index], y.loc[test_index]

            data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

            train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
            valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

            data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']

            self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
            print('')
            print(f'--------------- FM TRAIN FOLD{i} ---------------')
            rmse.append(self.train())
            predicts.append(self.predict(test_dataloader))

            
        print('')
        print(f'--------------- FM 5-FOLD SCORE : {sum(rmse)/n} ---------------')

        return np.sum(np.array(predicts), axis=0)/n
    
class HighOrderFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        if args.FOLD == False:
            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.seed = args.SEED
        self.embed_dim = args.HOFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.order = args.ORDER
        self.device = args.DEVICE
        self.batch_size = args.BATCH_SIZE
        self.f4 = True if args.MODEL == 'F4' else False 
        self.model = _HighOrderFactorizationMachineModel(self.field_dims, self.order, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        rmse_score = 0
        rmse = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
            rmse.append(rmse_score)
        print(min(rmse))
        return min(rmse)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        if self.f4:
            pd.DataFrame({'targets':targets, 'predicts':predicts}).to_csv('hofm.csv',index=False)
            print('save_hofm')
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts

    def train_fold(self, data):
        n = 5
        str_kf = StratifiedKFold(n_splits = n, shuffle = True, random_state = self.seed)
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predicts = []
        rmse = []
        X, y = data['train'].drop(['rating'], axis=1), data['train']['rating']

        for i, (train_index, test_index) in enumerate(str_kf.split(X, y)):
            X_train, X_valid = X.loc[train_index], X.loc[test_index]
            y_train, y_valid = y.loc[train_index], y.loc[test_index]

            data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

            train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
            valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

            data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']

            self.model = _HighOrderFactorizationMachineModel(self.field_dims, self.order, self.embed_dim).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
            print('')
            print(f'--------------- HOFM TRAIN FOLD{i} ---------------')
            rmse.append(self.train())
            predicts.append(self.predict(test_dataloader))
            
        print('')
        print(f'--------------- FM 5-FOLD SCORE : {sum(rmse)/n} ---------------')
        return np.sum(np.array(predicts), axis=0)/n
    
    
    
    
class DeepFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        if args.FOLD == False:
            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.seed = args.SEED
        self.embed_dim = args.DFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.mlp_dims = args.MLP_DIMS
        self.drop_out = args.DFM_DROP_OUT
        self.device = args.DEVICE
        self.batch_size = args.BATCH_SIZE
        
        self.model = _DeepFactorizationMachineModel(self.field_dims, self.embed_dim, self.mlp_dims, self.drop_out).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
#field_dims, embed_dim, mlp_dims, dropout

    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        rmse_score = 0
        rmse = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
            rmse.append(rmse_score)
        print(min(rmse))
        return min(rmse)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        pd.DataFrame({'targets':targets, 'predicts':predicts}).to_csv('hofm.csv',index=False)
        print('save_hofm')
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts

    def train_fold(self, data):
        n = 5
        str_kf = StratifiedKFold(n_splits = n, shuffle = True, random_state = self.seed)
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predicts = []
        rmse = []
        X, y = data['train'].drop(['rating'], axis=1), data['train']['rating']

        for i, (train_index, test_index) in enumerate(str_kf.split(X, y)):
            X_train, X_valid = X.loc[train_index], X.loc[test_index]
            y_train, y_valid = y.loc[train_index], y.loc[test_index]

            data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

            train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
            valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

            data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']

            self.model = _DeepFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
            print('')
            print(f'--------------- DFM TRAIN FOLD{i} ---------------')
            rmse.append(self.train())
            predicts.append(self.predict(test_dataloader))
            
        print('')
        print(f'--------------- FM 5-FOLD SCORE : {sum(rmse)/n} ---------------')
        return np.sum(np.array(predicts), axis=0)/n