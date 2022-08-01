import rpyc as r


from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np



class FoldDistributor:
    def __init__(self, nFolds):
        print(r.list_services(timeout=10))
        if len(r.list_services(timeout=10)) < nFolds:
            raise Exception("Not enough services available.")
        
        self.nFolds = nFolds
        self.establish_connection()

    # tries to connect to all Fold Services
    def establish_connection(self):
        self.services = [r.connect_by_service(f"FOLD{k}").root for k in range(self.nFolds)]
    
    def ping_services(self):
        successful = 0
        for s in self.services:
            if s.exposed_ping(): successful += 1

        return f"There are {successful} out of {len(self.services)} services responded to the ping."

    def distribute_folds(self, data: pd.DataFrame, target: pd.Series, model: LinearSVC) -> list:
        # build indices of each fold
        skf = StratifiedKFold(n_splits=self.nFolds, shuffle=True)

        # for each fold, train on fold remotely
        results = list()
        for i, (train_i, test_i) in enumerate(skf.split(data, target)):
            
            # build the request Object
            requestObj = {
                "id": i,
                "default_model": model,
                "all_data_x": data,
                "all_data_y": target,
                "train_index": train_i,
                "test_index": test_i
            }
            
            # send request Object and "await" reply
            replyObj = self.services[i].exposed_train_on_fold(requestObj)

            # consolidate to results
            results.append(replyObj)

        return results

