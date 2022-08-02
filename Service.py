import rpyc as r
from rpyc.utils.server import ThreadedServer
r.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

from typing import Any


from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np


# run this first on the terminal
# python rpyc_registry.py -l true -t 500

from plumbum import cli

fold = None
while True:
    fold = input("Fold Number: ")
    if int(fold) < 0:
        print("Put in an integer idiot.")
    else: break


class FoldService(r.Service):
    ALIASES = [f'FOLD{int(fold)}']

    def on_connect(self, conn):
        print(f"Someone connected to {FoldService.ALIASES[0]} Service!")


    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        print(f"Someone disconnected from {FoldService.ALIASES[0]} Service!")

        # supposedly check if something was processing and do something

    def exposed_ping(self) -> str:
        print("Pong!!")
        return f"{FoldService.ALIASES[0]} Pong"

    # takes in a requestObject dictionary, performs train-test on given fold, returns replyObject
    def exposed_train_on_fold(self, requestObject: dict[int, LinearSVC, Any, Any, Any, Any]):      
        # unpack object
        id, default_model, all_data_x, all_data_y, train_index, test_index = requestObject.values()
    
        # split all data into train and test based on given index
        X_train, X_test = all_data_x[train_index], all_data_x[test_index]
        y_train, y_test = all_data_y[train_index], all_data_y[test_index]
                                                                                 
        # fit the model
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)

        # evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        
        print("got accuracy", accuracy)
        
        # return replyObject
        return accuracy




# Go to command prompt and get your ipv4 to start this server..
t = ThreadedServer(service=FoldService, hostname='192.168.137.112', port=1856, auto_register=True, listener_timeout=14)
t.start()

