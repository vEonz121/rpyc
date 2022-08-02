<h3 align="left">Welcome to</h3>
<h1 align="center">BitFold: A Distributed K-Fold Cross Validation Process Flow</h1>
<h3 align="right">using RPYC</h3>

## 📑 Description

BitFold is a peer-to-peer distributed computing approach on the k-fold cross validation process for support vector machine models. BitFold runs on three entities which are the **Client**, **Server**, and **Registry**.

1. **Clients** - Query the registry for available servers and distribute each fold to every available server within the network to cross validate. The results of which, are to be returned as an array of accuracy scores across all folds
2. **Servers** - Provide computational resources for the network to perform cross validation on a fold distributed to them from other clients
3. **Registry** (_Middleware_) - Keeps track of each server within the peer-to-peer network and provide the list of available servers for clients to use

This process of distributed k-fold cross validation is performed sequentially on each fold. This means that each fold is distributed, split, and trined one after another once the client calls initiates the process.

Subject: **Distributed & Parallel Computing**
Lecturer: **Ts. Nazleeni Samiha Haron**

## 🖼 Contents

- [📑 Description](#-description)
- [🖼 Contents](#-contents)
- [😏​ How It Works](#-how-it-works)
  - [Clients `Client.py`](#clients-clientpy)
  - [Servers `Service.py`](#servers-servicepy)
  - [Registry](#registry)
- [🏁 Versions](#-versions)
- [🚀 Quick Start](#-quick-start)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [🗿 Before You Start Working...](#-before-you-start-working)
- [🤓 Recommendations](#-recommendations)
  <br/>

## 😏​ How It Works

#### Clients `Client.py`

**Initializing the FoldDistributor Client**

```py
class FoldDistributor:
  def __init__(self, nFolds):
    print(f"List of Found Services: {r.list_services(timeout=10)}")
    if len(r.list_services(timeout=10)) < nFolds:
      raise Exception("Not enough services available.")

    self.nFolds = nFolds
    self.establish_connection()
```

`FoldDistributor` is the class which is initiated as the client.

On initialization, the `__init__` method is called and the number of folds requested is passed. The client will then attempt to retrieve the list of available services from the registry. If there is no reply within 10 seconds, the client will timeout. If the number of listed servers within the network does not meet the number of folds requested, the client will raise a `Not enough services available` exception. Otherwise, the client will initialize itself with the number of folds requested, `nFolds` as its number of folds to be cross validated, `self.nFolds` and calls the method `establish_connection`.

```py
def establish_connection(self):
  self.services = [r.connect_by_service(f"FOLD{k}", config = {"allow_public_attrs" : True}).root for k in range(self.nFolds)]
```

`establish_connection` will then attempt to connect to servers within the network relative to the number of `nFolds` set for the client (i.e. 3 servers for 3 folds).

```py
# Import the FoldDistributor
from Client import FoldDistributor

# Initialize the FoldDistributor with nFolds of 2
skf = FoldDistributor(2)
```

Output: `List of Found Services: ('FOLD0', 'FOLD1')`

Here, the `FoldDistributor` class is imported from the `Client.py` file and is initialized with `nFolds` of 3. The initialization should return the list of servers which it is connected to. Then, the `ping_services` method is called to ping the available servers listed within the network. The output of the above code snippet should return as such:

**Pinging Servers**

```py
def ping_services(self):
  successful = 0
  for s in self.services:
    if s.exposed_ping(): successful += 1

  return f"There are {successful} out of {len(self.services)} services responded to the ping."
```

The client can then ping the servers it is connected to using the `ping_services` method which can be used to gauge the health of the servers which it is connected to.

```py
# Ping Connected Servers
skf.ping_services()
```

Client Output: `'There are 2 out of 2 services responded to the ping.'`

**Distributing and Processing Folds**

```py
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
      "all_data_x": data.to_numpy(),
      "all_data_y": target.to_numpy(),
      "train_index": train_i,
      "test_index": test_i
    }
    # send request Object and "await" reply
    replyObj = self.services[i].exposed_train_on_fold(requestObj)
    print(f"Fold {i} complete")
    # consolidate to results
    results.append(replyObj)

  return results
```

`distribute_folds` is then called to initiate the distribution of folds to the available servers, passing the data, target, and model. The data should be a `pd.DataFrame`, target being a `pd.Series`, and the model being a `LinearSVC`. This method should then return a list of strings which is the `replyObj` returned from the `exposed_train_on_fold` method called on each server. This should be a string which is the accuracy score of the model trained on their respective folds appended to the results list. This list of strings would then be returned to the client after the entire fold iteration process has completed.

```py
accuracy = skf.distribute_folds(data, target, best_model)
print(accuracy)
```

Client Output:

```
Fold 0 complete
Fold 1 complete
[0.9639308060360692, 0.9642988590357011]
```

#### Servers `Service.py`

**Running the Server**
The server can be run by simply running the `Service.py` file and entering the fold number of the service.

e.g: `python Service.py`

**On Server Run**

```py
fold = None
while True:
  fold = input("Fold Number: ")
  if int(fold) < 0:
      print("Please enter a valid integer of more than 0")
  else: break
```

Upon running the service, a fold number will be requested to be entered from the user. This fold number would be used to identify the service within the network.

```py
ALIASES = [f'FOLD{int(fold)}']
```

Server Output: `Fold Number: 2`

```py
from rpyc.utils.server import ThreadedServer

t = ThreadedServer(service=FoldService, hostname='localhost', port=1856, auto_register=True, listener_timeout=14)
t.start()
```

Then, a `ThreadedServer` is run with the following configuration. The `FoldService` class is passed to the service parameter.

**On Client Connect**

```py
def on_connect(self, conn):
  print(f"Someone connected to {FoldService.ALIASES[0]} Service!")
```

When a client connects to a service, the service will print a message.

Server Output: `Someone connected to FOLD2 Service!`

**On Client Disconnect**

```py
def on_disconnect(self, conn):
  print(f"Someone disconnected from {FoldService.ALIASES[0]} Service!")
```

When a client disconnects to a service, the service will print a message.

Server Output: `Someone disconnected to FOLD2 Service!`

**Responding to Pings**

```py
def exposed_ping(self) -> str:
  print("Pong!!")
  return f"{FoldService.ALIASES[0]} Pong"
```

When a client invokes the `exposed_ping` method, the service will print `"Pong!!"` on the server while returning a message containing its alias with the word pong.

Server Output: `Pong!!`
Client Output: `FOLD2 Pong`

**Validating a Fold**

```py
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
```

When the client invokes the `exposed_train_on_fold` method with the fold passed as the `requestObject`, the server begins performing the entire process of validation on the given fold. This involves splitting the data provided into training and testing sets, training the default model on the split data, testing the model prediction, and calculating the accuracy score of the trained model. This accuracy score is then printed out by the server and is passed back to the client to be appended to the results of the entire K-Fold cross validation process.

Server Output: `got accuracy 0.9639308060360692`

#### Registry

The Registry is a default component of RPyC library and is created with reference to the [RPyC documentation](https://rpyc.readthedocs.io/en/latest/api/utils_registry.html).
<br/>

## 🏁 Versions

- 0.0.1 - **_(HELLO WORLD!)_** Initial build
- 0.1.0 - **_GOT IT WORKING_** Working Build 0.1.0 
  <br/>

## 🚀 Quick Start

### Requirements

- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/)

### Installation

1. Clone the repo to a desired directory

```
git clone https://github.com/vEonz121/rpyc
```

3. Create a python virtual environment for the project directory

```
python -m venv env
```

4. Activate the virtual environment

```
.\env\Scripts\activate
```

5. Install the requirements from `requirements.txt`

```
python -m pip install -r requirements.txt
```

### ...and you can start working!

<br/>

## 🗿 Before You Start Working...

Please create a branch with the following naming scheme:

```
[type of change]/[name of change]
```

### Type of Changes:

- `feat` = Feature or new function
- `fix` = Bug or config fix
- `docs` = Documentation changes
- `test` = For testing purposes
- `perf` = Optimization changes which affect performance
- `chore` = Menial tasks that don't do much but are necessary

### Name of Change:

- All lowercase with dashes for spaces: i.e. `this-is-a-branch-name`

### Example: `feat/new-feature`

#### DO NOT COMMIT CHANGES DIRECTLY ON THE MASTER BRANCH UNLESS YOU KNOW WHAT YOU ARE DOING

<br/>

## 🤓 Recommendations

### Here is a list of recommended extensions used in this repository:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [autoDocstring - Python Docsstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
- [Python Environment Manager](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-environment-manager)
- [Prettier - Code formatter](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
- [Git Graph](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph)
- [GitLens — Git supercharged](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
- [Live Share](https://marketplace.visualstudio.com/items?itemName=MS-vsliveshare.vsliveshare)
- [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
- [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments)
- [TODO Tree](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree)
- [Color Highlight](https://marketplace.visualstudio.com/items?itemName=naumovs.color-highlight)
- [Material Icon Theme](https://marketplace.visualstudio.com/items?itemName=PKief.material-icon-theme)

> Yes, these are optional. But it makes eveyone more organized if you use them. 🙂

<br/>

[Back to Top](#welcome-to)
