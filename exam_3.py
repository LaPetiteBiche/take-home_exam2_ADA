import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
# from ml_model import DNN, RNN
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
from sklearn.metrics import auc, confusion_matrix

# Because all code should be in exam_3.py, I copy pasted the DNN and RNN class from ml_model -> ex begins line 200
class DNN:
    def __init__(self):
        self.model = None
        self.hist_training = None

    def create_model(
            self,
            data,
            learning_rate=0.005,
            opti='adam',  # you can put 'adam', 'sgd', or 'rms_prop'
            batch_normalization=False,
            activation='relu',  # 'sigmoid', or 'relu' are the main two, but others are coded in keras (see documentation)
            architecture=[64, 32, 16],  # each number correspond to a number of neurons in a layer
            drop_out=0.0,
            verbose=0):

        # check that the input are correct
        assert type(architecture) == list, 'architecture must be a list of integers (e.g. [64,32,16], each representing a number of neurones in a layer'
        assert opti in ['adam', 'sigmoid', 'rms_prop'], "invalid optimizer, please chose among ['adam','sigmoid','rms_prop']"

        # extract the input shape and output shape from the data

        # select the optimizer
        if opti == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate)
        if opti == 'rms_prop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        if opti == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)

        output_dim = data.y_te.shape[1]
        model = self._create_network(data, architecture, batch_normalization, activation, drop_out)

        ##################
        # define an additional metrics
        ##################
        if output_dim == 1:
            # here we do a regression---that, is we predict one continous thing, hence we can add the R squared as a metric
            def r_square(y_true, y_pred):
                SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
                SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
                return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

            # we have three metrics:
            # mean absolute error -> mae
            # mean square error -> mse
            # and our custom r-square.
            model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse', r_square])
        if output_dim > 1:
            # here we do a classifier ---that, is we predict a percentage of being in one dimension instead of another
            # we have two  metrics:
            # mean absolute error -> mae
            # accracy
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=optimizer, metrics=['accuracy', 'mae'])
        if verbose:
            # if specified, print the network architecture
            print(model.summary())
        self.model = model

    def _create_network(self, data, architecture, batch_normalization, activation, drop_out):
        input_dim = data.X_te.shape[1]
        output_dim = data.y_te.shape[1]
        # build the network
        L = []
        for i, l in enumerate(architecture):
            if i == 0:
                L.append(tf.keras.layers.Dense(l, activation=activation, input_shape=[input_dim]))  # add the first layer
            else:
                L.append(tf.keras.layers.Dense(l, activation=activation))  # add a layer
            if drop_out > 0.0:
                L.append(tf.keras.layers.Dropout(rate=drop_out, seed=12345))

            if batch_normalization:
                # add batch normalization if specified
                L.append(tf.keras.layers.BatchNormalization())

        # add the final layer
        if output_dim == 1:
            L.append(tf.keras.layers.Dense(output_dim))
        else:
            # if we are doing classification, we wish to normalize the ouput between 0 and 1, hence the softmax
            L.append(tf.keras.layers.Dense(output_dim, activation='softmax'))

        # keras command to build a simple feed forward network with the parameters defined above
        model = tf.keras.Sequential(L)
        return model

    def show_performance(self, label_, data):


        # check the output_dim to calibrate the plot
        output_dim = data.y_te.shape[1]

        # use the pandas function to start the plot (history training is a PD data frame).
        self.hist_training[['loss', 'val_loss']].plot()
        plt.grid(True)  # add a grid for visibiliy
        plt.xlabel('epochs')
        if output_dim == 1:
            plt.ylabel('mean absolute error')  # if its a regression, we plot the mae
        else:
            plt.ylabel('cross entropy loss')  # if its a classification, we plot the cross entropy loss
        plt.title(label_)
        plt.savefig(label_ + '.png')
        plt.show()

        if output_dim == 1:
            # if regression, compute the out of sample performance measure
            print('=' * 50)
            print('Out of sample performance:')
            self.model.evaluate(data.X_te, data.y_te, verbose=2)
            print('=' * 50)

        else:
            # if classification  print the  confusion matrix
            pred = self.model.predict(data.X_te)
            cf = confusion_matrix(y_true=np.argmax(data.y_te, 1), y_pred=np.argmax(pred, 1))
            cf = pd.DataFrame(cf)
            index = ['True ' + str(x) for x in cf.index]
            col = ['Predicted ' + str(x) for x in cf.columns]
            cf.index = index
            cf.columns = col
            print(cf)

    def train_model(self, data, epoch=10, bs=256, verbose=0, tensor_board_name=None):
        tf.random.set_seed(1234)
        np.random.seed(1234)
        print('### start training for', epoch, 'epochs')
        # Prepare the validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((data.X_va, data.y_va))
        val_dataset = val_dataset.batch(256)

        if tensor_board_name is not None:
            # set_up the tensorboard name

            log_dir = "logs/" + tensor_board_name  # +"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            history_training = self.model.fit(x=data.X_tr, y=data.y_tr, batch_size=bs, epochs=epoch, validation_data=val_dataset, verbose=verbose, callbacks=tensorboard_callback)

        else:
            # the keras command to launch the training routine
            history_training = self.model.fit(x=data.X_tr, y=data.y_tr, batch_size=bs, epochs=epoch, validation_data=val_dataset, verbose=verbose)
        print('### training finish \n')

        # return the history of training process
        self.hist_training = pd.DataFrame(history_training.history)



class RNN(DNN):
    def __init__(self):
        super().__init__()
        self.model = None
        self.hist_training = None

    def _create_network(self, data, architecture, batch_normalization,activation, drop_out):

        output_dim = data.y_te.shape[1]
        # build the network
        L = [tf.keras.layers.Input(shape=(data.X_te.shape[1],data.X_te.shape[2]))] # very important input layer

        for i, l in enumerate(architecture):
            if i == 0:
                L.append(tf.keras.layers.LSTM(l)) # first layer is now an SLTM
            else:
                L.append(tf.keras.layers.Dense(l, activation=activation))  # add a layer
            if drop_out > 0.0:
                L.append(tf.keras.layers.Dropout(rate=drop_out, seed=12345))

            if batch_normalization:
                # add batch normalization if specified
                L.append(tf.keras.layers.BatchNormalization())


        # add the final layer
        if output_dim == 1:
            L.append(tf.keras.layers.Dense(output_dim))
        else:
            # if we are doing classification, we wish to normalize the ouput between 0 and 1, hence the softmax
            L.append(tf.keras.layers.Dense(output_dim, activation='softmax'))

        # keras command to build a simple feed forward network with the parameters defined above
        model = tf.keras.Sequential(L)
        return model


# a)
# Data
file = 'dat_ex/otp.csv'
df = pd.read_csv(file)
df = df.dropna()
# Dummy
df['y'] = (df['status']=='On Time')*1

# Hour
df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='%Y-%m-%d %H:%M:%S')
df['hour'] = df['timeStamp'].dt.hour

# x - hours
x = []
for i in range(24):
    x.append(i)
# y - % obs by hours
y = []
for i in range(24):
    df2 = df[df["hour"] == i]
    y.append(df2['y'].sum()/df2.shape[0])

# Mean line
def mean(lst):
    return sum(lst) / len(lst)
avg = []
for i in range(24):
    avg.append(mean(y))

# Plot
plt.bar(x, y)
plt.plot(x, avg, color='red', label='mean')
plt.xlabel("Hour")
plt.ylabel("%OnTime")
plt.legend()
plt.savefig(r'res/ex3_a.png')
plt.close()
plt.savefig(sys.stdout.buffer)

# Answer 3_a_i
your_answer_part_i = "We can see on the plot that between 11pm to 5am, the trains are more likely to be on time, so the hours of the day contains information about the probability that a train is on time."

# Answer 3_a_ii
your_answer_part_ii = "From a rational thinking, a train is most likely on time when there are few trains and few passengers. During the hours when this happend, a train is less likely to be disturbed by another train or to be delayed by the entry of the passengers."

print('Answer 3_a_i)',your_answer_part_i)
print('Answer 3_a_ii)',your_answer_part_ii)

# b)
# Class for data, load_y = question b, load_y2 = question c, load_y3 = question d
class Data:
    def __init__(self):
        self.X_tr = None
        self.X_te = None
        self.X_va = None

        self.y_tr = None
        self.y_te = None
        self.y_va = None

    # 10 lags, only y values
    def load_y(self, LAG=10):
        df2 = df[["y"]]
        y=df2[['y']].values
        X = []
        for i in range(LAG+1,0-1,-1):
            if i > 0:
                X.append(y[LAG+1 - i:-i])
            else:
                X.append(y[LAG+1 - i:])

        X = np.concatenate(X, 1)
        y = X[:,-1].reshape(-1,1)
        X = X[:,:-1]
        X = X.reshape((X.shape[0], X.shape[1], 1)) 

        ind = np.arange(0, y.shape[0], 1)
        tr = int(np.ceil(len(ind) * 0.8))
        te = int(np.ceil(len(ind) * 0.9))

        self.X_tr = X[np.where(ind[:tr])[0], :,:]
        self.X_te = X[np.where(ind[tr:te])[0], :,:]
        self.X_va = X[np.where(ind[te:])[0], :,:]

        self.y_tr = y[np.where(ind[:tr])[0], :]
        self.y_te = y[np.where(ind[tr:te])[0], :]
        self.y_va = y[np.where(ind[te:])[0], :]
    
    # 10 lags, y and hour
    def load_y2(self, LAG=10):
        df2 = df[["y", "hour"]]
        y=df2[['y','hour']].values
        X = []
        for i in range(LAG+1,0-1,-1):
            if i > 0:
                X.append(y[LAG+1 - i:-i])
            else:
                X.append(y[LAG+1 - i:])

        X = np.concatenate(X, 1)
        y = X[:,-1].reshape(-1,1)
        X = X[:,:-1]
        X = X.reshape((X.shape[0], X.shape[1], 1)) 

        ind = np.arange(0, y.shape[0], 1)
        tr = int(np.ceil(len(ind) * 0.8))
        te = int(np.ceil(len(ind) * 0.9))

        self.X_tr = X[np.where(ind[:tr])[0], :,:]
        self.X_te = X[np.where(ind[tr:te])[0], :,:]
        self.X_va = X[np.where(ind[te:])[0], :,:]

        self.y_tr = y[np.where(ind[:tr])[0], :]
        self.y_te = y[np.where(ind[tr:te])[0], :]
        self.y_va = y[np.where(ind[te:])[0], :]

    def load_y3(self):
        # Get each origin place + reduce to 10k obs
        df2 = df.head(10000)
        origin = df2.origin.unique()
        # Create dummy column for each origin
        for i in df2 :
            df2[i] = (df2['origin']==i)*1

        # Same for next destination
        next_station = df2.next_station.unique()
        for i in next_station :
            df2[i] = (df2['next_station']==i)*1
        # Place timeStamp at the end
        df2['timeStamp_last'] = df2['timeStamp']
        # Select the last columns we just created as X
        X = df2[df2.columns[-344:]].values
        y = df2[['y']].values

        ind = np.arange(0, y.shape[0], 1)
        np.random.shuffle(ind)

        tr = int(np.ceil(len(ind) * 0.8))
        te = int(np.ceil(len(ind) * 0.9))

        self.X_tr = X[np.where(ind[:tr])[0], :]
        self.X_te = X[np.where(ind[tr:te])[0], :]
        self.X_va = X[np.where(ind[te:])[0], :]

        self.y_tr = y[np.where(ind[:tr])[0], :]
        self.y_te = y[np.where(ind[tr:te])[0], :]
        self.y_va = y[np.where(ind[te:])[0], :]

# Data and Shape
self = Data()
data = Data()
data.load_y(10)
X_training_shape = data.X_tr.shape
print('EX3b) shape:',X_training_shape)

# Model
model=RNN()
model.create_model(data,architecture=[10])
model.train_model(data,verbose=1,epoch=1)
pred=model.model(data.X_te)

# Confusion Matrix - not working so well
# Reg
model.show_performance("test", data = data)
# Matrice 
cf = confusion_matrix(y_true=np.argmax(data.y_te, 1), y_pred=np.argmax(pred, 1))
cf = pd.DataFrame(cf)
index = ['True ' + str(x) for x in cf.index]
col = ['Predicted ' + str(x) for x in cf.columns]
cf.index = index
cf.columns = col
confusion_matrix1 = cf
print("EX3b) confusion: '\n'", confusion_matrix1)

# c)
# Data and Shape
data2 = Data()
data2.load_y2(10)
X_training_shape = data2.X_tr.shape
print('EX3c) shape:',X_training_shape)

# Model
model2=RNN()
model2.create_model(data2,architecture=[10])
model2.train_model(data2,verbose=1,epoch=1)
pred=model2.model(data2.X_te)

# Confusion Matrix - not working so well
# Reg
model2.show_performance("test", data = data2)
# Matrice
pred = model2.model.predict(data2.X_te)
cf2 = confusion_matrix(y_true=np.argmax(data2.y_te, 1), y_pred=np.argmax(pred, 1))
cf2 = pd.DataFrame(cf2)
index = ['True ' + str(x) for x in cf2.index]
col = ['Predicted ' + str(x) for x in cf2.columns]
cf2.index = index
cf2.columns = col
confusion_matrix2 = cf2
print("EX3c) confusion: '\n'", confusion_matrix2)

# d)
# Data
data3 = Data()
data3.load_y3()

# Model -> Relu activation and 64, 32, 16 neurons specified in ml_model.py under activation and architecture
model3=DNN()
model3.create_model(data3)
model3.train_model(data3)

# Confusion matrix
model3.show_performance("test", data = data3)
pred = model3.model.predict(data3.X_te)
cf3 = confusion_matrix(y_true=np.argmax(data3.y_te, 1), y_pred=np.argmax(pred, 1))
cf3 = pd.DataFrame(cf3)
index = ['True ' + str(x) for x in cf3.index]
col = ['Predicted ' + str(x) for x in cf3.columns]
cf3.index = index
cf3.columns = col
confusion_matrix3 = cf3
print("EX3d) confusion: '\n'", confusion_matrix3)

# e) 
answer_e = "If we want to maximize the recall, we could set up a threshold so that we reach a recall for each model equal to 1. However, this will come a the cost of losing precision. Our model will return all the relevant results, but will return a lot of irrelevant ones too."
print("EX3e) answer: '\n'", answer_e)