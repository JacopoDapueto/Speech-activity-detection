import numpy as np
import time
import matplotlib.pyplot as plt
from SpeechDetection.NeuralNetwork.Network import Network
from Data.LoadData import LoadData
from Data.Standardization import Standardization

# Optimizers    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, RMSprop

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
adad = Adadelta(lr=1.0,rho=0.95,epsilon=None,decay=0.0)
adag = Adagrad(lr=0.01,epsilon=None,decay=0.0)
adamax = Adamax(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)
nadam = Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,schedule_decay=0.004)
rms = RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0)


# Losses    https://keras.io/losses/
loss = ['binary_crossentropy','mean_squared_error','mean_absolute_error','categorical_hinge']

optimizer = [(adam, "Adam"), (sgd, "SGD"), (adad, "Adadelta"), (adag, "Adagrad"), (adamax, "Adamax"), (nadam, "Nadam"), (rms, "RMSprop")]

train_error = np.zeros((len(loss), len(optimizer)))
test_error = np.zeros((len(loss), len(optimizer)))
time_ex = np.zeros((len(loss), len(optimizer)))


print("Loading data...")
X_train, X_test, y_train, y_test = LoadData().getTrainAndTest()

sd = Standardization()
sd.calculateMeanStd(X_train)
X_train = sd.standardizeData(X_train)
X_test = sd.standardizeData(X_test)

i = 0
for l in loss:
  j=0
  for o, o_name in optimizer:
    print("Optimizer: {} , Loss: {}".format(o_name, l))
    # create model
    model= Network(input_shape=17, optimizer=o, loss=l, metrics='accuracy')
    model.createModel()
    # compile model
    model.compileModel()
    # fit model
    start = time.time()
    history = model.trainModel(X_train, y_train)
    end = time.time()

    #print("Fit time: ", end - start)
    train_acc = model.testModel(X_train, y_train)
    test_acc = model.testModel(X_test, y_test)
    #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    train_error[i,j] = train_acc
    test_error[i,j] = test_acc
    time_ex[i,j] = end - start
    j +=1
    print(" ")
  i +=1

plt.figure(figsize=(15, 10))


def autolabel(rects, unit):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        # print(type(height))
        plt.annotate('{0:2.2f}'.format(float(height)) + unit,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# plotting results with respect to the optimizer
bar_width = 0.55
opacity = 0.8
index = np.arange(len(optimizer))

train_b = plt.bar(2 * index - bar_width, np.mean(train_error * 100, axis=0), bar_width, alpha=opacity, color='b', label='train accuracy')

test_b = plt.bar(2 * index, np.mean(test_error * 100, axis=0), bar_width, alpha=opacity, color='g', label='test accuracy')

time_b = plt.bar(2 * index + bar_width, np.mean(time_ex, axis=0), bar_width, alpha=opacity, color='r', label='time execution')

plt.xlabel('optimizer')
plt.ylabel('Scores')
plt.title('means wrt optimizer')
plt.xticks(2 * index + bar_width, [o[1] for o in optimizer])
plt.legend()

autolabel(train_b, '%')
autolabel(test_b, '%')
autolabel(time_b, 'sec')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

# plotting results with respect to the loss
bar_width = 0.55
opacity = 0.8
index = np.arange(len(loss))

train_b = plt.bar(2 * index - bar_width, np.mean(train_error * 100, axis=1), bar_width, alpha=opacity, color='b',label='train accuracy')

test_b = plt.bar(2 * index, np.mean(test_error * 100, axis=1), bar_width, alpha=opacity, color='g', label='test accuracy')

time_b = plt.bar(2 * index + bar_width, np.mean(time_ex, axis=1), bar_width, alpha=opacity, color='r', label='time execution')

plt.xlabel('optimizer')
plt.ylabel('Scores')
plt.title('means wrt loss')
plt.xticks(2 * index + bar_width, [l for l in loss])
plt.legend()

autolabel(train_b, '%')
autolabel(test_b, '%')
autolabel(time_b, 'sec')

plt.tight_layout()
plt.show()