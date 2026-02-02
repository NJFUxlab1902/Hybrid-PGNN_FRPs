from __future__ import print_function
import scipy.io as spio
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error

def relu(m):
    m[m < 0] = 0
    return m

def evaluate_physics_loss_lambda(model, uX1, uX2):
    tolerance = 0
    uout1 = model.predict(uX1)
    uout2 = model.predict(uX2)
    udendiff_lambda = (density(uout2) - density(uout1))
    percentage_phy_incon_lambda = np.sum(udendiff_lambda > tolerance)/udendiff_lambda.shape[0]
    phy_loss_lambda = np.mean(relu(udendiff_lambda))
    return phy_loss_lambda, percentage_phy_incon_lambda

def evaluate_physics_loss_hfe(model, uX3, uX4):
    tolerance = 0
    uout3 = model.predict(uX3)
    uout4 = model.predict(uX4)
    udendiff_hfe = (density(uout4) - density(uout3))
    percentage_phy_incon_hfe = np.sum(udendiff_hfe < tolerance) / udendiff_hfe.shape[0]
    phy_loss_hfe = np.mean(relu(udendiff_hfe))
    return phy_loss_hfe, percentage_phy_incon_hfe

# function to compute the room_mean_squared_error given the ground truth (y_true) and the predictions(y_pred)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# function for
def density(temp):
    return temp


def phy_loss_mean_lambda(params):
    # useful for cross-checking training
    udendiff_lambda, lam1 = params

    def phy_loss_lambda(y_true, y_pred):
        return K.mean(K.relu(udendiff_lambda))

    return phy_loss_lambda

def phy_loss_mean_hfe(params):
    # useful for cross-checking training
    udendiff_hfe, lam2 = params

    def phy_loss_hfe(y_true, y_pred):
        return K.mean(K.relu(udendiff_hfe))

    return phy_loss_hfe


# function to calculate the combined loss = sum of rmse and phy based loss
def combined_loss(params):
    udendiff_lambda, udendiff_hfe, lam1, lam2 = params

    def combined_loss_func(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + lam1 * K.mean(K.relu(udendiff_lambda)) + lam2 * K.mean(K.relu(udendiff_hfe))
    return combined_loss_func


def PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, iteration, n_layers, n_nodes, tr_size, lamda1,lamda2,
                    ):
    # Hyper-parameters of the training process
    batch_size =1000
    num_epochs = 1000
    val_frac = 0.1
    patience_val = 500

    # Initializing results filename
    exp_name = optimizer_name + '_drop' + str(drop_frac) + '_usePhy' + str(use_YPhy) + '_nL' + str(
        n_layers) + '_nN' + str(n_nodes) + '_trsize' + str(tr_size) + '_lamda' + str(lamda1) + '_iter' + str(iteration)
    exp_name = exp_name.replace('.', 'pt')
    results_dir = '../results/'
    model_name = results_dir + exp_name + '_model.h5'  # storing the trained model
    results_name = results_dir + exp_name + '_results.mat'  # storing the results of the model

    # Load features (Xc) and target values (Y)
    data_dir = '../datasets/'
    filename = 'shear.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True,
                       variable_names=['Y', 'Xc_doy', 'Modeled_temp'])
    Xc = mat['Xc_doy']
    Y = mat['Y']

    trainX, trainY = Xc[:tr_size, :], Y[:tr_size]
    testX, testY = Xc[tr_size:, :], Y[tr_size:]

    # Loading unsupervised data
    unsup_filename1 = 'shear2.mat'
    unsup_mat1 = spio.loadmat(data_dir + unsup_filename1, squeeze_me=True,
                             variable_names=['Xc_doy1', 'Xc_doy2'])
    unsup_filename2 = 'shear3.mat'
    unsup_mat2 = spio.loadmat(data_dir + unsup_filename2, squeeze_me=True,
                             variable_names=['Xc_doy1', 'Xc_doy2'])

    uX1 = unsup_mat1['Xc_doy1']  # Xc at depth i for every pair of consecutive depth values
    uX2 = unsup_mat1['Xc_doy2']  # Xc at depth i + 1 for every pair of consecutive depth values
    uX3 = unsup_mat2['Xc_doy1']
    uX4 = unsup_mat2['Xc_doy2']

    if use_YPhy == 0:
        # Removing the last column from uX (corresponding to Y_PHY)
        uX1 = uX1[:, :-1]
        uX2 = uX2[:, :-1]
        uX3 = uX3[:, :-1]
        uX4 = uX4[:, :-1]

    # Creating the model
    model = Sequential()
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add(Dense(n_nodes, activation='relu', input_shape=(np.shape(trainX)[1],)))
        else:
            model.add(Dense(n_nodes, activation='relu'))
        model.add(Dropout(drop_frac))
    model.add(Dense(1, activation='linear'))

    # physics-based regularization
    uin1 = K.constant(value=uX1)  # input at depth i
    uin2 = K.constant(value=uX2)  # input at depth i + 1
    uin3 = K.constant(value=uX3)
    uin4 = K.constant(value=uX4)
    lam1 = K.constant(value=lamda1)  # regularization hyper-parameter
    lam2 = K.constant(value=lamda2)

    uout1 = model(uin1)  # model output at depth i
    uout2 = model(uin2)  # model output at depth i + 1
    uout3 = model(uin3)
    uout4 = model(uin4)
    udendiff_lambda = (density(uout2) - density(uout1))  # difference in density estimates at every pair of depth values
    udendiff_hfe = (density(uout3) - density(uout4))

    totloss = combined_loss([udendiff_lambda, lam1, udendiff_hfe, lam2])
    phyloss_lambda = phy_loss_mean_lambda([udendiff_lambda, lam1])
    phyloss_hfe = phy_loss_mean_hfe([udendiff_hfe, lam2])

    model.compile(loss=totloss,
                  optimizer=optimizer_val,
                  metrics=[phyloss_lambda, phyloss_hfe, root_mean_squared_error])

    early_stopping = EarlyStopping(monitor='val_loss_1', patience=patience_val, verbose=1)

    print('Running...' + optimizer_name)
    history = model.fit(trainX, trainY,
    batch_size = batch_size,
    epochs = num_epochs,
    verbose = 1,
    validation_split = val_frac, callbacks = [early_stopping, TerminateOnNaN()])
    #计算totloss
    test_score = model.evaluate(testX, testY, verbose=0)
    phy_cons_lambda, percent_phy_incon_lambda = evaluate_physics_loss_lambda(model, uX1, uX2)
    phy_cons_hfe, percent_phy_incon_hfe = evaluate_physics_loss_hfe(model, uX3, uX4)

    train_predictions = model.predict(trainX)
    test_predictions = model.predict(testX)
    val_predictions_lambda = model.predict(uX1)
    val_predictions_hfe = model.predict(uX3)
    doy_lambda=model.predict(uX1)
    doy_hfe = model.predict(uX3)
    np.savetxt('train_predictions.csv', train_predictions, delimiter=',')
    np.savetxt('test_predictions.csv', test_predictions, delimiter=',')
    np.savetxt('val_predictions_lambda.csv', val_predictions_lambda, delimiter=',')
    np.savetxt('val_predictions_hfe.csv', val_predictions_hfe, delimiter=',')
    np.savetxt('doy_lambda.csv', doy_lambda, delimiter=',')
    np.savetxt('doy_hfe.csv', doy_hfe, delimiter=',')




    print('iter: ' + str(iteration) + ' useYPhy: ' + str(use_YPhy) + ' nL: ' + str(n_layers) + ' nN: ' + str(
        n_nodes) + ' lamda1: ' + str(lamda1) + ' lamda2: ' + str(lamda2) +' trsize: ' + str(tr_size) + ' TestRMSE: ' + str(
        test_score[3]) + ' PhyLoss_lambda: ' + str(test_score[1]) + ' PhyLoss_hfe: ' + str(test_score[2]) + ' combined_loss: ' + str(test_score[0]))
    print(" Physical Consistency lambda = ", phy_cons_lambda)
    print(" Incon lambda = ", percent_phy_incon_lambda)
    print(" Physical Consistency hfe = ", phy_cons_hfe)
    print(" Incon hfe = ", percent_phy_incon_hfe)

    model.save(model_name)
    spio.savemat(results_name, {'train_loss_1': history.history['loss_1'], 'val_loss_1': history.history['val_loss_1'],
                                'train_rmse': history.history['root_mean_squared_error'],
                                'val_rmse': history.history['val_root_mean_squared_error'], 'test_rmse': test_score[2]})


if __name__ == '__main__':
    # Main Function

    # List of optimizers to choose from
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1),
                      RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]

    # selecting the optimizer
    optimizer_num = 3  # Adam
    optimizer_name = optimizer_names[optimizer_num]
    optimizer_val = optimizer_vals[optimizer_num]

    # Selecting Other Hyper-parameters
    drop_frac = 0  # Fraction of nodes to be dropped out
    use_YPhy = 1  # Whether YPhy is used as another feature in the NN model or not
    n_layers = 2  # Number of hidden layers
    n_nodes = 12  # Number of nodes per hidden layer

    # set lamda=0 for pgnn0
    lamda1 = 0.6 # Physics-based regularization constant
    lamda2 = 0.5

    # Iterating over different training fractions and splitting indices for train-test splits
    trsize_range = [160, 2500, 1000, 5000, 10000]
    iter_range = np.arange(1)  # range of iteration numbers for random initialization of NN parameters

    # default training size = 5000
    tr_size = trsize_range[0]

    # List of lakes to choose from

    # iterating through all possible params
    for iteration in iter_range:
        PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, iteration, n_layers, n_nodes, tr_size,
                        lamda1, lamda2)


