#!/usr/bin/env python
# Created by "Thieu" at 04:51, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

# 1. Fitness function
# 2. Lower bound and upper bound of variables
# 3. Min or max


# Assumption that we are trying to optimize the multi-layer perceptron with 3 layer 1 input, 1 hidden, 1 output.
# 1. Batch-size training
# 2. Epoch training
# 3. Optimizer
# 4. Learning rate
# 5. network weight initialization
# 6. activation functions
# 7. number of hidden units

# Rules:
# x1. Batch-size: [ 2, 4, 8 ]
# x2. Epoch : [700, 800, .... 2000]
# x3. Optimizer: ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# x4. Learning rate: [0.01 -> 0.5]       real number
# x5. network weight initialization: ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# x6. activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# x7. hidden units: [5, 100] --> integer number

# solution = vector of float number = [ x1, x2, x3, x4, x5, x6, x7 ]

# x3, x5, x6: need LabelEncoder to convert string into integer number, need to use int function
# x1, x2, x7: is integer number ---> need to use int function
# x4: is float number


# univariate MLP example
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from permetrics.regression import RegressionMetric
from mealpy.utils.problem import Problem


class TimeSeriesMLP(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="TimeSeries Multi-Layer Perceptron", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def decode_solution(self, solution):
        # batch_size = 2**int(solution[0])
        # # 1 -> 1.99 ==> 1
        # # 2 -> 2.99 ==> 2
        # # 3 -> 3.99 ==> 3
        #
        # epoch = 10 * int(solution[1])
        # # 10 * 70 = 700
        # # 10 * 200 = 2000
        #
        # opt_integer = int(solution[2])
        # opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
        # # 0 - 0.99 ==> 0 index ==> should be SGD (for example)
        # # 1 - 1.99 ==> 1 index ==> should be RMSProp
        #
        # learning_rate = solution[3]
        #
        # network_weight_initial_integer = int(solution[4])
        # network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
        #
        # act_integer = int(solution[5])
        # activation = ACT_ENCODER.inverse_transform([act_integer])[0]
        #
        # n_hidden_units = int(solution[6])

        batch_size = 2 ** int(solution[0])
        epoch = 10 * int(solution[1])
        opt_integer = int(solution[2])
        opt = self.data["OPT_ENCODER"].inverse_transform([opt_integer])[0]
        learning_rate = solution[3]
        network_weight_initial_integer = int(solution[4])
        network_weight_initial = self.data["NWI_ENCODER"].inverse_transform([network_weight_initial_integer])[0]
        act_integer = int(solution[5])
        activation = self.data["ACT_ENCODER"].inverse_transform([act_integer])[0]
        n_hidden_units = int(solution[6])
        return {
            "batch_size": batch_size,
            "epoch": epoch,
            "opt": opt,
            "learning_rate": learning_rate,
            "network_weight_initial": network_weight_initial,
            "activation": activation,
            "n_hidden_units": n_hidden_units,
        }

    def generate_trained_model(self, structure):
        # define model
        model = Sequential()
        model.add(Dense(structure["n_hidden_units"], activation=structure["activation"],
                        input_dim=self.data["n_steps"], kernel_initializer=structure["network_weight_initial"]))
        model.add(Dense(1))

        # Compile model
        optimizer = getattr(optimizers, structure["opt"])(learning_rate=structure["learning_rate"])
        model.compile(optimizer=optimizer, loss='mse')

        # fit model
        model.fit(self.data["X_train"], self.data["y_train"], epochs=structure["epoch"], batch_size=structure["batch_size"], verbose=2)

        return model

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        y_pred = model.predict(self.data["X_test"])

        evaluator = RegressionMetric(self.data["y_test"], y_pred, decimal=6)
        return evaluator.mean_squared_error()

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        return fitness
