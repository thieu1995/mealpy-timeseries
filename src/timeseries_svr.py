#!/usr/bin/env python
# Created by "Thieu" at 05:33, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

# Assumption that we are trying to optimize the hyper-parameter of SVR model
# 1. Kernel
# 2. C
# 3. Epsilon


# Rules:
# x1. Kernel: [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]
# x2. C: float [0.1 to 1000.0]
# x3. Epsilon: float [0.01 to 10.0]

## kernel = 'precomputed', will show the error with dataset, better to not use it.


# solution = vector of float number = [ x1, x2, x3 ]

# x1: need LabelEncoder to convert string into integer number, need to use int function
# x2, x3: is float number


# univariate SVR example
from sklearn.svm import SVR
from permetrics.regression import RegressionMetric
from mealpy.utils.problem import Problem


class TimeSeriesSVR(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="TimeSeries Support Vector Regression", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def decode_solution(self, solution):
        # kernel_integer = int(solution[0])
        # kernel = KERNEL_ENCODER.inverse_transform([kernel_integer])[0]
        # 0 - 0.99 ==> 0 index ==> should be linear (for example)
        # 1 - 1.99 ==> 1 index ==> should be poly
        #
        # C = solution[1]
        # epsilon = solution[2]
        
        kernel_integer = int(solution[0])
        kernel = self.data["KERNEL_ENCODER"].inverse_transform([kernel_integer])[0]
        C, epsilon = solution[1], solution[2]
        return {
            "kernel": kernel,
            "C": C,
            "epsilon": epsilon,
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        model = SVR(kernel=structure["kernel"], C=structure["C"], epsilon=structure["epsilon"])
        model.fit(self.data["X_train"], self.data["y_train"])
        # print("Return model")
        return model

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        y_pred = model.predict(self.data["X_test"])

        evaluator = RegressionMetric(self.data["y_test"], y_pred, decimal=6)
        loss = evaluator.mean_squared_error()
        return loss

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        return fitness
