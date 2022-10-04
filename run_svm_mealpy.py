#!/usr/bin/env python
# Created by "Thieu" at 05:52, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.preprocessing import LabelEncoder
from src.timeseries_svr import TimeSeriesSVR
from src.utils import ts_util
from mealpy.evolutionary_based import FPA
from permetrics.regression import RegressionMetric


if __name__ == "__main__":

    list_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_encoder = LabelEncoder()
    kernel_encoder.fit(list_kernels)

    data = ts_util.generate_data()
    data["KERNEL_ENCODER"] = kernel_encoder

    # x1. Kernel: [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]
    # x2. C: float [0.1 to 1000.0]
    # x3. Epsilon: float [0.01 to 10.0]

    LB = [0, 0.1, 0.01]
    UB = [3.99, 1000.0, 10.0]

    problem = TimeSeriesSVR(lb=LB, ub=UB, minmax="min", data=data, save_population=False, log_to="console")

    algorithm = FPA.OriginalFPA(epoch=2, pop_size=20)
    best_position, best_fitness = algorithm.solve(problem)

    best_solution = problem.decode_solution(best_position)

    print(f"Best fitness (MSE) value: {best_fitness}")
    print(f"Best parameters: {best_solution}")

    ###### Get the best tuned neural network to predict test set
    best_network = problem.generate_trained_model(best_solution)
    y_pred = best_network.predict(data["X_test"])

    evaluator = RegressionMetric(data["y_test"], y_pred, decimal=6)
    print(evaluator.get_metrics_by_list_names(["MAE", "RMSE", "MAPE", "R2"]))
