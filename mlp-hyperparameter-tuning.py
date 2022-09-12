#!/usr/bin/env python
# Created by "Thieu" at 04:17, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.preprocessing import LabelEncoder
from models.timeseries_mlp import TimeSeriesMLP
from models.utils import ts_util
from mealpy.evolutionary_based import FPA
from permetrics.regression import RegressionMetric


if __name__ == "__main__":

    list_optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    list_network_weight_initials = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    list_activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

    # LABEL ENCODER
    opt_encoder = LabelEncoder()
    opt_encoder.fit(list_optimizers)  # domain range ==> 7 values

    nwi_encoder = LabelEncoder()
    nwi_encoder.fit(list_network_weight_initials)

    act_encoder = LabelEncoder()
    act_encoder.fit(list_activations)

    data = ts_util.generate_data()
    data["OPT_ENCODER"] = opt_encoder
    data["NWI_ENCODER"] = nwi_encoder
    data["ACT_ENCODER"] = act_encoder

    LB = [1, 5, 0, 0.01, 0, 0, 5]
    UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 50]

    problem = TimeSeriesMLP(lb=LB, ub=UB, minmax="min", data=data, save_population=False, log_to="console")

    algorithm = FPA.OriginalFPA(epoch=5, pop_size=20)
    algorithm.solve(problem)

    best_solution = problem.decode_solution(algorithm.solution[0])

    print(f"Best fitness (MSE) value: {algorithm.solution[1]}")
    print(f"Best parameters: {best_solution}")

    ###### Get the best tuned neural network to predict test set
    best_network = problem.generate_trained_model(best_solution)
    y_pred = best_network.predict(data["X_test"])

    evaluator = RegressionMetric(data["y_test"], y_pred, decimal=6)
    print(evaluator.get_metrics_by_list_names(["MAE", "RMSE", "MAPE", "R2"]))
