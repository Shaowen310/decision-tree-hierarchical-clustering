import numpy as np
import os
import json
import operator


class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _fit_recur(self, X, y, tree_iter, depth):
        # to find argmin (j,s), try all (j,s) and store result in array of shape (j,s)
        if depth > self.max_depth:
            return
        mse = np.zeros(X.shape)

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        # depth-first fit

        pass

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = []
        for x in X:
            model_node_iter = self.root
            while type(model_node_iter) is dict:
                if x[model_node_iter['splitting_variable']] <= model_node_iter['splitting_threshold']:
                    model_node_iter = model_node_iter['left']
                else:
                    model_node_iter = model_node_iter['right']
            y_pred.append(model_node_iter)
        return y_pred

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


def compare_json_dic(json_dic, sample_json_dic):
    if isinstance(json_dic, dict):
        result = 1
        for key in sample_json_dic:
            if key in json_dic:
                result = result * \
                    compare_json_dic(json_dic[key], sample_json_dic[key])
                if result == 0:
                    return 0
            else:
                return 0
        return result
    else:
        rel_error = abs(json_dic - sample_json_dic) / \
            np.maximum(1e-8, abs(sample_json_dic))
        if rel_error <= 1e-5:
            return 1
        else:
            return 0


def compare_predict_output(output, sample_output):
    rel_error = (abs(output - sample_output) /
                 np.maximum(1e-8, abs(sample_output))).mean()
    if rel_error <= 1e-5:
        return 1
    else:
        return 0


# For test
if __name__ == '__main__':
    for i in range(1):
        x_train = np.genfromtxt("Test_data" + os.sep +
                                "x_" + str(i) + ".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep +
                                "y_" + str(i) + ".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(
                max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_dict = tree.get_model_dict()
            y_pred = tree.predict(x_train)

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            # TODO Debugging
            # tree.root = test_model_dict
            # model_dict = tree.get_model_dict()
            # y_pred = tree.predict(x_train)

            y_test_pred = np.genfromtxt(
                "Test_data" + os.sep + "y_pred_decision_tree_" + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")
