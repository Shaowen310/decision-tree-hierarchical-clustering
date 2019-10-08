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

    def _fit_recur(self, X, y, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return np.sum(y) / len(X)

        X_T = X.T
        compare_lhs = np.tile(X_T, len(X)).reshape(
            (X.shape[1], X.shape[0], -1))
        compare_rhs = np.reshape(X_T, (X.shape[1], X.shape[0], 1))
        tree_left = compare_lhs <= compare_rhs
        tree_right = compare_lhs > compare_rhs
        y_tile = np.tile(y, X.size).reshape((X.shape[1], X.shape[0], -1))
        tree_left_count = np.sum(tree_left, axis=2)
        tree_right_count = np.sum(tree_right, axis=2)
        tree_left_y_sum = np.sum(tree_left * y_tile, axis=2)
        tree_right_y_sum = np.sum(tree_right * y_tile, axis=2)
        c_hat_left = np.where(tree_left_count > 0,
                              tree_left_y_sum / tree_left_count, 0)
        c_hat_right = np.where(tree_right_count > 0,
                               tree_right_y_sum / tree_right_count, 0)
        mse = (1 - c_hat_left) ** 2 * tree_left_y_sum \
            + c_hat_left ** 2 * (tree_left_count - tree_left_y_sum) \
            + (1 - c_hat_right) ** 2 * tree_right_y_sum \
            + c_hat_right ** 2 * (tree_right_count - tree_right_y_sum)

        s_cand = np.where(np.abs(mse - np.min(mse)) < 1e-5, X_T, np.nan)
        split_0 = np.nonzero(~np.all(np.isnan(s_cand), axis=1))[0][0]
        split = (split_0, np.nanargmin(s_cand[split_0, :]))

        j = split[0]
        s = X_T[split]

        if tree_left_count[split] == len(X):
            return c_hat_left[split]
        # else tree_left_count != len(X):
        return {
            "splitting_variable": j,
            "splitting_threshold": s,
            "left": self._fit_recur(X[tree_left[split], :], y[tree_left[split]], depth+1),
            "right": self._fit_recur(X[tree_right[split], :], y[tree_right[split]], depth+1)
        }

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        # depth-first fit
        self.root = self._fit_recur(X, y, 1)

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = np.empty(len(X))
        for i_x, x in enumerate(X):
            model_node_iter = self.root
            while type(model_node_iter) is dict:
                if x[model_node_iter['splitting_variable']] <= model_node_iter['splitting_threshold']:
                    model_node_iter = model_node_iter['left']
                else:
                    model_node_iter = model_node_iter['right']
            y_pred[i_x] = model_node_iter
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

            y_test_pred = np.genfromtxt(
                "Test_data" + os.sep + "y_pred_decision_tree_" + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")
