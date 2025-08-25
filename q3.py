import numpy as np
import pandas as pd
import math
from math import log
from scipy.stats import chi2
import random

file_name = "flightdelay.csv"
buckets = 2

dep_time_bucket = {
    '0600-0659': 1, '0700-0759': 1, '0800-0859': 1, '0900-0959': 1, '1000-1059': 1, '1100-1159': 1,
    '1200-1259': 2, '1300-1359': 2, '1400-1459': 2, '1500-1559': 2,
    '1600-16:59': 3, '1700-1759': 3, '1800-1859': 3, '1900-1959': 3, '2000-2059': 3, '2100-2159': 3,
    '2200-2259': 4, '2300-2359': 4, '0001-0559': 4
}


class DecisionTreeNode:
    def __init__(self, parent, examples: pd.DataFrame):
        self.parent = parent
        self.children = {}
        self.examples = examples
        self.is_leaf = False
        self.leaf_return_value = None
        self.chosen_column = None
        self.decision_tree_learning()

    def decision_tree_learning(self):
        if all_the_same(self.examples):  # all the rows with the same y value - late or not
            self.is_leaf = True
            self.leaf_return_value = self.examples.iat[0, self.examples.columns.get_loc('DEP_DEL15')]
            return
        elif self.examples.shape[0] == 1:  # no rows left
            self.is_leaf = True
            self.leaf_return_value = all_the_same(self.parent.examples)
            return
        elif self.examples.shape[1] == 1:  # no column left
            self.is_leaf = True
            self.leaf_return_value = all_the_same(self.examples)
            return

        else:
            self.chosen_column = min_entropy(self.examples)  # choose the column with the min entropy
            values, is_range = get_ranges(self.chosen_column, self.examples)
            # make for each range the children tree
            for value in values:
                child_examples = get_chosen_examples(self.chosen_column, self.examples, value,
                                                     is_range)  # return the values in the row/range
                child_examples = child_examples.loc[:, child_examples.columns != self.chosen_column]  # drop column
                self.children[value] = DecisionTreeNode(self, child_examples)

    # purning the tree by chi2
    def pruning(self):
        if self.is_leaf:
            return
        for child in self.children.values():
            child.pruning()
        for child in self.children.values():
            if not child.is_leaf:
                return

        statistic = 0
        p_True = self.examples['DEP_DEL15'].sum() / self.examples.shape[0]
        p_False = (self.examples.shape[0] - self.examples['DEP_DEL15'].sum()) / self.examples.shape[0]
        for child in self.children.values():
            # estimate value
            estimate_True_child = child.examples.shape[0] * p_True
            estimate_False_child = child.examples.shape[0] * p_False
            # real value
            True_child = child.examples['DEP_DEL15'].sum()
            False_child = child.examples.shape[0] - child.examples['DEP_DEL15'].sum()
            # calculate statistic
            statistic += ((True_child - estimate_True_child) ** 2) / estimate_True_child
            statistic += ((False_child - estimate_False_child) ** 2) / estimate_False_child

        degrees_of_freedom = self.examples.shape[0] - 1
        critical = chi2.ppf(0.6, degrees_of_freedom)

        if statistic < critical:  # prune the tree
            self.is_leaf = True
            self.leaf_return_value = all_the_same(self.examples)
            self.children = None
            self.chosen_column = None
            return

    # decide if the flight will be late or not
    def decide(self, entry: pd.DataFrame):
        if self.is_leaf:  # reached a decision
            return self.leaf_return_value
        if isinstance(entry, pd.DataFrame):
            value = entry[self.chosen_column].values[0]  # extract the first (and only) value from DataFrame
        else:
            value = entry[self.chosen_column]  # Directly access the value from Series
        for key in self.children:
            if type(key) is tuple:  # if the key is a range of values
                if key[0] < value <= key[1]:  # find the range
                    return self.children[key].decide(entry)
            else:  # if the key is a singular value
                if value == key:  # if our value matches the key
                    return self.children[key].decide(entry)

    # printing the tree
    def print_tree(self, prefix=""):
        print(self.chosen_column + ":" if not self.is_leaf else self.leaf_return_value)
        if self.children is not None:
            for index, key in enumerate(self.children.keys()):
                print(f"{prefix}+-- {key} -> ", end="")
                self.children[key].print_tree(
                    prefix + "    " if index == len(self.children.keys()) - 1 else prefix + "|   ")


# return True if all the y values are late/ not late
def all_the_same(examples):
    total_late = examples['DEP_DEL15'].sum()
    if total_late == examples.shape[0] or total_late == 0:
        return 1
    return 0


#  returns the row/s in the column for the value/range
def get_chosen_examples(column, examples, value, is_ranges):
    if is_ranges == False:
        return examples.loc[examples[column] == value]
    return examples.loc[(value[0] < examples[column]) & (examples[column] <= value[1])]


# converts string values into 2 catgories - 1:A-M, 2:N-Z
def map_values(column):
    def map_value(value):
        if value[0].upper() >= 'A' and value[0].upper() <= 'M':
            return 1
        elif value[0].upper() >= 'N' and value[0].upper() <= 'Z':
            return 2

    return column.apply(map_value)


# return the column divided to buckets
def get_ranges(column, examples: pd.DataFrame):
    values = set(examples[column])  # unique values in the column
    value_list = list(values)
    mid_points = [float("-inf")] + [examples[column].quantile(x / buckets, interpolation='midpoint') for x in
                                    range(1, buckets)] + [float("inf")]
    bounds = set((mid_points[i], mid_points[i + 1]) for i in range(buckets))
    bounds = set(bound for bound in bounds if not get_chosen_examples(column, examples, bound, True).empty)
    return bounds, True


# return the column with the min entropy
def min_entropy(examples):
    entropy_columns_value = []
    for column in examples.columns:
        if column != 'DEP_DEL15':
            values, is_ranges = get_ranges(column, examples)
            examples_rows = examples.shape[0]
            entropy = 0
            for value in values:
                value_examples = get_chosen_examples(column, examples, value, is_ranges)
                N_k = value_examples.shape[0]  # number of values
                True_k = value_examples['DEP_DEL15'].sum()  # number of values that late
                False_k = N_k - True_k  # number of values that not late
                p_True = True_k / N_k
                p_False = False_k / N_k
                True_entropy = 0 if p_True == 0 else p_True * log(p_True)
                False_entropy = 0 if p_False == 0 else p_False * log(p_False)
                entropy += (N_k / examples_rows) * (True_entropy + False_entropy)

            entropy_columns_value.append((column, -entropy))

    return min(entropy_columns_value, key=lambda x: x[1])[0]


# arrenge the un-numeric values
def set_data(data):
    data['DEP_TIME_BLK'] = data['DEP_TIME_BLK'].map(dep_time_bucket)
    data['CARRIER_NAME'] = map_values(data['CARRIER_NAME'])
    data['DEPARTING_AIRPORT'] = map_values(data['DEPARTING_AIRPORT'])
    data['PREVIOUS_AIRPORT'] = map_values(data['PREVIOUS_AIRPORT'])
    return data


# calculate precision
def calculate_precision(testing_data, decision_tree):
    success_cases = 0
    for index, row in testing_data.iterrows():
        actual = row['DEP_DEL15']
        prediction = decision_tree.decide(row)
        if prediction == actual:
            success_cases += 1
    return success_cases


# building the tree with ratio % of the data and printing the tree. test with the rest and print accuracy
def build_tree(ratio):
    data = pd.read_csv(file_name)
    data = data.sample(frac=1).reset_index(drop=True)  # suffle the data

    # arrenge the un-numeric values
    data = set_data(data)

    # set training and testing data
    total_rows = data.shape[0]
    train_rows = math.floor(total_rows * ratio)
    training_data = data.head(train_rows)
    testing_data = data.tail(total_rows - train_rows)

    # make the tree
    print("building the tree...")
    decision_tree = DecisionTreeNode(None, training_data)
    decision_tree.pruning()
    decision_tree.print_tree()

    # calculate precision
    print("calculate Accuracy...")
    success_cases = calculate_precision(testing_data, decision_tree)
    accuracy = success_cases / testing_data.shape[0]
    print(f"Accuracy: {accuracy * 100:.2f}%")


# building k-fold trees and calculate the mean error
def tree_error(k):
    data = pd.read_csv(file_name)
    data = data.sample(frac=1).reset_index(drop=True)  # suffle the data

    # arrenge the un-numeric values
    data = set_data(data)

    k_fold_rows = data.shape[0] // k
    sum_error = 0

    for i in range(k):
        training_data = data.drop(data.index[k_fold_rows * i:k_fold_rows * (i + 1)])
        testing_data = data.iloc[k_fold_rows * i:k_fold_rows * (i + 1)]

        decision_tree = DecisionTreeNode(None, training_data)
        decision_tree.pruning()
        success_cases = calculate_precision(testing_data, decision_tree)
        sum_error += success_cases / testing_data.shape[0]

    print((f"Error: {(1 - (sum_error / k)) * 100:.2f}%"))


# predict if the flight will be late based on the given row input
def is_late(row_input):
    data = pd.read_csv(file_name)
    data = data.sample(frac=1).reset_index(drop=True)  # suffle the data

    # arrenge the un-numeric values
    data = set_data(data)

    decision_tree = DecisionTreeNode(None, data)
    decision_tree.pruning()

    test_data = pd.DataFrame([row_input], columns=list(data.columns)[:-1])
    test_data = set_data(test_data)
    predict = decision_tree.decide(test_data)

    prediction_message = "the flight will be late" if predict == 1 else "the flight will not be late"
    print(f"\nThe prediction is: {prediction_message}")
