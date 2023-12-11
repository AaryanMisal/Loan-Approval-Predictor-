import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def sigmoid(values):
    activation_values = 1 / ( 1 + np.exp(-values))
    return activation_values

def forward_pass(dic_weights, input):
    dic = {}
    temp = np.array([1])
    input_elements = np.concatenate((temp, input))
    dic["a" + str(1)] = np.array([input_elements])
    dic["a" + str(1)] = dic["a" + str(1)].T
    for i in range(2, len(dic_weights) + 1):
        dic["z" + str(i)] = np.dot(dic_weights[str(i - 1)], dic["a" + str(i - 1)])
        dic["a" + str(i)] = sigmoid(dic["z" + str(i)])
        dic["a" + str(i)] = np.concatenate((np.array([[1]]), dic["a" + str(i)]))

    dic["z" + str(len(dic_weights) + 1)] = np.dot(dic_weights[str(len(dic_weights))],  dic["a" + str(len(dic_weights))])
    dic["a" + str(len(dic_weights) + 1)] = sigmoid(dic["z" + str(len(dic_weights) + 1)])

    #print(dic)
    return dic["a" + str(len(dic_weights) + 1)], dic


def cost_function(dic_weights, input, label, lam):
    J = 0
    for i in range(0, len(input)):
        output, activation_data = forward_pass(dic_weights, input[i])
        J_i = []
        for j in range(0, len(output)):
            y = label[i][j]
            f_x = output[j][0]
            val = (-y * math.log(f_x) - ((1 - y)* math.log(1 - f_x)))
            J_i.append(val)
        #print((sum(J_i)))
        J = J + sum(J_i)

    J = J / len(input)
    
    S = 0
    for i in dic_weights:
        for j in dic_weights[i]:
            for z in range(0, len(j)):
                if z != 0:
                    S = S + j[z]**2
            
    S = (lam / (2 * len(input))) * S
    
    #print(J+S)
    return J + S

def back_propogation(dic_weights, input, label, lam, alpha):
    gradient_total  = {}

    for i in range(0, len(input)):
        delta = {}
        output, activation_data = forward_pass(dic_weights, input[i])
        delta_last_layer = []
        for j in range(0, len(output)):
            d_val = output[j][0] - label[i][j]
            delta_last_layer.append([d_val])

        delta["delta" + str(len(dic_weights) + 1)] = np.array(delta_last_layer)
        
        for j in reversed(range(2, len(dic_weights) + 1)):
            delta["delta" + str(j)] = np.dot(dic_weights[str(j)].T, delta["delta" + str(j + 1)])
            delta["delta" + str(j)] = delta["delta" + str(j)]*(1 - activation_data["a" + str(j)])*activation_data["a" + str(j)]
            delta["delta" + str(j)] = np.delete(delta["delta" + str(j)], 0, 0)
        
        #print(delta)
        gradient = {}
        for j in reversed(range(1, len(dic_weights) + 1)):
            gradient[str(j)] = np.dot(delta["delta" + str(j+1)], activation_data["a" + str(j)].T)
        
        #print(gradient)

        for j in gradient:
            if j in gradient_total:
                gradient_total[j] = gradient_total[j] + gradient[j]
            else:
                gradient_total[j] = gradient[j]
    
    P ={}
    gradient_final = {}

    for i in reversed(range(1, len(dic_weights) + 1)):
        P[str(i)] = lam *dic_weights[str(i)]
        for j in range(0, len(P[str(i)])):
            P[str(i)][j][0] = 0
        gradient_final[str(i)] = (1/len(input))* (gradient_total[str(i)] + P[str(i)])
    
    #print(gradient_final)
    for i in reversed(range(1, len(dic_weights) + 1)):
        dic_weights[str(i)] = dic_weights[str(i)] - alpha*gradient_final[(str(i))]

    return dic_weights


def initialise_weights(neural_net_shape):
    dic_weights = {}

    for i in range(0, len(neural_net_shape) - 1):
        dic_weights[str(i+1)] = np.random.randn(neural_net_shape[i + 1], neural_net_shape[i] + 1)

    return dic_weights


def k_fold(data, k, label, total_labels):
    #finding percentage of each label in the dataset
    percentage = {}
    for i in total_labels:
        percentage[i] = len(data[data[label] == i]) / len(data)

    #finding the length of each fold
    length_fold = int(len(data) / k)

    #finding total of each label in folds
    total_class_numbers = {}
    for i in percentage:
        total_class_numbers[i] = round(percentage[i] * length_fold)
    
    k_fold_data = []
    temp_data = data
    for i in range(1, k):
        lst_data = []
        for j in total_class_numbers:
            temp = temp_data[temp_data[label] == j]
            temp = temp.sample(n= total_class_numbers[j],replace= False)
            index_lst = list(temp.index)
            temp_data = temp_data.drop(index_lst)

            lst_data.append(temp)
        d = pd.concat(lst_data)
        k_fold_data.append(shuffle(d))


    new_lst = []
    for j in total_class_numbers:
        temp = temp_data[temp_data[label] == j]
        temp = shuffle(temp)
        temp = temp[:total_class_numbers[j]]
        index_lst = list(temp.index)
        temp_data = temp_data.drop(index_lst)
        new_lst.append(temp)
    
    d = pd.concat(new_lst)
    k_fold_data.append(shuffle(d))
    
    return k_fold_data


def normalise_train(data, file):
    dic = {}
    if file == "data/loan_data.csv":
        columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for i in columns:
        dic[i] = tuple([data[i].max(), data[i].min()])
        data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    return data, dic     


def normalise_test(data, max_min_data, file):
    if file == "data/loan_data.csv":
        columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for i in columns:
        data[i] = (data[i] - max_min_data[i][1]) / (max_min_data[i][0] - max_min_data[i][1])
    return data       

def calculate_performance_two_class(predictions, test_data_label_check, total_labels):
    true_predictions = {}
    count = 0
    for i in total_labels:
        true_predictions[i] = {}

    for i in true_predictions:
        for j in total_labels:
                true_predictions[i][j] = 0

    for i in range(0, len(predictions)):
        if predictions[i] == test_data_label_check[i]:
                count += 1
                true_predictions[predictions[i]][predictions[i]] += 1
        
        else:
                true_predictions[test_data_label_check[i]][predictions[i]] += 1

    accuracy = count / (len(predictions))
    
    if true_predictions[1][1] == 0 and true_predictions[0][1] == 0:
        precision = 0
    else:
        precision = true_predictions[1][1] / (true_predictions[1][1] + true_predictions[0][1])

    recall = true_predictions[1][1] / (true_predictions[1][1] + true_predictions[1][0])
    
    if precision == 0:
        F1_score = 0
    else:
        F1_score = (2*(precision*recall))/(precision + recall)

    return accuracy, precision, recall, F1_score


def testing_results(final_weights, test_data_input, test_data_label, test_data_label_check, total_labels, file):
    real_prediction = []
    for i in range(0, len(test_data_input)):
        prediction , network_dic = forward_pass(final_weights, test_data_input[i])
        max = -math.inf
        for j in range(0,len(prediction)):
            if prediction[j][0] > max:
                max = prediction[j][0]
                max_index = j
        if file == "data/loan_data.csv":
            real_prediction.append(max_index)

    real_predictions = np.array(real_prediction)

    if file == "data/loan_data.csv":
        accuracy, precision, recall, F1_score = calculate_performance_two_class(real_predictions, test_data_label_check, total_labels)

    return accuracy, precision, recall, F1_score

def plot_graph(loss, iteration):
    plt.plot(iteration, loss)
    plt.ylabel('loss')
    plt.xlabel('training examples')
    plt.title('loss vs training examples')
    plt.show()

def main(file, n_shape, iterations, lam, alpha, graph_flag):

    dic_weights = {}

    if file == 'data/loan_data.csv':
        data = pd.read_csv(file)

        data = data.drop("Loan_ID", axis = 1)
        data['Gender'].replace(['Male', 'Female'], [1, 0], inplace=True)
        data['Married'].replace(['Yes','No'], [1, 0], inplace=True)
        data['Dependents'].replace(['0', '1','2','3+'], [0, 1, 2, 3], inplace=True)
        data['Education'].replace(['Graduate', 'Not Graduate'], [1, 0], inplace=True)
        data['Self_Employed'].replace(['No', 'Yes'], [0, 1], inplace=True)
        data['Property_Area'].replace(['Urban', 'Rural', 'Semiurban'], [0, 1, 2], inplace=True)
        data['Loan_Status'].replace(['Y', 'N'], [1, 0], inplace=True)
        catogerical_attributes = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        total_remove = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']
        label = data.columns[-1]
        total_labels = data[label].unique()
        

        k_fold_data = k_fold(data, 10, label, total_labels)

        accuracy_count = []
        precision_count = []
        recall_count = []
        F1_score_count = []
        
        if graph_flag != True:
            for i in range(0, len(k_fold_data)):
                train_d =  k_fold_data[:i] + k_fold_data[i+1:]
                train_data = pd.concat(train_d)
                test_data  =  k_fold_data[i].copy()

                temp_train_data = train_data.copy()
                train_data_normalise, max_min_data = normalise_train(temp_train_data, file)

                temp_train_data_label = train_data[label].to_numpy()
                exe_loan_input_1 =  OneHotEncoder()
                # train data input
                temp_train_data_input_1 = exe_loan_input_1.fit_transform(train_data_normalise[catogerical_attributes]).toarray()

                temp_train_data_input_2 = (train_data_normalise.drop(columns = total_remove)).to_numpy()

                train_data_input = np.concatenate((temp_train_data_input_1, temp_train_data_input_2), axis =1)

                exe_loan_label_1  = OneHotEncoder()
                # train data label
                train_data_label = exe_loan_label_1.fit_transform(temp_train_data_label.reshape(-1, 1)).toarray()

                temp_test_data = test_data.copy()
                test_data_normalise = normalise_test(temp_test_data, max_min_data, file)

                test_data_label_check = test_data[label].to_numpy()
                temp_test_data_label = test_data[label].to_numpy()

                # test data input
                temp_test_data_input_1 = exe_loan_input_1.transform(test_data_normalise[catogerical_attributes]).toarray()

                temp_test_data_input_2 = (test_data_normalise.drop(columns = total_remove)).to_numpy()

                test_data_input = np.concatenate((temp_test_data_input_1, temp_test_data_input_2), axis =1)
                # test data label
                test_data_label = exe_loan_label_1.transform(temp_test_data_label.reshape(-1, 1)).toarray()

                neural_net_shape = []
                for j in range(0, len(n_shape)):
                    if j == 0:
                        neural_net_shape.append(len(train_data_input[0]))
                    else:
                        neural_net_shape.append(n_shape[j])

                print(neural_net_shape)
                
                dic_weights = initialise_weights(neural_net_shape)

                for j in range(0, iterations):
                    final_weights = back_propogation(dic_weights, train_data_input, train_data_label, lam, alpha)
                    dic_weights = final_weights


                accuracy, precision, recall, F1_score = testing_results(final_weights, test_data_input, test_data_label, test_data_label_check, total_labels, file)

                accuracy_count.append(accuracy)
                precision_count.append(precision)
                recall_count.append(recall)
                F1_score_count.append(F1_score)
            

            print(accuracy_count)

            accuracy_final = (sum(accuracy_count)) / (len(accuracy_count))
            precision_final  = (sum(precision_count)) / (len(precision_count))
            recall_final  = (sum(recall_count)) / (len(recall_count))
            F1_score_final  = (sum(F1_score_count)) / (len(F1_score_count))

            print(accuracy_final)
            print(precision_final)
            print(recall_final)
            print(F1_score_final)
        
        else:
            tr, te = train_test_split(data, test_size= 0.2)
           
            temp_tr = tr.copy()
            tr_normalise, max_min_data = normalise_train(temp_tr, file)
 
            temp_tr_label = tr[label].to_numpy()
            exe_loan_input_2 =  OneHotEncoder()

            temp_tr_input_1 = exe_loan_input_2.fit_transform(tr_normalise[catogerical_attributes]).toarray()

            temp_tr_input_2 = (tr_normalise.drop(columns = total_remove)).to_numpy()
            
            # train data input
            tr_input = np.concatenate((temp_tr_input_1, temp_tr_input_2), axis =1)
            
            exe_loan_label_2  = OneHotEncoder()
            # train data label
            tr_label = exe_loan_label_2.fit_transform(temp_tr_label.reshape(-1, 1)).toarray()


            temp_te = te.copy()

            te_normalise = normalise_test(temp_te, max_min_data, file)

            te_label_check = te[label].to_numpy()
            temp_te_label = te[label].to_numpy()

            temp_te_input_1 = exe_loan_input_2.transform(te_normalise[catogerical_attributes]).toarray()

            temp_te_input_2 = (te_normalise.drop(columns = total_remove)).to_numpy()

            # test data input
            te_input = np.concatenate((temp_te_input_1, temp_te_input_2), axis =1)

            # test data label
            te_label = exe_loan_label_2.transform(temp_te_label.reshape(-1, 1)).toarray()


            loss_lst = []
            samples = []
            neural_net_shape = []
            for i in range(0, len(n_shape)):
                if i == 0:
                    neural_net_shape.append(len(tr_input[0]))
                else:
                    neural_net_shape.append(n_shape[i])

            for i in range(0,iterations):
                samples.append(i*len(tr_input))
                d_weights = initialise_weights(neural_net_shape)
                for j in range(0, i):
                    f_weights = back_propogation(d_weights, tr_input, tr_label, lam, alpha)
                    d_weights = f_weights

                l = cost_function(d_weights, te_input, te_label, lam)
                loss_lst.append(l)
            
            plot_graph(loss_lst, samples)   

if __name__ == "__main__":  
    file = 'data/loan_data.csv'
    iterations = 100
    lam = 0.05
    alpha = 3
    n_shape = [9, 5, 2]
    graph_flag = False
    main(file, n_shape, iterations, lam, alpha, graph_flag)