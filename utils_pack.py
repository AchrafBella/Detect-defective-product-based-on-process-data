import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from tqdm import tqdm
from time import perf_counter

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


# define ensemble modeling
# stacking model
def get_stacking(seed):
    """
    :return:
    """
    model_list = list()
    model_list.append(('lr', LogisticRegression(random_state=seed)))
    model_list.append(('RandomForestClassifier', RandomForestClassifier(random_state=seed, n_jobs=-1)))
    model_list.append(('ExtraTreesClassifier', ExtraTreesClassifier(random_state=seed, n_jobs=-1)))
    model_list.append(('bayes', GaussianNB()))
    level1 = LogisticRegression(random_state=seed)
    return StackingClassifier(estimators=model_list, final_estimator=level1)


# voting model
def get_voting(seed):
    """
    :param seed:
    :return:
    """
    model_list = list()
    model_list .append(('lr', LogisticRegression(random_state=seed)))
    model_list .append(('RandomForestClassifier', RandomForestClassifier(random_state=seed, n_jobs=-1)))
    model_list.append(('ExtraTreesClassifier', ExtraTreesClassifier(random_state=seed, n_jobs=-1)))
    model_list .append(('svm', SVC(probability=True)))
    model_list .append(('bayes', GaussianNB()))
    return VotingClassifier(estimators=model_list, voting='soft')


# define metrics
def error_score(predictions, actual):
    """
    :param predictions:
    :param actual:
    :return:
    """
    return (1/len(predictions)) * (sum(np.where(predictions != actual, 1, 0)))


# define helper function
# oversampling and under sampling techniques
def sampling(dataset, technic):
    """
    This function balance the data collected
    :param dataset: the dataset
    :param technic: choose a technic to balance your data
    :return: balanced data
    """
    # extract values of label 1 and 0
    defective_product = dataset.loc[dataset["Label"] == 1, :]
    good_product = dataset.loc[dataset["Label"] == 0, :]
    if technic == 'over':
        over = defective_product.sample(len(good_product), replace=True)
        balanced_dataset = pd.concat([good_product, over], ignore_index=True)
        return balanced_dataset.sample(frac=1)

    if technic == 'under':
        under = good_product.sample(len(defective_product))
        balanced_dataset = pd.concat([defective_product, under], ignore_index=True)
        return balanced_dataset.sample(frac=1)


# cv score and test score
def bias_variance_detector(models, x_train, y_train, x_test, y_test):
    """
    :param models:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model_information = []
    k = StratifiedKFold(n_splits=8)
    for model in tqdm(models):
        model.fit(x_train, y_train)
        predictions_test = model.predict(x_test)
        predictions_train = model.predict(x_train)
        cv_predictions = cross_val_predict(model, x_train, y_train, cv=k)
        model_information.append((model.__class__.__name__,
                                  cv_predictions,      # validation score
                                  predictions_test,    # test score
                                  predictions_train))  # training score

    # scores
    models_accuracy_cv = list(map(lambda x: accuracy_score(y_train, x[1]), model_information))
    models_accuracy_test = list(map(lambda x: accuracy_score(y_test, x[2]), model_information))
    models_accuracy_train = list(map(lambda x: accuracy_score(y_train, x[3]), model_information))

    models_precision_cv = list(map(lambda x: precision_score(y_train, x[1]), model_information))
    models_precision_test = list(map(lambda x: precision_score(y_test, x[2]), model_information))
    models_precision_train = list(map(lambda x: precision_score(y_train, x[3]), model_information))

    models_recall_cv = list(map(lambda x: recall_score(y_train, x[1]), model_information))
    models_recall_test = list(map(lambda x: recall_score(y_test, x[2]), model_information))
    models_recall_train = list(map(lambda x: recall_score(y_train, x[3]), model_information))

    models_f1_cv = list(map(lambda x: f1_score(y_train, x[1]), model_information))
    models_f1_test = list(map(lambda x: f1_score(y_test, x[2]), model_information))
    models_f1_train = list(map(lambda x: f1_score(y_train, x[3]), model_information))

    indexes = list(map(lambda x: x[0], model_information))
    # data

    data = {'Train accuracy': models_accuracy_train,
            'Train precision': models_precision_train,
            'Train recall': models_recall_train,
            'Train f1': models_f1_train,

            'dev accuracy': models_accuracy_cv,
            'dev precision': models_precision_cv,
            'dev recall': models_recall_cv,
            'dev f1': models_f1_cv,

            'Test accuracy': models_accuracy_test,
            'Test precision': models_precision_test,
            'Test recall': models_recall_test,
            'test f1': models_f1_test}

    scores = pd.DataFrame(data, index=indexes)
    scores.index.name = 'algorithm'
    return scores


# test plenty of models and return their scores
def models_tester(models, x_train, y_train, x_test, y_test):
    """
    :param models:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model_information = []
    for model in tqdm(models):
        t = perf_counter()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        model_information.append((model.__class__.__name__,
                                  predictions,
                                  perf_counter() - t))
        pass

    # scores
    models_accuracy = list(map(lambda x: accuracy_score(y_test, x[1]), model_information))
    models_precision = list(map(lambda x: precision_score(y_test, x[1]), model_information))
    models_recall = list(map(lambda x: recall_score(y_test, x[1]), model_information))
    models_f1 = list(map(lambda x: f1_score(y_test, x[1]), model_information))
    models_roc_auc = list(map(lambda x: roc_auc_score(y_test, x[1]), model_information))
    classification_error = list(map(lambda x: error_score(y_test, x[1]), model_information))
    matthews = list(map(lambda x: matthews_corrcoef(y_test, x[1]),  model_information))

    # metadata
    cpu = list(map(lambda x: x[2], model_information))
    indexes = list(map(lambda x: x[0], model_information))
    # data
    data = {'Accuracy': models_accuracy,
            'Error': classification_error,
            'Precision': models_precision,
            'Recall': models_recall,
            'f1': models_f1,
            'ROC_AUC': models_roc_auc,
            'Matthews_score': matthews,
            'CPU': cpu}
    scores = pd.DataFrame(data, index=indexes)
    scores.index.name = 'algorithm'
    return scores


# you plot these scores in scatter plot
def model_evaluation_scatter(scores):
    """
    :param scores:
    :return:
    """
    sns.set_palette("GnBu_d")
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(data=scores, x="Precision", y="Recall", hue="algorithm",
                         s=150)
    base_model_lines = scores.loc['GaussianNB']

    _ = ax.vlines(base_model_lines.Precision, 0, base_model_lines.Recall,
                  color="red", linestyle="--")
    _ = ax.hlines(base_model_lines.Recall, 0, base_model_lines.Precision,
                  color="red", linestyle="--")

    ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=2))
    ax.set_title(f"Performance per algorithm")
    ax.get_legend_handles_labels()
    for i, txt in enumerate(scores.index):
        ax.annotate(txt, (scores.Precision[i], scores.Recall[i]))
    plt.show()


# score with score validation
def model_cross_val_score(models, x_train, y_train, scoring):
    """
    :param models:
    :param x_train:
    :param y_train:
    :param scoring:
    :return:
    """
    model_information = []
    k = StratifiedKFold(n_splits=8)
    for model in tqdm(models):
        cv_results = cross_val_score(model, x_train, y_train, cv=k, scoring=scoring)
        model_information.append((model.__class__.__name__, cv_results))
    # compute the scores
    scores = list(map(lambda x: x[1], model_information))
    indexes = list(map(lambda x: x[0], model_information))
    return pd.DataFrame(scores, index=indexes).T


# box plot
def plot_box_plot_result_models(scores):
    """
    :param scores:
    :return:
    """
    plt.figure(figsize=(18, 6))
    ax = sns.boxplot(data=scores, orient="h", palette="Set2", width=.6)
    ax.set_yticklabels(scores.T.index)
    ax.set_title('Cross validation accuracies with different classifiers')
    ax.set_xlabel('F1 scores')
    plt.show()


# for features selections
def permutation_importance_plot(model, x_test, y_test):
    """
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    result = permutation_importance(model, x_test, y_test, n_repeats=5)
    sorted_df_idx = result.importances_mean.argsort()
    importance = pd.DataFrame(
        result.importances[sorted_df_idx].T,
        columns=x_test.columns[sorted_df_idx],
    )

    ax = importance.plot.box(vert=False, whis=2, figsize=(10, 12))
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    return importance


# test one model
def model_tester(model, x_train, y_train, x_test, y_test):
    k = StratifiedKFold(n_splits=8)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test, y_test)
    cv_predictions = cross_val_predict(model, x_train, y_train, cv=k)

    # scores
    models_accuracy_cv = accuracy_score(y_train, cv_predictions)
    models_accuracy_test = accuracy_score(y_test, predictions)

    models_precision_cv = precision_score(y_train, cv_predictions)
    models_precision_test = precision_score(y_test, predictions)

    models_recall_cv = recall_score(y_train, cv_predictions)
    models_recall_test = recall_score(y_test, predictions)

    models_f1_cv = f1_score(y_train, cv_predictions)
    models_f1_test = f1_score(y_test, predictions)

    indexes = list(map(lambda x: x[0], model_information))
    # data

    data = {'dev accuracy': models_accuracy_cv,
            'Test accuracy': models_accuracy_test,

            'dev precision': models_precision_cv,
            'Test precision': models_precision_test,

            'dev recall': models_recall_cv,
            'Test recall': models_recall_test,

            'dev f1': models_f1_cv,
            'test f1': models_f1_test}

    scores = pd.DataFrame(data, index=indexes)
    scores.index.name = 'algorithm'
    return scores


# search for best weights params

# search for best weights params
def grid_search_best_weight(model_name, weights, x_train, y_train, x_test, y_test):
    """
    :param model_name:
    :param weights:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    info = []
    for weight in tqdm(weights):
        model = model_name(n_jobs=-1,
                           random_state=42,
                           class_weight=weight)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        info.append(
            (weight, predictions)
        )

        # scores
    models_accuracy = list(map(lambda x: accuracy_score(y_test, x[1]), info))
    classification_error = list(map(lambda x: error_score(y_test, x[1]), info))
    models_precision = list(map(lambda x: precision_score(y_test, x[1]), info))
    models_recall = list(map(lambda x: recall_score(y_test, x[1]), info))
    models_f1 = list(map(lambda x: f1_score(y_test, x[1]), info))
    models_roc_auc = list(map(lambda x: roc_auc_score(y_test, x[1]), info))
    matthews = list(map(lambda x: matthews_corrcoef(y_test, x[1]), info))

    # meta data
    weights = list(map(lambda x: x[0], info))

    # data
    data = {'Accuracy': models_accuracy,
            'Error': classification_error,
            'Precision': models_precision,
            'Recall': models_recall,
            'f1': models_f1,
            'ROC_AUC': models_roc_auc,
            'matthews_score': matthews,
            'weight': weights}
    return pd.DataFrame(data)


# helper function score visualizer
def confusion_matrix_graph(model, predictions, actual):
    """
    :param model:
    :param predictions:
    :param actual:
    :return:
    """
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(confusion_matrix(actual, predictions), annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("Y_predictions")
    plt.ylabel("Y_true")
    plt.title(f"Confusion matrix for {model.__class__.__name__}")
    plt.show()
    print()
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(actual, predictions)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(actual, predictions)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(actual, predictions)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(actual, predictions)
    print('F1 score: %f' % f1)
    roc_auc = roc_auc_score(actual, predictions)
    print('roc_auc score: %f' % roc_auc)




