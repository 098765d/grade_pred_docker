# dualPredictor/model_plot.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd


def plot_scatter(y_pred, y_true):
    # Create a copy of y_pred and impute NaN values with the mean
    y_true_imputed = y_true.copy()
    mean_value = np.nanmean(y_true)
    y_true_imputed[np.isnan(y_true_imputed)] = mean_value

    # Calculate the r2 and mse using the imputed values
    r2 = r2_score(y_true_imputed, y_pred)
    r2 = abs(r2)
    mse = mean_squared_error(y_true_imputed, y_pred)

    print('R2 = ', r2)
    print('MSE = ', mse)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)

    # Plot the data using the imputed values
    sns.scatterplot(x=y_true_imputed, y=y_pred, ax=ax)

    # Set the x and y limits
    ax.set_xlim(min(y_true)*0.8, max(y_true)*1.1)
    ax.set_ylim(min(y_true)*0.8, max(y_true)*1.1)

    # Add the r2 and mse to the figure
    ax.text(0.05, 0.95, f"R2 =  {r2:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"MSE = {mse:.2f}", transform=ax.transAxes)

    # Add a diagonal line
    ax.plot([min(y_true)*0.8, max(y_true)*1.1], [min(y_true)*0.8, max(y_true)*1.1], color='red', linestyle='--')

    # Set the x and y axis descriptions
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')

    # Add a title with total data points and purpose
    num_data_points = len(y_true)
    title = f"Actual vs Predicted Plot\nTotal Data Points: {num_data_points}"
    ax.set_title(title)

    return fig



def plot_cm(y_label_true, y_label_pred):

    # Calculate the confusion matrix
    cm = confusion_matrix(y_label_true, y_label_pred)
    tn, fp, fn, tp=cm.ravel()

    # Calculate the number of data points, number of miss detects, number of false alarms, and classification rate
    num_data_points = len(y_label_true)
    num_false_alarms,num_miss_detects = fp,fn
    classification_rate = round(np.trace(cm) / num_data_points,3)
    num_tp=tp+fn

    # Print the number of data points, number of miss detects, number of false alarms, and classification rate
    print("Number of data points:", num_data_points)
    print("Number of label=1 points:", num_tp)
    print("Number of miss detects:", num_miss_detects)
    print("Number of false alarms:", num_false_alarms)
    print("Classification rate:", classification_rate)

    # Display the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=500)
    disp.plot(ax=ax)

    # Add figure title with total number of data points and classification rate
    title = f"Confusion Matrix\nTotal Data Points: {num_data_points}\nClassification Rate: {classification_rate:.2f}"
    ax.set_title(title)

    return fig

def plot_feature_coefficients(coef, feature_names):

    # Round the coefficients to two decimal places
    rounded_coef = np.round(coef, decimals=2)

    # Get non-zero feature coefficients and corresponding names
    nonzero_coef = rounded_coef[rounded_coef != 0]
    nonzero_names = np.array(feature_names)[rounded_coef != 0]
    num_of_features=len(nonzero_names)

    # Sort non-zero coefficients and names in descending order
    sorted_indices = np.argsort(np.abs(nonzero_coef))[::-1]
    sorted_coef = nonzero_coef[sorted_indices]
    sorted_names = nonzero_names[sorted_indices]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 6), dpi=500)

    # Plot horizontal bars of feature coefficients
    ax.barh(sorted_names, sorted_coef)

    # Set limits for x-axis
    ax.set_xlim([np.min(sorted_coef) * 1.1, np.max(sorted_coef) * 1.1])

    # Add coefficient values at the end of each bar
    for i, coef in enumerate(sorted_coef):
        ax.text(coef, i, str(coef), va="center")

    # Add a title to the plot
    ax.set_title(f"Feature Coefficient Plot - {num_of_features} features")

    return fig

def plot_cutoff_tuning(cutoffs,metrics, xlabel="Cutoff", ylabel="Metric"):
    """Plots the results of cutoff tuning.

    Args:
        dual_clf: A DualModel object.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
    """

    # Plot the results
    plt.figure(figsize=(7, 3.5), dpi=500)
    plt.plot(cutoffs, metrics, label=f'{ylabel}', color='blue')
    plt.title("Cutoff Tuning Plot")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()



def plot_local_shap(model, X, idx, dpi=300, figsize=(5.5, 4.5)):

    # Calculate SHAP values for the specified row of X_train
    w = model.coef_ # model coefficients
    x = X.loc[idx] # the individual data point with index = idx
    shaps = w * (x - X.mean()) # approximating linear Shapley Value for every feature

    shaps=np.round(shaps,2)

    # Create a DataFrame of SHAP values and feature names
    shap_x = pd.DataFrame({'features': model.feature_names_in_, 'shap': shaps})
    shap_x = shap_x[shap_x['shap'] != 0]
    shap_x['positive_contribution'] = shap_x['shap'] > 0
    # Sort bars by absolute SHAP value in descending order
    shap_x = shap_x.sort_values(by='shap', ascending=False, key=abs)

    # Plot SHAP values using seaborn barplot
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax = sns.barplot(data=shap_x, x='shap', y='features', hue='positive_contribution',legend=False)
    plt.title("Feature Contribution for ID = '{}'".format(idx))
    plt.xlabel("Feature Contribution")
    plt.ylabel("Feature")

    for i in ax.containers:
      ax.bar_label(i,)

    # Set x-axis limits to 1.5 times the minimum SHAP value
    plt.xlim(left=2.7 * shap_x['shap'].min(), right=1.2*shap_x['shap'].max())
    return fig
    
# local_shap=plot_local_shap(model, X_train, idx='8112C5A6')
