import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import base64
import time
import datetime
import io


from dualPredictor.df_preprocess import data_preprocessing
from dualPredictor import DualModel
from dualPredictor.model_plot import plot_scatter,plot_cm,plot_feature_coefficients


def upload_data():
    st.subheader('1. Upload your datasets')
    df_train=None
    df_test=None
    # Create file uploaders for training and test datasets
    train_file = st.file_uploader("Upload Training Dataset", type=["csv", "xlsx"])
    test_file = st.file_uploader("Upload Test Dataset", type=["csv", "xlsx"])

    # If a file is uploaded, read it into a DataFrame
    if train_file is not None:
        if train_file.name.endswith('.csv'):
            df_train = pd.read_csv(train_file)
        elif train_file.name.endswith('.xlsx'):
            df_train = pd.read_excel(train_file)
   

    if test_file is not None:
        if test_file.name.endswith('.csv'):
            df_test= pd.read_csv(test_file)
        elif test_file.name.endswith('.xlsx'):
            df_test = pd.read_excel(test_file)
    return df_train,df_test

def df_info(df):
    buf=io.StringIO()
    df.info(buf=buf)
    s=buf.getvalue()
    return s


def user_input(df_train):
    with st.popover("User Input"):
        st.subheader('2. Specify the target column and ID col')
        cols=df_train.columns.tolist()
        cols.append(None)
        target_col=st.selectbox(label='target column',options=cols)
        id_col=st.selectbox(label='ID column',options=cols)
        drop_cols=st.multiselect(label='drop columns',options=cols,default=cols[0])
        default_cut_off=st.number_input('default cut-off')
        return target_col,id_col,drop_cols,default_cut_off


def plot_result(X,y,model,default_cut_off):
    st.subheader("Model Performance Plots")
    y_pred,y_label_pred=model.predict(X)
    y_label_true = (y < default_cut_off ).astype(int)
    # Display images with Lasso Performance Metrics
    col1, col2= st.columns(2)
    with col1:
        st.subheader("Model Performance - Regression R2")
        scatter_fig=plot_scatter(y_pred, y)
        st.pyplot(scatter_fig)
    with col2:
        st.subheader("Model Performance - Confusion Matrix")
        cm_fig=plot_cm(y_label_true, y_label_pred)
        st.pyplot(cm_fig)

def df_test_pred(model,X_test,df_test):
    # Define the function to create the download link
    def create_download_button(data, file_name, mime):
        download_data = base64.b64encode(data).decode()
        href = f'<a href="data:{mime};base64,{download_data}" download="{file_name}"><button>Download {file_name}</button></a>'
        st.markdown(href, unsafe_allow_html=True)
    y_pred,y_label_pred=model.predict(X_test)
    df=df_test.copy()
    df['y_pred']=y_pred
    df['y_label_pred']=y_label_pred
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.subheader('4. Download Predictions Result')
    create_download_button(csv_data, 'predictions.csv', 'text/csv')
    return df


def main():
    st.set_page_config(layout="wide")
    st.title("Student's Grade Prediction Web App")

    st.markdown('''
### The Model Mechanism
Model Package PyPI Link: https://pypi.org/project/dualPredictor/
- **Step 1: Grade Prediction Using the Trained Regressor**
  fit the linear model f(x) using the training data, and grade prediction can be generated from the fitted model
  
  $$
      y\_pred = f(x) = \sum_{j=1}^{M} w_j x_j + b
  $$
  
- **Step 2: Determining the Optimal Cut-off** 
  
  The goal is to find the **optimal cut-off (C)** that maximizes the binary classification accuracy.
  here we offer 3 options of metrics that measure the classification accuracy: Youden index, f1_score, and f2_score.
  Firstly, the user specifies the metric type used for the model (e.g., Youden index) and denotes the **metric function as g(y_true_label, y_pred_label)**, where:
                
  $$
   {C}_{optimal} = arg\max_c g(y_{true label}, y_{pred label})
  $$      
                
  This formula searches for the cut-off value that produces the highest value of the metric function g, where:
  
  * **c**: The tunned cut-off that determines the y_pred_label
  * y_true_label: True label of the data point based on the default cut-off (e.g., 1 for at-risk, 0 for normal)
  * y_pred_label: Predicted label of the data point based on the tunned cut-off value

    
- **Step 3: Binary Label Prediction**: 
  
  - y_pred_label = 1 (at-risk): if y_pred < optimal_cut_off
  - y_pred_label = 0 (normal): if y_pred >= optimal_cut_off

''')
    # Sidebar content
    st.sidebar.title("About the Web App")
    st.sidebar.markdown('created by D')
    st.sidebar.markdown("# User Guide:")
    st.sidebar.markdown("1. Upload your training and test datasets.")
    st.sidebar.markdown("2. Specify the User Input for Model Building")
    st.sidebar.markdown("3. Click 'Train the Model' to start the model training.")
    st.sidebar.markdown("4. Explore the model performance plots and download prediction results.")

    start_time=time.time()
    # Call the function and store the returned DataFrames
    df_train,df_test=upload_data()
    if df_train is not None:
        st.subheader('Training Dataset')
        st.write(df_train)
        s_df_train=df_info(df_train)
        st.text(s_df_train)
    if df_test is not None:
        st.subheader('Testing Dataset')
        st.write(df_test)
        s_df_test=df_info(df_test)
        st.text(s_df_test)
        st.subheader('2. User Input for Model Building')
        target_col,id_col,drop_cols,default_cut_off=user_input(df_train)

        st.subheader("3. Start Training the Model")
        run_model=st.button(label="Train the Model")
        if run_model:
            scaler=None
            imputer=None
            X_train, y_train,scaler,imputer =data_preprocessing(df=df_train, target_col=target_col, id_col=id_col, drop_cols=drop_cols, scaler=scaler, imputer=imputer)
            scaler=scaler
            imputer=imputer
            X_test, y_test,scaler,imputer = data_preprocessing(df=df_test, target_col=target_col, id_col=id_col, drop_cols=drop_cols, scaler=scaler, imputer=imputer)
        
            model_type='lasso'
            metric='youden_index'
            model = DualModel(model_type, metric, default_cut_off)
            model.fit(X_train, y_train)
            optimal_cut_off=model.optimal_cut_off
            st.subheader('The model has been built:')
            st.text(f'optimal cut off = {optimal_cut_off}')
            # Feature importance plot
            global_fc_fig=plot_feature_coefficients(coef=model.coef_, feature_names=model.feature_names_in_)
            colx, coly= st.columns(2)
            with colx:
                st.pyplot(global_fc_fig)
            with coly:
                expander=st.expander('See explanation')
                
                expander.markdown('''
                # Model Feature Coefficients
                The model coefficient plot illustrates the influence of each feature on the model's predictions. Here's how to read it:     
                - **Length**: Longer bars indicate a stronger effect, with positive bars for positive impact and negative bars for negative impact.            
                - **Direction**: Positive bars suggest a feature that increases the predicted outcome, while negative bars suggest a feature that decreases it.''')


            # for train set
            with st.expander('Model Performance on Training Dataset'):
                plot_result(X_train,y_train,model,default_cut_off)
            with st.expander('Model Performance on Testing Dataset'):
                plot_result(X_test,y_test,model,default_cut_off)
            
            end_time=time.time()
            runtime=round(end_time-start_time,2)
            st.text(f"Total Runtime: {runtime} seconds")

            df_pred=df_test_pred(model,X_test,df_test)



if __name__ == "__main__":
    main()