import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import base64

st.set_page_config(layout="wide")
coltitle, colsubtitle = st.columns(2)
with coltitle:
    st.title("Machine Learning Dashboard")


# Sidebar with clickable sections
st.sidebar.header('Navigation')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
section = st.sidebar.radio('Go to', ['Data Input', 'Data Processing and Model Creation', 'Model Download'])

# File uploader widget outside the if-elif blocks

filtered_data = None  # Initialize with None, will be updated if file is uploaded

# Check if file is uploaded and preprocess
if uploaded_file is not None:
    dataFrame = pd.read_csv(uploaded_file)
    filtered_data = dataFrame.dropna(axis=0)  # Preprocessed data available for all sections

col1, col2 = st.columns(2)

# Content of each section

if section == 'Data Input':
    with col1:
        st.header('Data Input Section')

        if filtered_data is not None:
            st.subheader('Preview of Uploaded Dataset')
            st.table(filtered_data.head(5))  # Display the first few rows of the uploaded dataset
            data_summary = filtered_data.describe()            
            st.subheader('Summary Statistics of Uploaded Dataset')
            st.table(filtered_data.describe().transpose().head(5))
    with col2:

        if filtered_data is not None:
            columns_to_plot = st.multiselect('Select columns to visualize:', filtered_data.columns.tolist())
            if len(columns_to_plot) > 0:
                # Set up tabs
                histogram_tab, pairwise_scatter_tab = st.tabs(["Histograms", "Pairwise Scatter Plots"])

                # Tab for Histograms
                with histogram_tab:
                    # Let the user select which histogram to display
                    selected_histogram = st.selectbox('Select a histogram to display:', columns_to_plot)
                    st.subheader(f'Distribution of {selected_histogram}')
                    fig = px.histogram(
                        filtered_data, 
                        x=selected_histogram, 
                        marginal='box', 
                        nbins=30, 
                        color_discrete_sequence=['#636EFA']  # Setting color for the histogram
                    )
                    fig.update_traces(marker=dict(line=dict(width=1, color='#483D8B')))  # Border line color for bars
                    st.plotly_chart(fig)


                # Tab for Pairwise Scatter Plots
                with pairwise_scatter_tab:
                    if len(columns_to_plot) >= 2:
                        st.subheader('Pairwise Scatter Plots')
                        fig = px.scatter_matrix(filtered_data[columns_to_plot], color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig)
                    else:
                        st.warning('Select at least two columns for Pairwise Scatter Plots.')
                        
            else:
                st.write('Select columns to visualize.')

        else:
            st.warning('Please upload data in the Data Input section.')



elif section == 'Data Processing and Model Creation':
    col3, col4 = st.columns(2)
    with col3:
        st.header('Data Processing and Model Creation Section')
        if filtered_data is not None:
            # Allow users to select features and target
            target = st.selectbox('Select Target Variable:', filtered_data.columns.tolist())
            available_features = [col for col in filtered_data.columns if col != target]  # Remove selected target from available features
            features = st.multiselect('Select Features:', available_features, default=available_features)  # Default to all available features
        

            # Check if users have selected at least one feature and one target variable
            if len(features) > 0 and target:
                # Separate the data into features and target variable
                X = filtered_data[features]
                y = filtered_data[target]
                # Split data
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
                # Allow users to select a model
                model_choice = st.selectbox('Select a Machine Learning Model:', ['DecisionTreeRegressor', 'RandomForestRegressor', 'OtherModel2'])

                # Train selected model
                if st.button('Train Model'):
                    if model_choice == 'DecisionTreeRegressor':
                        from sklearn.tree import DecisionTreeRegressor
                                                
                        def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
                            model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
                            model.fit(train_X, train_y)
                            preds_val = model.predict(val_X)
                            mae = mean_absolute_error(val_y, preds_val)
                            return(mae)

                        # Dictionary to store the MAE for each value of max_leaf_nodes
                        scores = {leaf_size: get_mae(leaf_size, X_train, X_valid, y_train, y_valid) for leaf_size in [5, 25, 50, 100, 250, 500, 1000, 5000]}

                        # Find the best value of max_leaf_nodes (the one that gives the smallest MAE)
                        best_tree_size = min(scores, key=scores.get)
                        final_mae = scores[best_tree_size]
                        st.write(f"Best Max Leaf Nodes: {best_tree_size}")
                        
                        # You can proceed with training your final model using best_tree_size as max_leaf_nodes
                        final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
                        final_model.fit(X, y)
                        
                        # Make predictions and calculate MAE
                        preds = final_model.predict(X_valid)
                        mae = mean_absolute_error(y_valid, preds)
                        maeis =  mean_absolute_error(y_train, final_model.predict(X_train))
                        
                        # Store variables in session state
                        st.session_state['model'] = final_model
                        st.session_state['X_valid'] = X_valid
                        st.session_state['y_valid'] = y_valid
                        st.session_state['X_train'] = X_train
                        st.session_state['y_train'] = y_train
                        st.session_state['preds'] = preds
                        st.session_state['mae'] = mae
                        st.session_state['maeis'] = maeis
                        st.session_state['features'] = features
                        st.session_state['model_choice'] = model_choice

                    elif model_choice == 'RandomForestRegressor':
                        from sklearn.ensemble import RandomForestRegressor
                        final_model = RandomForestRegressor(random_state=1)
                        final_model.fit(X_train, y_train)
                        preds = final_model.predict(X_valid)
                        mae = mean_absolute_error(y_valid, preds)
                        maeis =  mean_absolute_error(y_train, final_model.predict(X_train))
                        # Store variables in session state
                        st.session_state['model'] = final_model
                        st.session_state['X_valid'] = X_valid
                        st.session_state['y_valid'] = y_valid
                        st.session_state['X_train'] = X_train
                        st.session_state['y_train'] = y_train
                        st.session_state['preds'] = preds
                        st.session_state['mae'] = mae
                        st.session_state['maeis'] = maeis
                        st.session_state['features'] = features
                        st.session_state['model_choice'] = model_choice

                    elif model_choice == 'OtherModel2':
                        st.warning('Code for OtherModel2 is not implemented yet.')
                    
            else:
                st.warning('Please select at least one feature and one target variable.')

        else:
            st.warning('Please upload data in the Data Input section.')

    with col4:
        # Initialize variables with None as default values
        model = None
        X_valid = None
        features = None
        y_valid = None
        X_train = None
        y_train = None
        preds = None
        mae = None
        maeis = None
        model_choice = None
        # Update variables if they exist in session state
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_valid = st.session_state['X_valid']
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            features = st.session_state['features']
            y_valid = st.session_state['y_valid']
            preds = st.session_state['preds']
            mae = st.session_state['mae']
            maeis = st.session_state['maeis']
            model_choice = st.session_state['model_choice']
        else:
            st.warning('Train the model and see the results here.')
        

        if model is not None and X_valid is not None and y_valid is not None:
            # Display Model Evaluation Metrics
            st.subheader('Model Evaluation Metrics')

            # Calculate range and standard deviation of the target variable from the training data
            target_range = y_train.max() - y_train.min()
            target_std = y_train.std()

            # Calculate the proportion of MAEOS relative to the range and standard deviation
            maeos_to_range = mae / target_range
            maeos_to_std = maeis / target_std

            # Create a 3x2 DataFrame with your values
            data = {
                'Metric1': ['Mean Absolute Error (MAE)', 'Target Variable Range', 'MAEOS to Range Ratio'],
                'Value1': [f"{mae:.2f}", f"{target_range:.2f}", f"{maeos_to_range:.2f}"],
                'Metric2': ['Mean Absolute Error In-Sample (MAEIS)', 'Target Variable Standard Deviation', 'MAEOS to Standard Deviation Ratio'],
                'Value2': [f"{maeis:.2f}", f"{target_std:.2f}", f"{maeos_to_std:.2f}"]
            }
            df = pd.DataFrame(data)

            # Display the DataFrame as a table in Streamlit
            st.table(df)


            # Add more metrics as needed
            # Visualize Model Predictions

            actualvspred_tab, featureimp_tab = st.tabs(["Actual Vs Predicted Values Plot", "Feature Importance Plot"])

                # Tab for Histograms
            with actualvspred_tab:
                fig = px.scatter(x=y_valid, y=preds, labels={'x': 'Actual Values', 'y': 'Predicted Values'})
                fig.add_trace(go.Scatter(x=y_valid, y=y_valid, mode='lines', name='Ideal Prediction', line=dict(color='purple')))
                st.plotly_chart(fig)

            with featureimp_tab:
                # Feature Importance (only if model supports it)
                if model_choice == 'DecisionTreeRegressor':
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({'feature': features, 'importance': importances})
                    fig = px.bar(importance_df, x='feature', y='importance', color='importance')
                    st.plotly_chart(fig)

                elif model_choice == 'RandomForestRegressor':
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({'feature': features, 'importance': importances})
                    fig = px.bar(importance_df, x='feature', y='importance', color='importance')
                    st.plotly_chart(fig)

                elif model_choice == 'OtherModel2':
                    st.warning('Visualization for OtherModel2 is not implemented yet.')


elif section == 'Model Download':
    st.header('Model Download Section')
    st.write('Download the trained model for later use.')

    # Check if model is available in session state
    if 'model' in st.session_state:
        # Serialize the model
        serialized_model = pickle.dumps(st.session_state['model'])
        # Encode the serialized model to bytes
        b64_model = base64.b64encode(serialized_model).decode()
        # Create a download link
        href = f'<a href="data:application/octet-stream;base64,{b64_model}" download="trained_model.pkl">Download Trained Model</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning('Please train a model first in the Data Processing and Model Creation section.')
