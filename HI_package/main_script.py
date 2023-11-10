# Import custom modules
from HI_package.imports_module import *
from HI_package.functions_module import *

#from imports_module import *  # imports are in imports_module.py
#from functions_module import *  # required functions are in functions_module.py

def main(dataframes):

    selected_dataframe = input("Choose a dataframe (df_sulphur, df_turb, df_nh, df_po4, df_doc, df_nsol): ")

    if selected_dataframe not in dataframes:
        print("Invalid dataframe choice. Exiting.")
        return

    df_selected = dataframes[selected_dataframe]

    # Prompt user for the target column associated with the selected dataframe
    target_column_dict = {
        'df_sulphur': 'so4',
        'df_turb': 'turbidity',
        'df_nh': 'nh4',
        'df_po4': 'po4',
        'df_doc': 'doc',
        'df_nsol': 'nsol'
    }

    target_column = target_column_dict[selected_dataframe]

    # Split the merged DataFrame into features (X) and target (y)
    X = df_selected.drop(target_column, axis=1)
    y = df_selected[target_column]

    # Remove outliers from the feature columns and target values
    outlier_choice = input("Choose outlier removal method (1: IQR, 2: Z-score, 3: Isolation Forest, 4: None): ")

    # Handle the outlier removal based on the user's choice
    if outlier_choice == '1':
        X = remove_outliers_iqr(X)
    elif outlier_choice == '2':
        X = remove_outliers_zscore(X)
    elif outlier_choice == '3':
        X = remove_outliers_isolation_forest(X)
    elif outlier_choice == '4':
        pass  # Do nothing, no outlier removal
    else:
        print("Invalid outlier removal choice. Exiting.")
        return

    # Adjust target values based on removed outliers from X
    y = y.loc[X.index]
    X = X.dropna()

    # Get preprocessing function choice, threshold, model choice, and feature selection choice from user input
    preprocessing_choice = input("Choose preprocessing function (1: Min-Max Scaling, 2: Standard Scaling, 3: Normalization, 4: L2 Normalization, 5: None (no scaling)): ")

    if preprocessing_choice == '1':
        preprocess_func = preprocess_min_max_scaling
    elif preprocessing_choice == '2':
        preprocess_func = preprocess_standard_scaling
    elif preprocessing_choice == '3':
        preprocess_func = preprocess_normalization
    elif preprocessing_choice == '4':
        preprocess_func = preprocess_l2_normalization  # Added this choice for L2 normalization
    elif preprocessing_choice == '5':
        preprocess_func = None
    else:
        print("Invalid preprocessing choice. Exiting.")
        return


    # Only scale if a preprocess_func was specified
    if preprocess_func:
        X_scaled = preprocess_func(X)
    else:
        X_scaled = X.values  # Convert to NumPy array if no scaling is chosen

    # Applying savitzky filter
    savitzky_choice = input("Choose whether to apply savitzky (1: yes, 2: no): ")
    if savitzky_choice == '1':
        X_scaled = savgol_filter(X_scaled, window_length=27, polyorder=1)
    else:
        X_scaled = X_scaled

    # Applying PCA
    apply_pca = input("Do you want to apply PCA? (yes/no): ")
    if apply_pca.lower() == 'yes':
        pca_n_components = int(input("Enter number of PCA components (1-280): "))
        pca = PCA(n_components=pca_n_components)
        X_scaled = pca.fit_transform(X_scaled)

    model_choice = input("Choose model (1: PLS, 2: SVM, 3: XGBoost, 4: Random Forest, 5: Lasso Regression): ")

    if model_choice in ['1', '2', '5']:

        if model_choice == '1':
            tune_parameters_func = tune_parameters_pls
            default_estimator = PLSRegression(n_components=1)
        elif model_choice == '2':
            tune_parameters_func = tune_parameters_svm
            default_estimator = SVR(kernel = 'linear')
        elif model_choice == '3':
            tune_parameters_func = tune_parameters_xgboost
            default_estimator = xgb.XGBRegressor()
        elif model_choice == '4':
            tune_parameters_func = tune_parameters_randomforest
            default_estimator = RandomForestRegressor()
        elif model_choice == '5':
            tune_parameters_func = tune_parameters_lasso
            default_estimator = Lasso()
        else:
            print("Invalid model choice. Exiting.")
            return


    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    values = []

    if model_choice in ['1', '2', '5']:

        rmse_folds = []
        r2_folds = []
        predictions = []
        for train_index, test_index in kfold.split(X_scaled):
          if model_choice == "1" :
              X_train, X_test = X_scaled[train_index], X_scaled[test_index]
              y_train, y_test = y.iloc[train_index], y.iloc[test_index]
              best_model = tune_parameters_func(X_train, y_train)
              rfecv = RFECV(estimator=best_model, step=1, min_features_to_select=len(best_model.n_iter_),scoring ="neg_mean_squared_error", cv=KFold(n_splits=3, shuffle=True, random_state=20))
              rfecv.fit(X_train, y_train)
              X_train_selected = X_train[:, rfecv.support_]
              X_test_selected = X_test[:, rfecv.support_]
              #best_model = tune_parameters_func(X_train_selected, y_train)
              prediction = rfecv.estimator_.predict(X_test_selected)
              rmse, r2 = evaluate_model(rfecv.estimator_, X_test_selected, y_test)
              rmse_folds.append(rmse)
              r2_folds.append(r2)
              predictions.append(pd.DataFrame({'Prediction': prediction.flatten(), 'Actual': y_test.values.flatten()}))
              print("Mean Squared Error (RMSE):", rmse)
              print("R-squared value:", r2)
          else :
              X_train, X_test = X_scaled[train_index], X_scaled[test_index]
              y_train, y_test = y.iloc[train_index], y.iloc[test_index]
              best_model = tune_parameters_func(X_train, y_train)
              rfecv = RFECV(estimator=best_model, step=1,scoring ="neg_mean_squared_error", cv=KFold(n_splits=3,shuffle=True, random_state=20))
              rfecv.fit(X_train, y_train)
              X_train_selected = X_train[:, rfecv.support_]
              X_test_selected = X_test[:, rfecv.support_]
              #best_model = tune_parameters_func(X_train_selected, y_train)
              prediction = rfecv.estimator_.predict(X_test_selected)
              rmse, r2 = evaluate_model(rfecv.estimator_, X_test_selected, y_test)
              rmse_folds.append(rmse)
              r2_folds.append(r2)
              predictions.append(pd.DataFrame({'Prediction': prediction.flatten(), 'Actual': y_test.values.flatten()}))
              print("Mean Squared Error (RMSE):", rmse)
              print("R-squared value:", r2)



        avg_rmse = np.mean(rmse_folds)
        avg_r2 = np.mean(r2_folds)
        selected_features = get_selected_feature_names(X, rfecv.support_)
        print("Selected Features:", ', '.join(selected_features))
        print("Average Root Mean Squared Error (RMSE):", avg_rmse)
        print("Average R-squared value:", avg_r2)
        prediction_df = pd.concat(predictions, ignore_index=True)
        values.append([len(selected_features), avg_rmse, avg_r2,', '.join(selected_features)])


    elif model_choice in ['3', '4']:  # For XGBoost and RandomForest
        rmse_folds = []
        r2_folds = []

        for train_index, test_index in kfold.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Define estimator based on model choice
            if model_choice == '3':
                best_model = tune_parameters_xgboost(X_train, y_train)
            else:
                best_model = tune_parameters_randomforest(X_train, y_train)

            # Feature selection with RFE inside the loop
            rfe = RFECV(best_model, step=1,scoring = "r2", cv=KFold(3))
            rfe.fit(X_train, y_train)
            X_train_selected = X_train[:, rfe.support_]
            X_test_selected = X_test[:, rfe.support_]

            # Handle tuning based on the model choice
            #if model_choice == '3':
            #    best_model = tune_parameters_xgboost(X_train_selected, y_train)
            #else:
            #    best_model = tune_parameters_randomforest(X_train_selected, y_train)
            rmse, r2 = evaluate_model(rfe.estimator_, X_test_selected, y_test)  # Notice we're passing X_test_selected here
            rmse_folds.append(rmse)
            r2_folds.append(r2)
            print("Mean Squared Error (RMSE):", rmse)
            print("R-squared value:", r2)

        avg_rmse = np.mean(rmse_folds)
        avg_r2 = np.mean(r2_folds)
        selected_features = get_selected_feature_names(X, rfe.support_)
        print("Selected Features:", ', '.join(selected_features))
        print("Average Root Mean Squared Error (RMSE):", avg_rmse)
        print("Average R-squared value:", avg_r2)
        values.append([len(selected_features), avg_rmse, avg_r2, list[selected_features]])


    else:
        print("Invalid model choice. Exiting.")
        return

        # Create a DataFrame to store the results
    result_df = pd.DataFrame(values, columns=['Number of Features', 'Average RMSE', 'Average R2','Selected Features'])
    result_df.to_csv('results.csv', index=False)
    prediction_df.to_csv('prediction.csv', index=False)
if __name__ == "__main__":
    main()
