# this script defines functions to take a df and perform uni-variate
# glucose predictions

import pandas as pd

glucose_csv = '/Users/jonathanlifferth/PycharmProjects/longevity_solutions_ml/glucose.csv'


def train_model(df_path):
    # read csv path and create df
    df = pd.read_csv(df_path)
    df = df.drop(columns=['Unnamed: 0'])

    # create time windows
    window_interval = 30  # time in minutes, smallest possible interval is 5 minutes

    frame_1 = 'glucose_minus_' + str(window_interval)
    frame_2 = 'glucose_minus_' + str(window_interval * 2)
    frame_3 = 'glucose_minus_' + str(window_interval * 3)

    frame_shift_1 = int(window_interval / 5)
    frame_shift_2 = int((window_interval * 2) / 5)
    frame_shift_3 = int((window_interval * 3) / 5)
    # print(frame_shift_1, frame_shift_2, frame_shift_3)

    df[frame_1] = df['glucose'].shift(+frame_shift_1)
    df[frame_2] = df['glucose'].shift(+frame_shift_2)
    df[frame_3] = df['glucose'].shift(+frame_shift_3)

    # drop na values
    df = df.dropna()
    # print(df)

    # load sklearn models
    from sklearn.linear_model import LinearRegression
    lin_model = LinearRegression()
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, max_features=3, random_state=1)

    # organize and reshape data
    import numpy as np
    x1, x2, x3, y = df[frame_1], df[frame_2], df[frame_3], df['glucose']
    x1, x2, x3, y = np.array(x1), np.array(x2), np.array(x3), np.array(y)
    x1, x2, x3, y = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), y.reshape(-1, 1)
    final_x = np.concatenate((x1, x2, x3), axis=1)

    # split 70/30 into train and test sets
    X_train_size = int(len(final_x) * 0.7)
    set_index = len(final_x) - X_train_size
    # print(set_index)
    X_train, X_test, y_train, y_test = final_x[:-set_index], final_x[-set_index:], y[:-set_index], y[-set_index:]

    # fit models
    model.fit(X_train, y_train.ravel())  # random forest
    lin_model.fit(X_train, y_train)  # linear regression

    # make Random Forest Regressor prediction
    pred = model.predict(X_test)
    # make Linear Regression prediction
    lin_pred = lin_model.predict(X_test)
    # combine Aggregate predictions
    pred = pred.reshape(-1, 1)
    aggregate_pred = np.mean(np.array([lin_pred, pred]), axis=0)

    # create weekly report for diabetes patient 0
    total_time = len(y_test)
    time_out_of_range = (y_test > 200).sum()
    percent_in_range = ((total_time - time_out_of_range) / total_time) * 100
    percent_in_range = round(percent_in_range)
    pred_out_of_range = (aggregate_pred > 200).sum()
    pred_out_of_range = round(pred_out_of_range)
    pred_accuracy = round((pred_out_of_range / time_out_of_range) * 100)
    glucose_max = round(y_test.max())
    glucose_min = round(y_test.min())
    glucose_mean = round(y_test.mean())

    summary_out = """
    This week, you spent {} % of your time in range
    Great job!
    Average glucose this week was : {}
    Your highest glucose was: {}
    Your lowest glucose was: {}
    Nudge accurately predicted {} % of time out of range
    """.format(percent_in_range, glucose_mean, glucose_max, glucose_min, pred_accuracy)
    print(summary_out)

    # uncomment block to view mean squared error values
    # from sklearn.metrics import mean_squared_error
    # from math import sqrt
    # rmse_rf = sqrt(mean_squared_error(pred, y_test))
    # rmse_lr = sqrt(mean_squared_error(lin_pred, y_test))
    # rmse_agg = sqrt(mean_squared_error(aggregate_pred, y_test))
    # print('Mean Squared Error for Random Forest Model is:', rmse_rf)
    # print('Mean Squared Error for Linear Regression Model is:', rmse_lr)
    # print('Mean Squared Error for Aggregate Model is:', rmse_agg)

    return aggregate_pred, y_test, summary_out
