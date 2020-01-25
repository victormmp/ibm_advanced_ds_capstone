import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score, train_test_split, KFold


class Regressor:

    def __init__(self, model: str = 'xgboost', *args, **model_parameters):
        self._parameters = model_parameters
        self.backend = model
        self._space = None
        self.x = None
        self.y = None
        self.trials = None

        self._model_init = {
            'xgboost': self._xgboost,
            'keras': self._keras
        }.get(model, None)

        self._model_init()

    @staticmethod
    def _model_train():
        raise NotImplementedError(f'Backend model may not be correctly initialized.')

    @staticmethod
    def _model_predict():
        raise NotImplementedError(f'Backend model may not be correctly initialized.')

    @staticmethod
    def _get_params():
        raise NotImplementedError(f'Backend model may not be correctly initialized.')

    @staticmethod
    def _set_params():
        raise NotImplementedError(f'Backend model may not be correctly initialized.')

    @staticmethod
    def _objective(params):
        raise NotImplementedError(f'Backend model may not be correctly initialized.')

    def reset(self):
        self._model_init()

    def train(self, x, y, **parameters):
        self.x = x
        self.y = y
        self._model_train(
            x=x,
            y=y,
            **parameters
        )

        return self

    def predict(self, x, **parameters):
        return self._model_predict(
            x=x,
            **parameters
        )

    def tune_hyper_parameters(self, x, y, space=None):
        if space:
            self._space = space

        self.x = x
        self.y = y
        self.trials = Trials()

        best = fmin(
            fn=self._objective,
            space=self._space,
            algo=tpe.suggest,
            trials=self.trials,
            return_argmin=False,
            max_evals=100)

        print('Best: {}'.format(best))

        self._parameters.update(best)
        
        self._model_init()

        # with open(f'best_params_{self.backend}.json', 'w') as fp:
        #     json.dump(best, fp)

        return self

    @property
    def parameters(self):
        return self._get_params()

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        self._set_params(**parameters)

    def _xgboost(self):
        print('Using XGBoost as backend')

        if self._parameters:

            self._parameters['n_estimators'] = int(self._parameters['n_estimators'])
            self._parameters['max_delta_step'] = int(self._parameters['max_delta_step'])
            self._parameters['max_depth'] = int(self._parameters['max_depth'])

            params = {
                'tree_method': 'hist',
                'grow_policy': 'lossguide',
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'verbosity': 1
            }
            self._parameters.update(params)

        self._model = xgb.XGBRegressor(**self._parameters)

        # Clear state parameters
        self.x = None
        self.y = None

        # Define XGBoost train method
        def model_train(x, y, **params):
            self._model.fit(x, y, **params)

        self._model_train = model_train

        # Define XGBoost get parameter method
        self._get_params = self._model.get_xgb_params

        self._set_params = self._model.set_params

        # Define XGBoost predict method
        def predict(x, **parameters):
            return self._model.predict(x)

        self._model_predict = predict

        # Define model hyperparameter space for tuning
        self._space = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'reg_alpha': hp.uniform('reg_alpha', 0, 2),  # alpha
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 0.9),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 0.9),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.7),
            'learning_rate': hp.uniform('learning_rate', 0.05, 0.4),  # eta
            'min_split_loss': hp.uniform('min_split_loss', 0, 10),  # gamma
            'reg_lambda': hp.uniform('reg_lambda', 0, 10),  # lambda
            'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
            'max_depth': hp.quniform('max_depth', 0, 10, 1),
            'min_child_weight': hp.uniform('min_child_weight', 0, 10),
            'n_estimators': hp.quniform('n_estimators', 0, 500, 10),  # num_round
            'subsample': hp.uniform('subsample', 0.2, 0.6)
        }

        # Define objective function for hyperparameter tuning
        def objective(params):
            params['n_estimators'] = int(params['n_estimators'])
            params['max_delta_step'] = int(params['max_delta_step'])
            params['max_depth'] = int(params['max_depth'])

            params['verbose'] = -1
            params['tree_method'] = 'hist'
            params['grow_policy'] = 'lossguide'

            model = xgb.XGBRegressor(**params)

            scores = cross_val_score(model, self.x, self.y, scoring="neg_mean_squared_error", cv=4)
            score = np.mean(scores)

            return {'loss': -score, 'status': STATUS_OK}

        self._objective = objective

    def _keras(self):
        print('Using Keras as backend')

        input_shape = self._parameters.get('input_shape', (32,))
        layers = self._parameters.get('layers', [32])
        layers_activation = self._parameters.get('layers_activation', 'relu')
        output_activation = self._parameters.get('output_activation', 'linear')
        loss = self._parameters.get('loss', 'mean_squared_error')

        if not isinstance(layers, list):
            layers = [layers]

        layers = [int(l) for l in layers]

        layers.append(1)

        adam_parameters = {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'amsgrad': False
        }
        optimizer = self._parameters.get('optimizer', Adam)
        optimizer_parameters = self._parameters.get('optimizer_parameters', adam_parameters)

        opt = optimizer(**optimizer_parameters)

        metrics = self._parameters.get('metrics', ['mean_squared_error'])

        self._model = Sequential()

        if len(layers) == 1:
            self._model.add(Dense(layers[0], input_shape=input_shape, activation=output_activation))
        elif len(layers) == 2:
            self._model.add(Dense(layers[0], input_shape=input_shape, activation=layers_activation))
            self._model.add(Dense(layers[1], activation=output_activation))
        elif len(layers) > 2:
            self._model.add(Dense(layers[0], input_shape=input_shape, activation=output_activation))
            for layer in layers[1:-1]:
                self._model.add(Dense(layer, activation=layers_activation))
            self._model.add(Dense(layers[-1], activation=output_activation))

        self._model.compile(loss=loss,
                            optimizer=opt,
                            metrics=metrics)

        # Clear state parameters
        self.x = None
        self.y = None

        # Define Keras train method
        def model_train(x, y, **params):
            try:
                params['epochs'] = int(params['epochs'])
            except:
                params['epochs'] = int(self._parameters.get('epochs', 10))

            try:
                params['batch_size'] = int(params['batch_size'])
            except:
                params['batch_size'] = int(self._parameters.get('batch_size', 36))

            self._model.fit(x, y, **params)

        self._model_train = model_train

        def get_params():
            return self._parameters

        self._get_params = get_params

        def set_params(*args, **kwargs):
            self._keras()

        self._set_params = set_params

        # Define XGBoost predict method
        def predict(x, **params):
            try:
                params['batch_size'] = int(params['batch_size'])
            except:
                params['batch_size'] = int(self._parameters.get('batch_size', 36))

            return self._model.predict(x, **params)

        self._model_predict = predict

        # Define model hyperparameter space for tuning
        self._space = {
            'optimizer': Adam,
            'layers': hp.quniform('layers', 1, 100, 1),
            'layers_activation': hp.choice('layers_activation', ['tanh', 'relu', 'linear']),
            'optimizer_parameters': {
                'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
                'beta_1': hp.uniform('beta_1', 0.001, 1),
                'beta_2': hp.uniform('beta_2', 0.001, 1),
                'amsgrad': hp.choice('amsgrad', (True, False)),
            },
            'epochs': hp.quniform('epochs', 1, 100, 1),
            'batch_size': hp.quniform('batch_size', 10, 500, 10)
        }

        # Define objective function for hyperparameter tuning
        def objective(params):

            input_shape = self._parameters.get('input_shape', (32,))
            layers = [int(params['layers']), 1]
            layers_activation = params['layers_activation']
            output_activation = self._parameters.get('output_activation', 'linear')
            loss = self._parameters.get('loss', 'mean_squared_error')

            optimizer = params['optimizer']
            optimizer_parameters = params['optimizer_parameters']

            opt = optimizer(**optimizer_parameters)
            metrics = self._parameters.get('metrics', ['mean_squared_error'])

            model = Sequential()

            if len(layers) == 1:
                model.add(Dense(layers[0], input_shape=input_shape, activation=output_activation))
            elif len(layers) == 2:
                model.add(Dense(layers[0], input_shape=input_shape, activation=layers_activation))
                model.add(Dense(layers[1], activation=output_activation))
            elif len(layers) > 2:
                model.add(Dense(layers[0], input_shape=input_shape, activation=output_activation))
                for layer in layers[1:-1]:
                    model.add(Dense(layer, activation=layers_activation))
                model.add(Dense(layers[-1], activation=output_activation))

            model.compile(loss=loss,
                          optimizer=opt,
                          metrics=metrics)

            losses = []
            k_fold = KFold(n_splits=5, shuffle=True)
            for train, test in k_fold.split(self.x, self.y):
                model.fit(self.x.iloc[train, :], self.y.iloc[train, :], epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)
                loss = model.evaluate(self.x.iloc[test, :], self.y.iloc[test, :], batch_size=int(params['batch_size']), verbose=0)
                losses.append(loss[1])

            # loss contains loss and mse

            return {'loss': np.mean(losses), 'status': STATUS_OK}

        self._objective = objective


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    file_path = './data/processed/1_1_processed_ordinal_encoding.csv'
    print(f'Loading file from {file_path}')

    df = pd.read_csv(file_path)
    x = df.drop(['Value', 'Wage'], axis=1)
    y = df[['Value']]

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    params = {}

    model = Regressor(model='keras', input_shape=(x_train.shape[1],),  **params)

    print('Tuning HyperParameters')
    model.tune_hyper_parameters(x_train, y_train)

    print(model._parameters)

    # print(f'Training model')
    # model.train(x_train, y_train)
    #
    # print('Predicting test samples')
    # predictions: pd.DataFrame = model.predict(x_test)
    #
    # print('Prediction Finished. RMSE:')
    # print(mean_squared_error(y_test, predictions, squared=False))
    #
    # fig, ax = plt.subplots(figsize=(12,18))
    # xgb.plot_importance(model._model, max_num_features=50, height=0.8, ax=ax)
    # plt.show()
