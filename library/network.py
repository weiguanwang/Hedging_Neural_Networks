import keras

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.backend import clear_session
from sklearn.preprocessing import StandardScaler

from . import common as cm



class HedgeNet:
    """
    This class encapsulates methods including building a network,
    fitting it, loading existing model, etc.
    """

    def __init__(self):
        self.model = None

        
    def load_model(
            self, file_path,
            overwrite=False,
            custom_objects=None):
        if (overwrite is False) and (self.model is not None):
            raise Exception(
                'Model already exists, and overwrite received False')
        else:
            self.model = load_model(
                filepath=file_path,
                custom_objects=custom_objects)

            
    def save_model(self, file_path):
        save_model(self.model, filepath=file_path)

        
    def build_model(
            self, feature_shape,
            nodes_per_layer,
            lr=0.001,
            outact=None,
            loss='mse',
            metrics=[None],
            reg_alpha=1e-4,
    ):
        features = Input(
            shape=feature_shape, dtype='float32', name='features')
        x = Dense(
            nodes_per_layer[0], activation='relu',
            kernel_regularizer=l2(reg_alpha)
        )(features)
        for num_nodes in nodes_per_layer[1:]:
            x = Dense(
                num_nodes, activation='relu',
                kernel_regularizer=l2(reg_alpha)
            )(x)
        out_trainable = Dense(
            1, activation=outact, name='delta_before_flag'
        )(x)

        cp_int = Input(
            (1,), dtype='float32', name='cp_int')
        delta = keras.layers.subtract(
            [out_trainable, cp_int], name='delta'
        )
        V0 = Input((1, ), dtype='float32', name='V0')
        S0 = Input((1, ), dtype='float32', name='S0')
        S1 = Input((1, ), dtype='float32', name='S1')
        on_ret = Input((1, ), dtype='float32', name='on_ret')

        delta_S0 = keras.layers.multiply([S0, delta], name='delta_S0')
        delta_S1 = keras.layers.multiply([S1, delta], name='delta_S1')
        cash0 = keras.layers.subtract([V0, delta_S0], name='cash0')
        cash1 = keras.layers.multiply([cash0, on_ret], name='cash1')
        V1_hat = keras.layers.add([cash1, delta_S1], name='V1_hat')

        self.model = Model(
            inputs=[features, cp_int, V0, S0, S1, on_ret],
            outputs=[V1_hat]
        )
        self.model.compile(
            optimizer=Adam(lr=lr),
            loss=loss,
            metrics=metrics
        )

        
    def summary(self):
        self.model.summary()

        
    def fit(
            self,
            train_data,
            val_data,
            use_features,
            V1,
            epochs=100,
            batch_size=64,
            callbacks=[]
    ):
        """
        This method trains a network, while evaluating on
        a validation data set.
        :param V1: the column name for target V1
        """
        # Make two dictionary for feeding training and validation data.

        train_pair = {}
        val_pair = {}
        for data, pair in zip([train_data, val_data], [train_pair, val_pair]):
            x = {
                'features': data[use_features].values,
                'cp_int': data['cp_int'].values,
                'V0': data['V0_n'].values,
                'S0': data['S0_n'].values,
                'S1': data['S1_n'].values,
                'on_ret': data['on_ret'].values
            }
            t = {'V1_hat': data[V1].values}
            pair['x'] = x
            pair['t'] = t

        history = self.model.fit(
            x=train_pair['x'],
            y=train_pair['t'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_pair['x'], val_pair['t']),
            callbacks=callbacks
        )
        return history

    
    def calc_V1(self, df, use_features):
        """
        Return the predicted V1. The exact meaning of V1 depends on setting.
        """
        x = {
            'features': df[use_features].values,
            'cp_int': df['cp_int'].values,
            'V0': df['V0_n'].values,
            'S0': df['S0_n'].values,
            'S1': df['S1_n'].values,
            'on_ret': df['on_ret'].values
        }
        return self.model.predict(x=x)

    
    def calc_delta(self, df, use_features):
        output_name = 'delta'
        submodel = Model(
            inputs=[self.model.get_layer('features').input,
                    self.model.get_layer('cp_int').input],
            outputs=self.model.get_layer(name=output_name).output
        )
        var = submodel.predict(
            x={'features': df[use_features].values,
               'cp_int': df['cp_int'].values})
        return var.flatten()
    
    
def normcdf(x):
    return tf.distributions.Normal(loc=0., scale=1.).cdf(x)


def make_checkpoint(
        filepath,
        monitor='val_mean_squared_error'
):
    ckp = ModelCheckpoint(
        filepath,
        monitor=monitor,
        save_best_only=True)
    return ckp


def save_history(history, filepath):
    pd.DataFrame(history.history).to_csv(filepath)


def read_history(filepath):
    return pd.read_csv(filepath, index_col=0)


def plot_history(
        history, 
        filepath,
        df_train,
        df_val
    ):
    history = pd.DataFrame(history.history)

    pnl_train_bs = (cm.calc_pnl(df_train, df_train['delta_bs'])**2).mean()
    pnl_val_bs = (cm.calc_pnl(df_val, df_val['delta_bs'])**2).mean()

    x_min_trainmse = history['mean_squared_error'].idxmin()
    x_min_valmse = history['val_mean_squared_error'].idxmin()

    f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(12, 4))
    ax1.plot(history['mean_squared_error'], label='Network MSE')
    ax1.axhline(pnl_train_bs, xmin=0, xmax=1, c='r', label='Black-Scholes MSE')
    ax1.axvline(x_min_trainmse, ymin=0, ymax=1,  linestyle='dashed', label='Minimum loss')
    ax1.legend()
    ax2.plot(history['val_mean_squared_error'], label='Network MSE')
    ax2.axhline(pnl_val_bs, xmin=0, xmax=1, c='r', label='Black-Scholes MSE')
    ax2.axvline(x_min_valmse, ymin=0, ymax=1,  linestyle='dashed', label='Minimum loss')
    ax2.legend()
    plt.ylim(0., 0.03)
    ax1.set_xlim(20, history['mean_squared_error'].shape[0])
    ax2.set_xlim(20, history['mean_squared_error'].shape[0])

    f.savefig(filepath)


def train_net_core(
        df_train,
        df_val,
        use_fea,
        hypers,
        sub_res_paths
    ):
    """
    Core function to train network.
    """
    net = HedgeNet()
    ckp_path = sub_res_paths['ckp'] 
    ckp = make_checkpoint(ckp_path)

    if hypers['outact'] == 'normcdf':
        hypers['outact'] = normcdf
        
    net.build_model(
        (len(use_fea),),
        nodes_per_layer=hypers['nodes_per_layer'],
        metrics=['mean_squared_error'],
        reg_alpha=hypers['reg_alpha'],
        lr=hypers['lr'],
        outact=hypers['outact']
    )
    history = net.fit(
        df_train, df_val,
        use_features=use_fea,
        V1='V1_n',
        epochs=hypers['epochs'],
        callbacks=[ckp]
    )
    save_history(history, sub_res_paths['history'])
    clear_session()
    return history


def test_net_core(
        df_test, 
        use_fea, 
        sub_res_paths
):
    net = HedgeNet()
    ckp_path = sub_res_paths['ckp']
    net.load_model(ckp_path, custom_objects={'normcdf': normcdf})
    delta_nn = net.calc_delta(df_test, use_fea)
    clear_session()
    return delta_nn



def standardize_feature(list_of_df, scaler, ori_feature):
    trans_fea = [x + '_t' for x in ori_feature]
    var_save = []
    for df in list_of_df:
        var = df.reindex(columns=df.columns.tolist() + trans_fea)
        var[trans_fea] = scaler.transform(var[ori_feature])
        var_save.append(var)
    return var_save




def rolling_net(
        df,
        ori_fea,
        use_fea,
        end_period,
        hypers=None,
        sub_res_dir=None,
        tune=False
):
    if hypers['outact'] == 'normcdf':
        hypers['outact'] = normcdf

    for i in range(0, end_period + 1):
        sub_res_paths = {
            'ckp': sub_res_dir['ckp'] + f'bestcp{i}.h5',
            'history': sub_res_dir['history'] + f'history{i}.csv',
            'plot': sub_res_dir['plot'] + f'losscurve{i}.png'
        }

        print('\n\n Working on period {}.\n'.format(i))

        df_train = df.loc[df['period{}'.format(i)] == 0]
        df_val = df.loc[df['period{}'.format(i)] == 1]
        df_test = df.loc[df['period{}'.format(i)] == 2]

        scaler = StandardScaler().fit(X=df_train[ori_fea])
        df_train, df_val, df_test = standardize_feature([df_train, df_val, df_test], scaler, ori_fea)

        history = train_net_core(
            df_train,
            df_val,
            use_fea,
            hypers=hypers,
            sub_res_paths=sub_res_paths
        )
        plot_history(
            history,
            sub_res_paths['plot'],
            df_train,
            df_val
        )
        if not tune:
            delta_nn = test_net_core(
                df_test,
                use_fea,
                sub_res_paths
            )
            df.loc[df_test.index, 'delta_nn'] = delta_nn

    return df




