from dash_app.data_util import build_flat_dataset, build_sequence_dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error as mse


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, activation='sigmoid', inner_activation='hard_sigmoid', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


if __name__ == "__main__":
    input_shape = (20, 23)
    model = build_model(input_shape)
    X, y = build_sequence_dataset(sequence_length=input_shape[0], window=20)
    base_x, base_y = build_flat_dataset()
    print(X.shape)
    model.fit(X[:50], y[:50], batch_size=128, epochs=20)
    pred = model.predict(X[50:])
    print(pred)
    true = y[50:]
    print(true)

    print("MSE: {}".format(mse(true, pred)))
    print("Baseline MSE: {}".format(mse(base_y, base_x.mean(axis=1))))

