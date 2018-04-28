import logging
import sys
import numpy as np
#keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import load_model
from keras.optimizers import SGD
#sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize

def load_input(input_name, output_name):
    #input("WARNING! Don't be stupid! This eval's the contents of the file you input. "
    #      "Double check the data source first! Press enter to continue.")
    global input_arr
    global output_arr
    input_arr = None
    output_arr = None
    with open(input_name, 'r') as input_file:
        with open(output_name, 'r') as output_file:
            logging.debug("Opening files {0} and {1} for input processing.".format(input_name, output_name))
            input_arr = eval(input_file.read())
            output_arr = eval(output_file.read())
    logging.info(input_arr)
    logging.info(output_arr)
    prepare(input_arr, output_arr)


def prepare(input_arr, output_arr):
    logging.info(len(input_arr))
    logging.info(len(output_arr))
    scaler = RobustScaler()
    scaler.fit(input_arr)
    adjusted_input = scaler.transform(input_arr)
    input_arr = [x*100 for x in input_arr]
    x_train, x_test, y_train, y_test = train_test_split(adjusted_input, output_arr, shuffle=True, test_size=0.2, stratify=output_arr)
    logging.info('Train test split complete.')
    logging.debug(y_train)
    logging.debug(y_test)
    network(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))


def network(x_train, y_train, x_test, y_test):
    """maybe I could have come up with a more creative name"""
    model = Sequential()  # declare model
    model.add(Dense(300, input_shape=(16, ), kernel_initializer='he_normal'))  # first layer
    model.add(Activation('selu'))
    model.add(Dense(300, activation='linear'))
    model.add(Dense(510, activation='selu'))
    model.add(Dense(500, activation='linear'))
    model.add(Dense(7, kernel_initializer='he_normal'))  # last layer
    model.add(Activation('softmax'))
    model.summary()
    sgd = keras.optimizers.sgd(lr=0.05)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])
    logging.info("Model compilation complete.")
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=250,
                        batch_size=1024,
                        verbose=1)


def main():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    load_input('inputData.dta', 'outputData.dta')


if __name__ == '__main__':
    main()