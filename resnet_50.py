
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet import ResNet50
from keras.models import Model
import keras
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Conv2D,Dense, Dropout
from keras.models import Sequential
from model_functions import load_data,print_results_model

def built_model_resnet50():
    IMG_WIDTH=81
    IMG_HEIGHT=81
    IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

    batch_size = 32

    train_path = 'train_test_split/train'
    valid_path = 'train_test_split/validate'
    test_path = 'train_test_split/test'

    class_mode='categorical'
    size=81
    classes = ['airplane','balloon', 'bird', 'drone','helicopter','other']

    train_generator,val_generator,test_generator=load_data(classes,
                                                           batch_size,
                                                           train_path,
                                                           test_path,
                                                           valid_path,
                                                           size,
                                                           class_mode)

    restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)

    restnet = Model(restnet.input,output)

    for layer in restnet.layers:
        layer.trainable = False

    restnet.summary()

    model = Sequential()

    model.add(restnet)

    model.add(Dense(512, activation='relu', input_dim=(IMG_HEIGHT,IMG_WIDTH,3)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=2e-5),
                  metrics=['categorical_accuracy'])
    model.build((None, 81, 81, 3))

    model.summary()
    save_as='detect_balloon_restnet_50_4.h5'
    checkpoint = ModelCheckpoint(save_as,
                                 verbose=1, monitor='val_loss',save_best_only=True,
                                 mode='auto')

    history = model.fit(train_generator,
                                  steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=test_generator,
                                  validation_steps=50,
                                  verbose=1, callbacks=[checkpoint])

    print_results_model(save_as,history)

built_model_resnet50()
# https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38
