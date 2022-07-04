
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from model_functions import load_data,print_results_model

def built_model_vgg16():

    # model 1= 224 categorical
    # model 2= 81 binary
    # model 3= 81 categorical

    batch_size = 32

    # model 1
    train_path = 'train_test_split/train'
    valid_path = 'train_test_split/validate'
    test_path = 'train_test_split/test'
    # model 2
    # train_path = 'create_dataset_2/train'
    # test_path = 'create_dataset_2/test'
    # valid_path='create_dataset_2/validate'
    # model 3
    # train_path='dataset_2_classes/dataset/train'
    # test_path='dataset_2_classes/dataset/test'
    # valid_path='dataset_2_classes/dataset/validate'

    # # model 1
    # size=224
    # model 2 and model 3
    size=81


    # model 1
    class_mode='categorical'
    # model 2
    # class_mode='binary'
    # model 3
    # class_mode='categorical'

    # model 1
    # classes = ['airplane', 'bird', 'drone','helicopter']
    # # model 2
    # classes = ['yes','no']
    # model 3
    classes = ['airplane','balloon', 'bird', 'drone','helicopter','other']

    train_batches,valid_batches,test_batches=load_data(classes,
                                                           batch_size,
                                                           train_path,
                                                           test_path,
                                                           valid_path,
                                                           size,
                                                           class_mode)

    # model 1
    # vgg16_model = VGG16()
    # model 2 model 3
    vgg16_model = VGG16(weights='imagenet',include_top=False,input_shape=(81,81,3))


    model = Sequential()

    for layer in vgg16_model.layers:
      model.add(layer)

    model.summary()

    conv_model = Sequential()

    for layer in vgg16_model.layers[:-6]:
      conv_model.add(layer)

    conv_model.summary()

    transfer_layer = model.get_layer('block5_pool')

    # define the conv_model inputs and outputs
    conv_model = Model(inputs=conv_model.input,
                       outputs=transfer_layer.output)
    # model 1
    # num_classes = 4
    # # model 2
    # num_classes = 2
    # model 3
    num_classes = 6

    # start a new Keras Sequential model.
    new_model = Sequential()

    # add the convolutional layers of the VGG16 model
    new_model.add(conv_model)

    # flatten the output of the VGG16 model because it is from a
    # convolutional layer
    new_model.add(Flatten())

    # add a dense (fully-connected) layer.
    # this is for combining features that the VGG16 model has
    # recognized in the image.

    # model 1
    new_model.add(Dense(1024, activation='relu'))
    # # model 2 and model 3
    # new_model.add(Dense(81, activation='relu'))

    # add a dropout layer which may prevent overfitting and
    # improve generalization ability to unseen data e.g. the test set
    new_model.add(Dropout(0.5))

    # add the final layer for the actual classification
    new_model.add(Dense(num_classes, activation='softmax'))


    optimizer = Adam(learning_rate=1e-5)

    # loss function should by 'categorical_crossentropy' for multiple classes
    # but here we better use 'binary_crossentropy' because we need to distinguish between 2 classes

    # model 1 and model 3
    # loss = 'binary_crossentropy'
    # print("compile_model")
    # new_model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    # # model 2
    loss = 'categorical_crossentropy'
    print("compile_model")
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])



    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=2,
                       verbose=1,
                       mode='auto')

    step_size_train=train_batches.n//train_batches.batch_size
    step_size_valid=valid_batches.n//valid_batches.batch_size

    history = new_model.fit_generator(train_batches,
                            epochs=30,
                            steps_per_epoch=step_size_train,
                            validation_data=valid_batches,
                            validation_steps=step_size_valid,
                            callbacks = [es],
                            verbose=1)

    step_size_test=test_batches.n//test_batches.batch_size

    result = new_model.evaluate_generator(test_batches, steps=step_size_test)

    print("Test set classification accuracy: {0:.2%}".format(result[1]))

    # test_batches.reset()
    # predictions = new_model.predict_generator(test_batches,steps=step_size_test,verbose=1)
    #
    #
    # # # predicted class indices
    # y_pred = np.argmax(predictions,axis=1)


    print("save_model")

    # # model 1
    # save_as='model_vgg2.h5'
    # # model 2
    # save_as='model_vgg_s81'
    # model 3
    save_as='detect_balloon_vgg16.h5'

    new_model.save(save_as)

    print_results_model(save_as,history)


built_model_vgg16()
