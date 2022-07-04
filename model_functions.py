import matplotlib.pyplot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt


def load_data(classes,batch_size,train_path,test_path,valid_path,size,class_mode):
    train_batches = ImageDataGenerator().flow_from_directory(directory=train_path,
                                                             classes=classes,
                                                             class_mode=class_mode,
                                                             target_size=(size, size),
                                                             batch_size=batch_size,
                                                             shuffle=True)

    valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                                             classes=classes,
                                                             class_mode=class_mode,
                                                             target_size=(size, size),
                                                             batch_size=batch_size,
                                                             shuffle=True)

    test_batches = ImageDataGenerator().flow_from_directory(directory=test_path,
                                                            classes=classes,
                                                            class_mode=class_mode,
                                                            target_size=(size, size),
                                                            batch_size=batch_size,
                                                            shuffle=False)
    return train_batches, valid_batches, test_batches


def print_results_model(path_model,history):

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.pyplot.plot(history.history['categorical_accuracy'])
    plt.pyplot.plot(history.history['val_categorical_accuracy'])
    plt.pyplot.title('model accuracy')
    plt.pyplot.ylabel('accuracy')
    plt.pyplot.xlabel('epoch')
    plt.pyplot.legend(['train', 'test'], loc='upper left')
    plt.pyplot.show()
    # smmarize history for loss
    plt.pyplot.plot(history.history['loss'])
    plt.pyplot.plot(history.history['val_loss'])
    plt.pyplot.title('model loss')
    plt.pyplot.ylabel('loss')
    plt.pyplot.xlabel('epoch')
    plt.pyplot.legend(['train', 'test'], loc='upper left')
    plt.pyplot.show()

    saved_model = load_model(path_model)
    saved_model.summary()


from keras.preprocessing import image
import numpy as np
from keras.models import load_model

def load_my_model(model,classes,path_img,size):
    saved_model = load_model(model)
    img = image.load_img(path_img, target_size=(size, size))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = saved_model.predict(img)
    i = np.argmax(output)
    return output,i,classes[i]
