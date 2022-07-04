from bs4 import BeautifulSoup
import pandas as pd
from os import makedirs
from yolo import yolo_detect
from os.path import splitext, dirname, basename, join
import numpy
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,img_to_array, array_to_img, load_img
import cv2
import os

def image_augmentation(img, p):
    datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1,) + x.shape)

    # Generating and saving 5 augmented samples
    # using the above defined parameters.
    i = 300
    for batch in datagen.flow(x, batch_size=1, save_to_dir=p, save_prefix=1, save_format='jpg'):

        i += 1
        if i > 305:
            break

    images_augmentation=[]
    for path in os.listdir(p):
        img_path=p+'/'+path
        img = cv2.imread(img_path)
        images_augmentation.append(img)
        os.remove(img_path)
    return images_augmentation

def change_resolution(frame,size):
    img = Image.fromarray(frame)
    resized_img = img.resize((size, size))
    open_cv_image = numpy.array(resized_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def cut_place(image,place,size):
    # cv2.imshow("txt", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    x=round(float(place[0]))
    y=round(float(place[1]))
    w = round(float(place[2]))
    h = round(float(place[3]))
    # crop_img = image[y:y + h, x:x + w]


    buf_img=[]
    if(h>size or w>size):
        if(h<size):
            h=size
        elif(w<size):
            w=size
        crop_img = image[y:h, x:w]
        images_augmentation=image_augmentation(crop_img,"image_augmentation")
        for img in images_augmentation:
            img=change_resolution(img,size)
            buf_img.append(img)
    else:
        w=size
        h=size
        crop_img = image[y:h, x:w]
        buf_img=image_augmentation(crop_img,"image_augmentation")

    # cv2.imshow("txt", crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return buf_img

def place(str_places):
    places_arr=[]
    places = str_places.split('{')
    places.pop(0)
    for p in places:
        place=[]
        # print("p: " + p)
        arr = p.split(',')
        xmin=int(arr[0].split(':')[-1])
        ymin=int(arr[1].split(':')[-1])
        xmax=int(arr[2].split(':')[-1])
        ymax=int(arr[3].split(':')[-1].split('}')[0])
        place.append(xmin)
        place.append(ymin)
        place.append(xmax)
        place.append(ymax)
        places_arr.append(place)
        print(str(place))
    return places_arr



def create_db_baloons(csv, frame_dir: str,name="image", ext="jpg"):
    v_name = splitext(basename(csv))[0]
    video_path_arr = csv.split('/')
    print(video_path_arr)
    if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)
    print(base_path)

    df = pd.read_csv(csv)
    numOfImages = 1
    for i in range(0,df.shape[0]):
        filePath = df['fname'][i]
        filePath = filePath.replace('\\', '/')
        filePath = filePath.replace('\'', '')
        print(filePath)
        places=df['bbox'][i]
        places=place(places)
        print("filePath: "+filePath)
        image = cv2.imread(frame_dir+'/'+filePath)

        for p in places:
            crop_images_buf=cut_place(image,p,81)
            for img in crop_images_buf:
                filled_numOfImages = str(numOfImages).zfill(4)
                cv2.imwrite("{}_{}.{}".format(base_path, filled_numOfImages, ext),img)
                print("{}_{}.{}".format(base_path, filled_numOfImages, ext))
                numOfImages += 1
    print(numOfImages)


# image_p=r"detectionBaloons\balloons img\25899693952_7c8b8b9edc_k.jpg"
# # p='[{'xmin': 135, 'ymin': 115, 'xmax': 811, 'ymax': 965}]'1536,2048
# place=[135,115,811,965]
# cut_place(image_p,place)
#
csv="detectionBaloons/balloon-data.csv"
frame_dir="detectionBaloons/balloons img"
create_db_baloons(csv,frame_dir)