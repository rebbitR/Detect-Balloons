
import classes
from datetime import datetime
import cv2
from log import CreateLog


def print_current_time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print(current_time)

def detect_balloon_main(img_path):
    CreateLog()

    image=cv2.imread(img_path)
    frame=classes.Frame(image)
    frame.show_img(waitKey=0)

    # find objects at frame  with yolo:
    frame.detect_objects()

    # cut objects from frame:
    frame.cut_objects()

    # find_kinds_with_model: options: model resnet_50 or vgg_16
    frame.model_detect('resnet_50')
    frame.model_detect('vgg_16')

    # draw rectangle on the object by kind object: (if its balloon- red rectangle with x)
    frame.result()

    # # add txt to the objects in the frame:
    # frame.add_txt()

    frame.show_img(waitKey=0)

    # save result:
    frame.save_frame("result_balloon.png")

    # write results to log file:
    frame.print_results_frame()

    return frame.frame

if __name__ == '__main__':
    image_path=r"pexels-this-is-zun-1680638.jpg"
    frame=detect_balloon_main(image_path)



