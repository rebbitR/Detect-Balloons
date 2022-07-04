import cv2

# left = box[0]
# bottom = box[1]
# right = box[2] + box[0]
# top = box[3] + box[1]

def yolo_detect(frame):
    classFile='yolo_file/coco.names'

    with open(classFile,'rt') as f:
        classNames=f.read().rstrip('\n').split('\n')
    configPath='yolo_file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath='yolo_file/frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath,configPath)

    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(frame)

    types=[]
    places = []
    my_confs=[]
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            myPlace=[box[0],box[1],box[2],box[3]]
            places.append(myPlace)
            types.append(classNames[classId-1])
            my_confs.append(confidence)
    return places,types,my_confs










