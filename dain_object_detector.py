import numpy as np
import cv2
import time
from config import *

def show_bounding_boxes(image, boxes, classes, scores, save=False):
    # Convert original image to RGB format
    clone = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    H,W = clone.shape[:2]
    # color = np.random.randint(0,100,size=(3))
    colors_to_BGR = {
        'yellow':[0,100,100],
        'red':[0,0,100],
        'green':[0,100,0],
        'blue':[100,0,0],
        'pink':[50,50,100],
    }

    for i, (startY, startX, endY, endX) in enumerate(boxes):
        startX, startY, endX, endY = int(startX * W), int(startY * H), int(endX * W), int(endY * H)
        # print('Object', i, 'at', startX, ',', startY, ',', endX, ',', endY)
        
        # draw the bounding box and label on the image
        label = str(int(classes[i]))
        score = str(scores[i])
        if scores[i] > 0.5:
            try:
                if int(label) in CUP_ID_TO_NAME.keys():
                    color = colors_to_BGR[CUP_ID_TO_NAME[int(label)].split(' ')[0]]
                elif int(label) in BIN_ID_TO_NAME.keys():
                    color = colors_to_BGR[BIN_ID_TO_NAME[int(label)].split(' ')[0]]
            except:
                print('NO COLOR FOR LABEL', label)
                color = [50,50,50]
            cv2.rectangle(clone, (startX, startY), (endX, endY), [int(color[0]),int(color[1]),int(color[2])], 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (int(startX-10), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [int(color[0]),int(color[1]),int(color[2])], 2)
            cv2.putText(clone, score, (int(endX), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [int(color[0]),int(color[1]),int(color[2])], 2)
    
    if save:
        cv2.imwrite("detection{}.png".format(round(time.time())), clone)
    
    print('displaying image')
    cv2.imshow("Objects Detected", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return clone

def show_image(image):
    print('displaying image')
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()