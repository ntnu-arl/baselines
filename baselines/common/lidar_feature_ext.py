import rospy
from sensor_msgs.msg import Image
import cv2
#import cv2 as cv2
import cv_bridge as cv
import numpy as np
import struct

class LidarFeatureExtract:
    def __init__(self):
        self.lidar_data = rospy.Subscriber("/lidar_depth_image", Image, self.print_data)
        self.bridge = cv.CvBridge()

    def print_data(self, data):
        print(type(data))
        #nparr = np.fromstring(data.data, np.uint8)
        #img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img_ipl = cv.CreateImageHeader((img_np.shape[1], img_np.shape[0]), cv.IPL_DEPTH_8U, 3)
        #img = np.array(struct.unpack('c'*data.height*data.width, data.data))
        #print(img)
        #img[~np.isnan(img)] = img[~np.isnan(img)] / np.nanmax(img)
        #img[~np.isnan(img)] *= 255
        #img = img.astype('uint8')
        #img = img.reshape((data.height,data.width))
        #print(img)
        #try:
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")#"mono8 bgr8")
        #
        #except cv.CvBridgeError as e:
        #   print(e)
        #   return
        print("done")
        #cv2.imshow("image", image)
        cv2.imshow("Image window", img_ipl)
        cv2.waitKey(3)