import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import cv2
import open3d as o3d
import math
import numpy as np
#import cv_bridge as cv


IMG_INPUT_SHAPE = [64, 180]

class LidarFeatureExtract:
    def __init__(self):
        #self.depth = rospy.Subscriber("/lidar_depth_image_array", UInt8MultiArray, self.show_depth_img_from_data)
        #self.bridge = cv.CvBridge() # <- fuck you

        self.pc_data = rospy.Subscriber("/os1_points", PointCloud2, self.store_lidar_data)
        self.batch_last_samples = np.empty((0,3), np.int32)#np.float32)
        self.size_batch = 0
        self.number_of_features = 4

    #def show_depth_img_from_data(self, data):
    #    img_arr = list(data.data)
    #    img_arr = np.array(img_arr)
    #    img_arr = img_arr.astype('uint8')
    #    img_arr = np.reshape(img_arr, (IMG_INPUT_SHAPE[0], IMG_INPUT_SHAPE[1]))
    #    cv2.imshow("Image", img_arr)
    #    cv2.waitKey(3)

    def store_lidar_data(self, data):
        points = np.array(list(read_points(data)))
        xyz = np.array([(x, y, z) for x, y, z, _, _ in points]) # assumes XYZIR
        if self.size_batch >= 10:
            #pcd.points = o3d.utility.Vector3dVector(self.batch_last_samples)
            #o3d.visualization.draw_geometries([pcd])
            
            self.batch_last_samples = np.delete(self.batch_last_samples , slice(0, xyz.shape[0]), axis=0)
            
        self.batch_last_samples = np.append(self.batch_last_samples, xyz, axis=0)
        self.size_batch += 1
        
    def extracted_lidar_features(self):
        pcd = o3d.geometry.PointCloud()
        extracted_features = np.empty((0,1), np.int32)
        extracted_features_points = np.empty((0,3), np.int32)
        print(self.batch_last_samples)
        if len(self.batch_last_samples) > 0:
            for n in range (self.number_of_features):
                print(self.batch_last_samples)
                sector = self.subdivide_pointcloud_to_sectors(self.batch_last_samples, n, self.number_of_features)
                pcd.points = o3d.utility.Vector3dVector(sector)
                o3d.visualization.draw_geometries([pcd])
                print(sector.shape())
                if sector.shape().shape[0] > 0 :
                    r, closesd_p = self.get_distance_to_closest_point(sector)
                    extracted_features = np.append(extracted_features, r)
                    extracted_features_points = np.vstack([extracted_features_points, closesd_p])
                
                pcd.points = o3d.utility.Vector3dVector(sector)
                o3d.visualization.draw_geometries([pcd])
        else:
            extracted_features = np.full(self.number_of_features, 3)
        #print(extracted_features)
        #print(extracted_features_points)
        
        #pcd.points = o3d.utility.Vector3dVector(extracted_features_points)
        #rgb = np.asarray([0.0, 255.0, 0.0])
        #rgb_t = np.transpose(rgb)/255.0
        #pcd.colors = o3d.utility.Vector3dVector([rgb_t, rgb_t, rgb_t, rgb_t])
        

        #pcd.points = o3d.utility.Vector3dVector(self.batch_last_samples)
        #o3d.visualization.draw_geometries([pcd])
        return extracted_features

    def reset_lidar_storage(self):
        print("empty lidar data")
        self.batch_last_samples = np.empty((0,3), np.float32) #clean all stored samples


    def subdivide_pointcloud_to_sectors(self, pc, sliceN, n_slices):
        sector = np.empty((0,3), np.int32)
        for xyz in pc:
            angle = math.atan2(xyz[1], xyz[0])
            angle = (angle + 2*math.pi) % (2*math.pi)
            if angle >= sliceN*2*math.pi/n_slices and angle < (sliceN+1)*2*math.pi/n_slices:
                sector = np.vstack([sector, xyz])
                # pop pc

        #sector = np.int32((math.pi + np.arctan2(pc[:, 1], pc[:, 0])) * (8 / (2*math.pi)))
        #print("Print sector info:")
        #print(sector.shape)
        #print(sector)

        return sector



    def get_distance_to_closest_point(self, xyz):
        #points = np.array(list(read_points(pc)))
        #xyz = np.array([(x, y, z) for x, y, z, _, _ in points]) # assumes XYZIR
        dist = 1000
        closesd_point = [10000, 10000, 10000]
        for d in xyz:
            pytag = d[0]**2 + d[1]**2 + d[2]**2
            if pytag <= dist:
                dist = pytag
                closesd_point = d

        r = np.linalg.norm(xyz, axis=-1)
        return np.min(r), closesd_point
