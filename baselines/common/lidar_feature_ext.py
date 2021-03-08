import rospy
import time
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import cv2
import open3d as o3d
import math
import numpy as np

#IMG_INPUT_SHAPE = [64, 180]

class LidarFeatureExtract:
    
    number_of_features = 8
    bach_size_pc = 10
    
    def __init__(self, feature_size, bach_size_pc):
        #self.depth = rospy.Subscriber("/lidar_depth_image_array", UInt8MultiArray, self.show_depth_img_from_data)
        self.pc_data = rospy.Subscriber("/os1_points", PointCloud2, self.store_lidar_data)
        self.batch_last_samples = np.empty((1,3), np.float32)
        self.size_batch = 0
        self.bach_size_pc = bach_size_pc
        self.number_of_features = feature_size
        self.extracted_features = np.empty((self.number_of_features,1), np.float32)
    
    #def show_depth_img_from_data(self, data):
        #show depth image without cv_bridge
    #    img_arr = list(data.data)
    #    img_arr = np.array(img_arr)
    #    img_arr = img_arr.astype('uint8')
    #    img_arr = np.reshape(img_arr, (IMG_INPUT_SHAPE[0], IMG_INPUT_SHAPE[1]))
    #    cv2.imshow("Image", img_arr)
    #    cv2.waitKey(3)

    def store_lidar_data(self, data):
        points = np.array(list(read_points(data)))
        xyz = np.array([(x, y, z) for x, y, z, _, _ in points]) # assumes XYZIR
        xyz = self.filter_points(xyz, -10, 10)

        if self.size_batch >= self.bach_size_pc:
            #self.vis_points(self.batch_last_samples)
            self.batch_last_samples = np.delete(self.batch_last_samples , slice(0, xyz.shape[0]), axis=0)

        if xyz.shape[0] > 0:
            self.batch_last_samples = np.vstack([self.batch_last_samples, xyz])
        
        self.size_batch += 1

    def filter_points(self, xyz, min_axis, max_axis):
        #reduce computation time by removing points very far away
        
        xyz = np.delete(xyz, xyz[:,0] > max_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,0] < min_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,1] > max_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,1] < min_axis, axis=0)
        #xyz = np.delete(xyz, xyz[:,2] > max_axis, axis=0)
        #xyz = np.delete(xyz, xyz[:,2] < min_axis, axis=0)
        
        return xyz
    
    def extracted_lidar_features(self):
        pcd = o3d.geometry.PointCloud()
        extracted_features_points = np.empty((0,3), np.int32)
        if len(self.batch_last_samples) > 0:
            pc = self.batch_last_samples #needed this due to concurrency issues
            index_sector = self.subdivide_pointcloud_to_sectors(self.batch_last_samples)
            for n in range(self.number_of_features):  
                if (len(index_sector[n]) > 0):
                    sector = np.delete(pc, np.array(index_sector[n]), 0)
                if sector.shape[0] > 0 :
                    distance, _ = self.get_distance_to_closest_point(sector)
                    self.extracted_features[n] = distance
                    #extracted_features_points = np.vstack([extracted_features_points, closesd_p])
                
                #self.vis_points(sector)
        else:
            self.extracted_features = np.full(self.number_of_features, 10.0)

        #self.vis_points(self.batch_last_samples)
        return self.extracted_features

    def reset_lidar_storage(self):
        #clean all stored samples
        
        self.batch_last_samples = np.empty((0,3), np.float32) 
        self.extracted_features = np.empty((self.number_of_features,1), np.float32)
        self.size_batch = 0


    def subdivide_pointcloud_to_sectors(self, pc):
        #divdes sphere of a point cloid in equal sectors
        
        i = 0
        index_sector = [[] for _ in range(self.number_of_features)]

        for xyz in pc:
            theta = math.atan2(xyz[1], xyz[0])
            theta = (theta + 2*math.pi) % (2*math.pi)
            
            r = math.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
            phi = math.acos(xyz[2]/r)  # 90 deg is planar with rmf
            
            phi_bound = math.pi*(15/24) #phi boundary set to ca 113 deg
            pi_div = 2*math.pi/self.number_of_features #speed up comp
            
            for sliceN in range(self.number_of_features):
                if (theta < sliceN*pi_div or theta >= (sliceN+1)*pi_div) and phi < phi_bound: 
                    index_sector[sliceN].append(i)
                    continue
                
            i += 1 
        
        return index_sector



    def get_distance_to_closest_point(self, xyz):
        dist = 1000
        closesd_point = [10000, 10000, 10000]
        #for d in xyz:
        #    pytag = d[0]**2 + d[1]**2 + d[2]**2
        #    if pytag <= dist:
        #        dist = pytag
        #        closesd_point = d

        r = np.linalg.norm(xyz, axis=-1)
        return np.min(r), closesd_point

    def vis_points(self, pc):
        #visualize points with open3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        
        #adding colors
        #rgb = np.asarray([0.0, 255.0, 0.0])
        #rgb_t = np.transpose(rgb)/255.0
        #pcd.colors = o3d.utility.Vector3dVector([rgb_t, rgb_t, rgb_t, rgb_t])
        
        o3d.visualization.draw_geometries([pcd])