import rospy
import time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import math


class LidarFeatureExtract:

    sector_size = 8
    stack_size = 3
    bach_size_pc = 10
    vis_pc = False

    def __init__(self, sector_size, stack_size, bach_size_pc):
        self.pc_data = rospy.Subscriber("/os1_points", PointCloud2, self.store_pc_data)
        self.pc_data_stored = rospy.Publisher('lidar_data_stored', PointCloud2, queue_size=1)
        self.pc_features_publisher = rospy.Publisher('lidar_features', MarkerArray, queue_size=1)

        self.batch_last_samples = np.empty((0,3), np.float32)
        self.size_batch = 0
        self.bach_size_pc = bach_size_pc

        self.max_dist_search = 20.0

        self.number_of_stacks = stack_size
        self.number_of_sectors = sector_size
        self.number_of_features = sector_size * stack_size

        self.extracted_features = np.full(self.number_of_features, self.max_dist_search)
        self.extracted_features_points = np.empty((0,3), np.int32)

        self.store_data = False
        self.vis_pc = False


    def store_pc_data(self, data):
        points = np.array(list(read_points(data)))
        xyz = np.array([(x, y, z) for x, y, z, _, _ in points]) # assumes XYZIR

        if xyz.size > 0 and self.store_data:
            xyz = self.filter_points(xyz, -self.max_dist_search, self.max_dist_search)

            if self.size_batch >= self.bach_size_pc:
                #self.vis_points(self.batch_last_samples)
                self.batch_last_samples = np.delete(self.batch_last_samples , slice(0, xyz.shape[0]), axis=0)

            self.batch_last_samples = np.vstack([self.batch_last_samples, xyz])
            xyz = np.empty((0,3), np.int32)

            self.extracted_lidar_features()

            self.size_batch += 1

        #visualize the filtered points in rviz
        if self.vis_pc:
            self.xyz_array_to_pointcloud2(self.batch_last_samples)


    def filter_points(self, xyz, min_axis, max_axis):
        '''
        Reduce computation time by removing points very far away
        '''
        xyz = np.delete(xyz, xyz[:,0] > max_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,0] < min_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,1] > max_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,1] < min_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,2] > max_axis, axis=0)
        xyz = np.delete(xyz, xyz[:,2] < min_axis, axis=0)

        return xyz


    def xyz_array_to_pointcloud2(self, points, stamp=False, frame_id="delta"):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points and publishes it.
        '''
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp #rospy.Time.now()
        if frame_id:
            msg.header.frame_id = frame_id
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12*points.shape[0]
        msg.is_dense = int(np.isfinite(points).all())
        msg.data = np.asarray(points, np.float32).tostring()

        self.pc_data_stored.publish(msg)


    def extracted_lidar_features(self):
        '''
        We run through the entire pcl and find all sectors and stacks
        '''
        self.extracted_features_points = np.empty((1,3), np.int32)
        if self.batch_last_samples.size > 0:
            pc = self.batch_last_samples #make copy. Needed this due to concurrency issues
            feature_index = 0
            index_stack = np.empty((0,1), np.int32)
            if self.number_of_sectors > 1:
                #divide pcl in sectors in xy plane
                index_sector = self.subdivide_pointcloud_to_sectors(pc)

            for n in range(self.number_of_sectors):
                if (len(index_sector[n]) > 0):
                    sector = np.delete(pc, np.array(index_sector[n]), 0)
                else:
                    #all points lies in one sector
                    sector = pc

                if self.number_of_stacks > 1:
                    #divide sectors in stacks along z axis
                    index_stack = self.subdivde_sector_to_stacks(sector)

                for k in range(self.number_of_stacks):
                    self.create_pcl_feature_vector(sector, index_sector, index_stack, n, k, feature_index)

                    feature_index += 1

        else:
            #no pcl found so we create init feature vector
            self.extracted_features = np.full(self.number_of_features, self.max_dist_search)
    

    def create_pcl_feature_vector(self, pcl_sector, index_sector, index_stack, n, k, feature_index):
        '''
        We populate the pcl feature vector with the shortest distances from each stack (and sector)
        '''
        if (len(index_sector[n]) > 0):
            if self.number_of_stacks > 1 and len(index_stack[k]) > 0:
                stack = np.delete(pcl_sector, np.array(index_stack[k]), 0)
            else:
                #all points lies in one stack
                stack = pcl_sector

            if (pcl_sector.size > 0 and self.number_of_stacks <= 1) or (stack.size > 0 and self.number_of_stacks > 1):
                distance, closesd_p = self.get_distance_to_closest_point(stack)
                self.extracted_features[feature_index] = distance
                self.extracted_features_points = np.vstack([self.extracted_features_points, closesd_p])
            else:
                self.extracted_features[feature_index] = self.max_dist_search

        else:
            #no sector found so we set init feature
            self.extracted_features[feature_index] = self.max_dist_search


    def reset_lidar_storage(self):
        '''
        Clean all stored samples and set to init values
        '''
        self.batch_last_samples = np.empty((0,3), np.float32) #np.full((1,3), -100.0)
        self.extracted_features_points = np.empty((1,3), np.int32)
        self.extracted_features = np.full(self.number_of_features, self.max_dist_search)
        self.size_batch = 0
        markerArray = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        markerArray.markers.append(marker)
        self.pc_features_publisher.publish(markerArray)


    def subdivde_sector_to_stacks(self, sector):
        '''
        Divdes sector of points in equal stacks along the z - axiz.
        Using sphere coordinates: phi = [0 pi] rad.
        We only find and store indexes to speed up computation.
        '''
        i = 0
        index_stack = [[] for _ in range(self.number_of_stacks)]

        #most points in the pcl are in between [70 110] deg
        upper_clip = math.pi - 11/18*math.pi #180 - 110 = 70 deg
        lower_clip = 7/18*math.pi #70 deg
        upper_boundary = 0

        for xyz in sector:
            r = math.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
            phi = math.acos(xyz[2]/r)
            #setting boundary for where the pcl should be divided
            phi_bound = (math.pi - lower_clip - upper_clip)/self.number_of_stacks

            for stackN in range(self.number_of_stacks):
                if stackN == self.number_of_stacks - 1:
                    #need to only set upper boundary at the last stack
                    upper_boundary = upper_clip
                else:
                    upper_boundary = 0

                if phi < stackN*(phi_bound) + lower_clip or phi >= (stackN+1)*phi_bound + lower_clip + upper_boundary:
                    index_stack[stackN].append(i)

            i += 1

        return index_stack



    def subdivide_pointcloud_to_sectors(self, pc):
        '''
        Divdes sphere of points in equal sectors using sphere coordinates.
        We only find and store indexes to speed up computation.
        '''
        i = 0
        index_sector = [[] for _ in range(self.number_of_sectors)]

        for xyz in pc:
            theta = math.atan2(xyz[1], xyz[0])
            theta = (theta + 2*math.pi) % (2*math.pi)

            r = math.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
            pi_div = 2*math.pi/self.number_of_sectors

            for sliceN in range(self.number_of_sectors):
                if (theta < sliceN*pi_div or theta >= (sliceN+1)*pi_div):
                    index_sector[sliceN].append(i)

            i += 1

        return index_sector


    def get_distance_to_closest_point(self, xyz):
        '''
        Finds the closest distance (and the point) to the rmf with respect to the rmf frame

        Input:    xyz  : np.array (dim: N x 3)

        Output: np.min(r) : int #closest distance
                xyz[index[0][0]] : np.array (dim: 1 x 3) #closest point

        '''
        r = np.linalg.norm(xyz, axis=-1)
        index = np.where(r == np.min(r))

        return np.min(r), xyz[index[0][0]]


    def mark_feature_points(self, robot_odom, pc_points):
        '''
        We want to visualize the pc points extracted and used
        as states in rviz.
        Input:    robot_odom  : Odometry() #for transformation purposes
                  pc_points   : np.array (dim: self.number_of_features x 3) #all feature points
        '''
        markerArray = MarkerArray()
        MARKERS_MAX = self.number_of_features
        count = 0

        if pc_points.size > 0:
            for xyz in pc_points:
                if xyz.size == 3:
                    if xyz[0] == 0.0 and xyz[1] == 0.0 and xyz[2] == 0.0:
                        continue #point does not exist and was set to 0,0,0 by default
                    point = Pose()
                    point.position.x = xyz[0]
                    point.position.y = xyz[1]
                    point.position.z = xyz[2]
                    point.orientation.x = 0
                    point.orientation.y = 0
                    point.orientation.z = 0
                    point.orientation.w = 1
                    world_point = self.transform_points_to_world_frame(robot_odom, point)

                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.type = marker.SPHERE
                    marker.action = marker.ADD
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2
                    marker.color.a = 1.0
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.pose = world_point

                    # We add the new marker to the MarkerArray, removing the oldest
                    # marker from it when necessary
                    if (count > MARKERS_MAX):
                        markerArray.markers.pop(0)

                    markerArray.markers.append(marker)
                    # Renumber the marker IDs
                    id = 0
                    for m in markerArray.markers:
                        m.id = id
                        id += 1

                    # Publish the MarkerArray
                    self.pc_features_publisher.publish(markerArray)

                    count += 1

    def transform_points_to_world_frame(self, robot_odom, feature_point):
        '''
        Input:    robot_odom  : Odometry()
                  point        : Pose(), in vehicle frame
        Return:   current_point  : Pose(), in world frame
        '''
        current_point = Pose()

        r_point = R.from_quat([feature_point.orientation.x, feature_point.orientation.y, feature_point.orientation.z, feature_point.orientation.w])
        point_euler_angles = r_point.as_euler('zyx', degrees=False)

        robot_pose = robot_odom.pose.pose
        r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
        robot_euler_angles = r_robot.as_euler('zyx', degrees=False)

        r_point_in_world = R.from_euler('z', point_euler_angles[0] + robot_euler_angles[0], degrees=False)
        point_pos_in_vehicle = np.array([feature_point.position.x, feature_point.position.y, feature_point.position.z])
        robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        point_pos_in_world = R.from_euler('z', robot_euler_angles[0], degrees=False).as_matrix().dot(point_pos_in_vehicle) + robot_pos

        current_point.position.x = point_pos_in_world[0]
        current_point.position.y = point_pos_in_world[1]
        current_point.position.z = point_pos_in_world[2]

        current_point_quat = r_point_in_world.as_quat()
        current_point.orientation.x = current_point_quat[0]
        current_point.orientation.y = current_point_quat[1]
        current_point.orientation.z = current_point_quat[2]
        current_point.orientation.w = current_point_quat[3]

        return current_point

    def set_vis_in_rviz(self, set):
        '''
        Visualize points with rviz
        '''
        self.vis_pc = set

    def vis_points(self, pc):
        '''
        Visualize points with open3d
        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        #adding colors. Need to color each point...
        #rgb = np.asarray([0.0, 255.0, 0.0])
        #rgb_t = np.transpose(rgb)/255.0
        #pcd.colors = o3d.utility.Vector3dVector([rgb_t, rgb_t, rgb_t, rgb_t])

        o3d.visualization.draw_geometries([pcd])
