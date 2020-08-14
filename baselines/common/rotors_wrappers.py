import time
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import collections
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ContactsState, ModelState
from std_srvs.srv import Empty, EmptyRequest
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray
from gym import core, spaces
from gym.utils import seeding

OB_ROBOT_STATE_SHAPE = 8
OB_PCL_SHAPE = (8, 45, 4)
PCL_FEATURE_SIZE = OB_PCL_SHAPE[0] * OB_PCL_SHAPE[1] * OB_PCL_SHAPE[2]

class RotorsWrappers:
    def __init__(self):
        rospy.init_node('rotors_wrapper', anonymous=True)

        self.current_goal = None
        self.get_params()

        # Imitiate Gym variables
        action_high = np.array([self.max_acc_x, self.max_acc_y, self.max_acc_z], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        #state_high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
        #                np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max],
        #                dtype=np.float32)
        state_robot_high = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, np.pi/2, np.pi/2], dtype=np.float32)
        pcl_feature_high = 10 * np.ones(PCL_FEATURE_SIZE, dtype=np.float32)
        state_robot_low = -state_robot_high
        pcl_feature_low = 0 * np.ones(PCL_FEATURE_SIZE, dtype=np.float32)

        state_high = np.concatenate((state_robot_high, pcl_feature_high), axis=None)
        state_low = np.concatenate((state_robot_low, pcl_feature_low), axis=None)

        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        self.ob_pcl_shape = OB_PCL_SHAPE
        self.ob_robot_state_shape = OB_ROBOT_STATE_SHAPE
        self.pcl_feature_size = PCL_FEATURE_SIZE
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {"t_start": time.time(), "env_id": "rotors-rmf"}

        self.done = False
        self.timeout = False
        self.timeout_timer = None

        self.robot_odom = collections.deque([]) 
        self.msg_cnt = 0
        self.pcl_feature = np.array([])

        self.sleep_rate = rospy.Rate(self.control_rate)

        self.seed()

        # ROS publishers/subcribers
        self.contact_subcriber = rospy.Subscriber("/delta/delta_contact", ContactsState, self.contact_callback)
        self.odom_subscriber = rospy.Subscriber('/delta/odometry_sensor1/odometry', Odometry, self.odom_callback)
        self.pcl_feature_subscriber = rospy.Subscriber('/lidar_depth_feature', Float64MultiArray, self.pcl_feature_callback)

        self.goal_training_publisher = rospy.Publisher("/delta/goal_training", Pose)
        self.goal_in_vehicle_publisher = rospy.Publisher("/delta/goal_in_vehicle", Odometry)
        self.goal_init_publisher = rospy.Publisher("/delta/goal", Pose)
        self.cmd_publisher = rospy.Publisher("/delta/command/rate_thrust", RateThrust)
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.sphere_marker_pub = rospy.Publisher('goal_published',
                                                 MarkerArray,
                                                 queue_size=1)

        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_params(self):
        self.initial_goal_generation_radius = rospy.get_param('initial_goal_generation_radius', 5.0)
        self.set_goal_generation_radius(self.initial_goal_generation_radius)
        self.waypoint_radius = rospy.get_param('waypoint_radius', 0.25)
        self.robot_collision_frame = rospy.get_param(
            'robot_collision_frame',
            'delta::delta/base_link::delta/base_link_fixed_joint_lump__delta_collision_collision'
        )
        self.ground_collision_frame = rospy.get_param(
            'ground_collision_frame', 'ground_plane::link::collision')
        self.Q_state = rospy.get_param('Q_state', [0.6, 0.6, 1.0, 0.03, 0.03, 0.05])
        self.Q_state = np.array(list(self.Q_state))
        self.Q_state = np.diag(self.Q_state)
        print('Q_state:', self.Q_state)
        self.R_action = rospy.get_param('R_action', [0.001, 0.001, 0.001])
        self.R_action = np.diag(self.R_action)
        print('R_action:', self.R_action)
        self.R_action = np.array(list(self.R_action))
        self.goal_reward = rospy.get_param('goal_reward', 1.0)
        self.time_penalty = rospy.get_param('time_penalty', 0.0)
        self.obstacle_max_penalty = rospy.get_param('obstacle_max_penalty', 1.0)

        self.max_acc_x = rospy.get_param('max_acc_x', 1.0)
        self.max_acc_y = rospy.get_param('max_acc_y', 1.0)
        self.max_acc_z = rospy.get_param('max_acc_z', 1.0)

        self.max_wp_x = rospy.get_param('max_waypoint_x', 10.0)
        self.max_wp_y = rospy.get_param('max_waypoint_y', 10.0)
        self.max_wp_z = rospy.get_param('max_waypoint_z', 6.0) 

        self.min_wp_x = rospy.get_param('min_waypoint_x', -10.0)
        self.min_wp_y = rospy.get_param('min_waypoint_y', -10.0)
        self.min_wp_z = rospy.get_param('min_waypoint_z', 1.0)                

        self.min_init_z = rospy.get_param('min_initial_z', 2.0)
        self.max_init_z = rospy.get_param('max_initial_z', 4.0)

        self.control_rate = rospy.get_param('control_rate', 20.0)

    def step(self, action):
        command = RateThrust()
        command.header.stamp = rospy.Time.now()
        command.angular_rates.x = 0.0
        command.angular_rates.y = 0.0
        command.angular_rates.z = 0.0
        # action = np.array([[.]])
        command.thrust.x = action[0][0]
        command.thrust.y = action[0][1]
        command.thrust.z = action[0][2]
        self.cmd_publisher.publish(command)

        # ros sleep 50ms
        self.sleep_rate.sleep()

        # get new obs
        new_obs = self.get_new_obs()
        # calculate reward
        action = np.array([command.thrust.x, command.thrust.y, command.thrust.z])
        Qx = self.Q_state.dot(new_obs[0:6])
        xT_Qx = new_obs[0:6].transpose().dot(Qx) / 250.0
        Ru = self.R_action.dot(action)
        uT_Ru = action.transpose().dot(Ru) / 250.0
        reward = - uT_Ru
        #reward = -0.01

        info = {'status':'none'}
        self.done = False        
        
        # reach goal?
        if (np.linalg.norm(new_obs[0:3]) < self.waypoint_radius) and (np.linalg.norm(new_obs[3:6]) < 0.3):
            reward = reward + self.goal_reward
            self.done = True
            info = {'status':'reach goal'}
            print('reach goal!')
        else:
            reward = reward - xT_Qx    
            pass

        # collide?
        if self.collide:
            self.collide = False
            reward = reward - self.obstacle_max_penalty
            self.done = True
            info = {'status':'collide'}        

        # time out?
        if self.timeout:
            self.timeout = False
            self.done = True
            print('timeout')
            info = {'status':'timeout'}

        return (new_obs, reward, self.done, info)

    def get_new_obs(self):
        if (len(self.robot_odom) > 0) and (len(self.pcl_feature) > 0):
            current_odom = self.robot_odom[0]
            goad_in_vehicle_frame, robot_euler_angles = self.transform_goal_to_vehicle_frame(current_odom, self.current_goal)
            new_obs = np.array([goad_in_vehicle_frame.pose.pose.position.x,
            goad_in_vehicle_frame.pose.pose.position.y,
            goad_in_vehicle_frame.pose.pose.position.z,
            goad_in_vehicle_frame.twist.twist.linear.x,
            goad_in_vehicle_frame.twist.twist.linear.y,
            goad_in_vehicle_frame.twist.twist.linear.z,
            robot_euler_angles[2], # roll [rad]
            robot_euler_angles[1]]) # pitch [rad]
            new_obs = np.concatenate((new_obs, self.pcl_feature), axis=None)
        else:
            new_obs = None
        return new_obs

    def odom_callback(self, msg):
        #print("received odom msg")
        self.robot_odom.appendleft(msg)
        if (len(self.robot_odom) > 10): # save the last 10 odom msg
            self.robot_odom.pop()

    def pcl_feature_callback(self, msg):
        arr = list(msg.data)
        arr = np.array(arr)
        if (arr.size == PCL_FEATURE_SIZE):
            self.pcl_feature = arr

    def contact_callback(self, msg):
        # Check inside the models states for robot's contact state
        for i in range(len(msg.states)):
            if (msg.states[i].collision1_name == self.robot_collision_frame):
                print('Contact found!')
                rospy.logdebug('Contact found!')
                self.collide = True
                if (msg.states[i].collision2_name ==
                        self.ground_collision_frame):
                    rospy.logdebug('Robot colliding with the ground')
                else:
                    rospy.logdebug(
                        'Robot colliding with something else (not ground)')
                    #self.reset()
            else:
                rospy.logdebug('Contact not found yet ...')

    # Input:    robot_odom  : Odometry()
    #           goal        : Pose(), in vehicle frame
    # Return:   current_goal  : Pose(), in world frame
    def transform_goal_to_world_frame(self, robot_odom, goal):
        current_goal = Pose()
        
        r_goal = R.from_quat([goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w])
        goal_euler_angles = r_goal.as_euler('zyx', degrees=False)
        
        robot_pose = robot_odom.pose.pose
        r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
        robot_euler_angles = r_robot.as_euler('zyx', degrees=False)

        r_goal_in_world = R.from_euler('z', goal_euler_angles[0] + robot_euler_angles[0], degrees=False)
        goal_pos_in_vehicle = np.array([goal.position.x, goal.position.y, goal.position.z])
        robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        goal_pos_in_world = R.from_euler('z', robot_euler_angles[0], degrees=False).as_matrix().dot(goal_pos_in_vehicle) + robot_pos
        # print('R abc:', R.from_euler('z', robot_euler_angles[0], degrees=False).as_matrix())
        # print('goal_pos_in_vehicle:', goal_pos_in_vehicle)
        # print('robot_pos:', robot_pos)
        # print('goal_pos_in_world:', goal_pos_in_world)

        current_goal.position.x = goal_pos_in_world[0]
        current_goal.position.y = goal_pos_in_world[1]
        current_goal.position.z = goal_pos_in_world[2]

        current_goal_quat = r_goal_in_world.as_quat()
        current_goal.orientation.x = current_goal_quat[0]
        current_goal.orientation.y = current_goal_quat[1]
        current_goal.orientation.z = current_goal_quat[2]
        current_goal.orientation.w = current_goal_quat[3]

        return current_goal

    # Input:    robot_odom  : Odometry()
    #           goal        : Pose(), in world frame
    # Return:   goal_odom   : Odometry(), in vehicle frame
    #           robot_euler_angles: np.array(), zyx order
    def transform_goal_to_vehicle_frame(self, robot_odom, goal):
        goal_odom = Odometry()
        
        r_goal = R.from_quat([goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w])
        goal_euler_angles = r_goal.as_euler('zyx', degrees=False)
        
        robot_pose = robot_odom.pose.pose
        r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
        robot_euler_angles = r_robot.as_euler('zyx', degrees=False)

        r_goal_in_vechile = R.from_euler('z', goal_euler_angles[0] - robot_euler_angles[0], degrees=False)
        goal_pos = np.array([goal.position.x, goal.position.y, goal.position.z])
        robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        goal_pos_in_vehicle = R.from_euler('z', -robot_euler_angles[0], degrees=False).as_matrix().dot((goal_pos - robot_pos))
        # print('goal_pos:', goal_pos)
        # print('robot_pos:', robot_pos)
        # print('R:', R.from_euler('z', -robot_euler_angles[0], degrees=False).as_matrix())

        goal_odom.header.stamp = robot_odom.header.stamp
        goal_odom.header.frame_id = "vehicle_frame"
        goal_odom.pose.pose.position.x = goal_pos_in_vehicle[0]
        goal_odom.pose.pose.position.y = goal_pos_in_vehicle[1]
        goal_odom.pose.pose.position.z = goal_pos_in_vehicle[2]
        goal_quat_in_vehicle = r_goal_in_vechile.as_quat()
        goal_odom.pose.pose.orientation.x = goal_quat_in_vehicle[0]
        goal_odom.pose.pose.orientation.y = goal_quat_in_vehicle[1]
        goal_odom.pose.pose.orientation.z = goal_quat_in_vehicle[2]
        goal_odom.pose.pose.orientation.w = goal_quat_in_vehicle[3]

        goal_odom.twist.twist.linear.x = -robot_odom.twist.twist.linear.x
        goal_odom.twist.twist.linear.y = -robot_odom.twist.twist.linear.y
        goal_odom.twist.twist.linear.z = -robot_odom.twist.twist.linear.z
        goal_odom.twist.twist.angular.x = -robot_odom.twist.twist.angular.x
        goal_odom.twist.twist.angular.y = -robot_odom.twist.twist.angular.y
        goal_odom.twist.twist.angular.z = -robot_odom.twist.twist.angular.z

        self.goal_in_vehicle_publisher.publish(goal_odom)

        return goal_odom, robot_euler_angles

    # Input:    robot_pose  : Pose()
    # Return:   current_goal    : Pose(), in world frame
    #           r               : float
    def generate_new_goal(self, robot_pose):
        # Generate and return a pose in the sphere centered at the robot frame with radius as the goal_generation_radius

        # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/5838055#5838055
        goal = Pose()
        # sphere_marker_array = MarkerArray()
        u = random.random()
        v = random.random()
        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        while np.isnan(phi):
            phi = np.arccos(2.0 * v - 1.0)
        r = self.goal_generation_radius * np.cbrt(random.random())
        if r < 3.0:
            r = 3.0
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi

        # limit z of goal
        [x, y, z] = np.clip([x,y,z], [self.min_wp_x - goal.position.x, self.min_wp_y - goal.position.y, self.min_wp_z - goal.position.z],
                                    [self.max_wp_x - goal.position.x, self.max_wp_y - goal.position.y, self.max_wp_z - goal.position.z])


        # rospy.loginfo_throttle(2, 'New Goal: (%.3f , %.3f , %.3f)', x, y, z)
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.orientation.x = 0
        goal.orientation.y = 0
        goal.orientation.z = 0
        goal.orientation.w = 1
        # Convert this goal into the world frame and set it as the current goal
        robot_odom = Odometry()
        robot_odom.pose.pose = robot_pose
        current_goal = self.transform_goal_to_world_frame(robot_odom, goal)

        return current_goal, r

    def draw_new_goal(self, p):
        markerArray = MarkerArray()
        count = 0
        MARKERS_MAX = 20
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose = p

        rospy.loginfo('Draw new goal: (%.3f , %.3f , %.3f)', p.position.x, p.position.y, p.position.z)

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
        self.sphere_marker_pub.publish(markerArray)

        count += 1

    def timer_callback(self, event):
        self.timeout = True

    def set_goal_generation_radius(self, radius):
        self.goal_generation_radius = radius

    def get_goal_generation_radius(self):
        return self.goal_generation_radius

    def pause(self):
        #rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

    def unpause(self):
        #rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

    def reset(self):
        # check if the start position collides with env
        start_pose, collide = self.spawn_robot(None)
        while collide:
            #rospy.loginfo('INVALID start pose: (%.3f , %.3f , %.3f)', start_pose.position.x, start_pose.position.y, start_pose.position.z)
            start_pose, collide = self.spawn_robot(None)

        #rospy.loginfo('New start pose: (%.3f , %.3f , %.3f)', start_pose.position.x, start_pose.position.y, start_pose.position.z)
        
        # check if the end position collides with env: fix it, so stupid!
        goal, r = self.generate_new_goal(start_pose)
        _, collide = self.spawn_robot(goal)
        while collide:
            #rospy.loginfo('INVALID end goal: (%.3f , %.3f , %.3f)', goal.position.x, goal.position.y, goal.position.z)
            goal, r = self.generate_new_goal(start_pose)
            _, collide = self.spawn_robot(goal)
        
        #rospy.loginfo('New end goal: (%.3f , %.3f , %.3f)', goal.position.x, goal.position.y, goal.position.z)

        # put the robot at the start pose
        self.spawn_robot(start_pose)

        self.current_goal = goal
        self.draw_new_goal(goal)
        self.goal_training_publisher.publish(goal)
        self.reset_timer(r * 3)

        obs = self.get_new_obs()
        return obs        

    # Input:    position  : Pose()
    # Return:   position  : Pose(), in world frame
    #           collide   : bool
    def spawn_robot(self, pose = None):
        #rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

        new_position = ModelState()
        new_position.model_name = 'delta'
        new_position.reference_frame = 'world'
        
        # Fill in the new position of the robot
        if (pose == None):
            # randomize initial position (TODO: angle?, velocity?)
            state_high = np.array([self.max_wp_x, self.max_wp_y, self.max_init_z], dtype=np.float32)
            state_low = np.array([self.min_wp_x, self.min_wp_y, self.min_init_z], dtype=np.float32)
            state_init = self.np_random.uniform(low=state_low, high=state_high, size=(3,))
            new_position.pose.position.x = state_init[0]
            new_position.pose.position.y = state_init[1]
            new_position.pose.position.z = state_init[2]
            new_position.pose.orientation.x = 0
            new_position.pose.orientation.y = 0
            new_position.pose.orientation.z = 0
            new_position.pose.orientation.w = 1
        else:
            new_position.pose = pose    
        # Fill in the new twist of the robot
        new_position.twist.linear.x = 0
        new_position.twist.linear.y = 0
        new_position.twist.linear.z = 0
        new_position.twist.angular.x = 0
        new_position.twist.angular.y = 0
        new_position.twist.angular.z = 0
        #rospy.loginfo('Placing robot')
        self.model_state_publisher.publish(new_position)

        self.collide = False
        self.timeout = False
        self.done = False
        self.msg_cnt = 0

        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()

        #rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

        self.robot_odom.clear()
        self.pcl_feature = np.array([])
        self.collide = False

        rospy.sleep(0.1) # wait for robot to get new odometry
        while (len(self.robot_odom) == 0) or (len(self.pcl_feature) == 0):
            rospy.sleep(0.001)
            pass        
        return new_position.pose, self.collide

    def reset_timer(self, time):
        #rospy.loginfo('Resetting the timeout timer')
        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()
        # self.timeout_timer = rospy.Timer(rospy.Duration(self.goal_generation_radius * 5), self.timer_callback)
        if time <= 0:
            time = 1.0
        self.timeout_timer = rospy.Timer(rospy.Duration(time), self.timer_callback)

    def render(self):
        return None

    def close(self):
        pass

if __name__ == '__main__':

    rospy.loginfo('Ready')
    rospy.spin()
