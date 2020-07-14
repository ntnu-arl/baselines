import time
import rospy
import numpy as np
import random
from rotors_control.msg import StateAction
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ContactsState, ModelState
from std_srvs.srv import Empty, EmptyRequest
from gym import core, spaces
from gym.utils import seeding
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class RotorsWrappers:
    def __init__(self):
        # ROS publishers/subcribers
        self.state_action_subscriber = rospy.Subscriber("/delta/state_action", StateAction, self.state_action_callback)
        self.contact_subcriber = rospy.Subscriber("/delta/delta_contact", ContactsState, self.contact_callback)
        self.odom_subscriber = rospy.Subscriber('/delta/odometry_sensor1/odometry', Odometry, self.odom_callback)

        self.goal_training_publisher = rospy.Publisher("/delta/goal_training", Pose)
        self.goal_init_publisher = rospy.Publisher("/delta/goal", Pose)
        self.cmd_publisher = rospy.Publisher("/delta/command/rate_thrust", RateThrust)
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.sphere_marker_pub = rospy.Publisher('goal_published',
                                                 MarkerArray,
                                                 queue_size=1)

        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.current_goal = None
        self.get_params()

        # Imitiate Gym variables
        action_high = np.array([self.MAX_ACC_X, self.MAX_ACC_Y, self.MAX_ACC_Z], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        state_high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-state_high, high=state_high, dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {"t_start": time.time(), "env_id": "rotors-rmf"}

        self.received_state_action = False
        self.obs = None
        self.new_obs = None
        self.reward = None
        self.action = None
        self.done = False
        self.collide = False
        self.timeout = False
        self.info = None
        self.state_action_msg = None
        self.robot_odom = None
        self.msg_cnt = 0

        self.timeout_timer = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_params(self):
        self.initial_goal_generation_radius = rospy.get_param('initial_goal_generation_radius',
                                                      1.0)
        self.set_goal_generation_radius(self.initial_goal_generation_radius)
        self.waypoint_radius = rospy.get_param('waypoint_radius', 0.2)
        self.robot_collision_frame = rospy.get_param(
            'robot_collision_frame',
            'delta::delta/base_link::delta/base_link_fixed_joint_lump__delta_collision_collision'
        )
        self.ground_collision_frame = rospy.get_param(
            'ground_collision_frame', 'ground_plane::link::collision')
        self.Q_state = rospy.get_param('Q_state', [60.0, 60.0, 100.0, 15.0, 15.0, 25.0])
        self.Q_state = np.array(list(self.Q_state))
        self.Q_state = np.diag(self.Q_state)
        print('Q_state:', self.Q_state)
        self.R_action = rospy.get_param('R_action', [0.35, 0.35, 5])
        self.R_action = np.diag(self.R_action)
        print('R_action:', self.R_action)
        self.R_action = np.array(list(self.R_action))
        self.goal_reward = rospy.get_param('goal_reward', 1000.0)
        self.time_penalty = rospy.get_param('time_penalty', 0.0)
        self.obstacle_max_penalty = rospy.get_param('obstacle_max_penalty', 1000.0)
        self.obstacle_th_distance = rospy.get_param('obstacle_th_distance', 0.5)
        self.obstacle_weight = rospy.get_param('obstacle_weight', 0.0)

        self.MAX_ACC_X = rospy.get_param('max_acc_x', 1.0)
        self.MAX_ACC_Y = rospy.get_param('max_acc_y', 1.0)
        self.MAX_ACC_Z = rospy.get_param('max_acc_z', 1.0)

        self.max_z_train = rospy.get_param('max_z_train', 5.0)

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

        self.clear_state_action_flag()
        while (not self.received_new_state_action()):
            time.sleep(0.001)
        return self.get_state_action()

    def odom_callback(self, msg):
        self.robot_odom = msg.pose.pose

    def state_action_callback(self, data):
        self.msg_cnt = self.msg_cnt + 1
        #if (self.msg_cnt >= 10): # throttle
        if (self.msg_cnt >= 1):
            self.msg_cnt = 0
            self.state_action_msg = data
            self.received_state_action = True
        return

    def clear_state_action_flag(self):
        self.msg_cnt = 0
        self.received_state_action = False

    def received_new_state_action(self):
        return self.received_state_action

    def get_state_action(self):
        data = self.state_action_msg

        self.new_obs = np.array([data.next_goal_odom.pose.pose.position.x,
        data.next_goal_odom.pose.pose.position.y,
        data.next_goal_odom.pose.pose.position.z,
        data.next_goal_odom.twist.twist.linear.x,
        data.next_goal_odom.twist.twist.linear.y,
        data.next_goal_odom.twist.twist.linear.z])
        #print('new_obs:', self.new_obs)

        # unnormalized actions
        self.action = np.array([data.action.thrust.x, data.action.thrust.y, data.action.thrust.z])

        self.obs = np.array([data.goal_odom.pose.pose.position.x,
        data.goal_odom.pose.pose.position.y,
        data.goal_odom.pose.pose.position.z,
        data.goal_odom.twist.twist.linear.x,
        data.goal_odom.twist.twist.linear.y,
        data.goal_odom.twist.twist.linear.z])

        Qx = self.Q_state.dot(self.new_obs)
        xT_Qx = self.new_obs.transpose().dot(Qx)
        Ru = self.R_action.dot(self.action)
        uT_Ru = self.action.transpose().dot(Ru)
        self.reward = -xT_Qx - uT_Ru

        self.info = {'status':'none'}
        self.done = False

        # reach goal?
        if np.linalg.norm(self.new_obs[0:3]) < self.waypoint_radius:
            self.reward = self.reward + self.goal_reward
            #self.done = True
            self.generate_new_goal()
            self.info = {'status':'reach goal'}
            print('reach goal!')

        # collide?
        if self.collide:
            self.collide = False
            self.reward = self.reward - self.obstacle_max_penalty
            self.done = True
            self.info = {'status':'collide'}

        # z increases too much?
        if self.robot_odom.position.z > self.max_z_train:
            self.reward = self.reward - self.obstacle_max_penalty
            self.done = True
            self.info = {'status':'invalid_z'}

        # time out?
        if self.timeout:
            self.timeout = False
            #self.done = True
            self.generate_new_goal()
            print('timeout')
            self.info = {'status':'timeout'}

        # fake multiple environments -> return np.array([[.]])
        return (self.new_obs, self.reward, self.done, self.info)

    def contact_callback(self, msg):
        # Check inside the models states for robot's contact state
        for i in range(len(msg.states)):
            if (msg.states[i].collision1_name == self.robot_collision_frame):
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

    def generate_new_goal(self):
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
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi

        if (z + self.robot_odom.position.z < 0.5):
            z = 0.5 - self.robot_odom.position.z
        elif (z + self.robot_odom.position.z > self.max_z_train - 0.5):
            z = self.max_z_train - 0.5 - self.robot_odom.position.z

        rospy.loginfo_throttle(2, 'New Goal: (%.3f , %.3f , %.3f)', x, y, z)
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.orientation.x = 0
        goal.orientation.y = 0
        goal.orientation.z = 0
        goal.orientation.w = 1
        # Convert this goal into the world frame and set it as the current goal
        # self.current_goal = self.transform_pose_to_world(goal)
        self.draw_new_goal(goal)

        self.goal_training_publisher.publish(goal)

        self.reset_timer()

        return

    def draw_new_goal(self, p):
        markerArray = MarkerArray()
        count = 0
        MARKERS_MAX = 20
        marker = Marker()
        marker.header.frame_id = "/delta/base_link"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = p.position.x
        marker.pose.position.y = p.position.y
        marker.pose.position.z = p.position.z

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

    def stop_robot(self):
        # make the robot stay at current position
        goal = Pose()
        goal.position.x = 0
        goal.position.y = 0
        goal.position.z = 0
        goal.orientation.x = 0
        goal.orientation.y = 0
        goal.orientation.z = 0
        goal.orientation.w = 1
        self.goal_init_publisher.publish(goal)

    def timer_callback(self, event):
        #print('timeout')
        self.timeout = True

    def set_goal_generation_radius(self, radius):
        self.goal_generation_radius = radius

    def get_goal_generation_radius(self):
        return self.goal_generation_radius

    def pause(self):
        rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

    def unpause(self):
        rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

    def reset(self):
        rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

        # randomize initial position (TODO: angle?, velocity?)
        state_high = np.array([1.0, 1.0, 3.0], dtype=np.float32)
        state_low = np.array([-1.0, -1.0, 2.0], dtype=np.float32)
        state_init = self.np_random.uniform(low=state_low, high=state_high, size=(3,))

        # Fill in the new position of the robot
        new_position = ModelState()
        new_position.model_name = 'delta'
        new_position.reference_frame = 'world'
        new_position.pose.position.x = state_init[0]
        new_position.pose.position.y = state_init[1]
        new_position.pose.position.z = state_init[2]
        new_position.pose.orientation.x = 0
        new_position.pose.orientation.y = 0
        new_position.pose.orientation.z = 0
        new_position.pose.orientation.w = 1
        new_position.twist.linear.x = 0
        new_position.twist.linear.y = 0
        new_position.twist.linear.z = 0
        new_position.twist.angular.x = 0
        new_position.twist.angular.y = 0
        new_position.twist.angular.z = 0
        rospy.loginfo('Placing robot')
        self.model_state_publisher.publish(new_position)

        self.collide = False
        self.timeout = False
        self.done = False
        self.received_state_action = False

        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()

        rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

        return np.array([state_init[0], state_init[1], state_init[2], 0.0, 0.0, 0.0])

    def reset_timer(self):
        #rospy.loginfo('Resetting the timeout timer')
        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()
            self.timeout_timer = rospy.Timer(rospy.Duration(self.goal_generation_radius * 10), self.timer_callback)

    def render(self):
        return None

    def close(self):
        pass

if __name__ == '__main__':

    rospy.loginfo('Ready')
    rospy.spin()
