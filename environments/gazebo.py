#!/usr/bin/env python3
import rospy
from threading import Lock
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates
import numpy as np
from numpy.random import uniform
from time import sleep, time
from bs4 import BeautifulSoup


def randomPoint(bound):
    return uniform(-bound, bound, 2)


class Model(object):

    def __init__(self, name, pose=None, twist=None):
        self.model_name = name
        self.robot_namespace = name
        self.pose = pose or Pose()
        self.twist = twist or Twist()
        self.reference_frame = ''


class Gazebo(object):

    def __init__(self, bound=5., robot_space=.7, time_step=.2):
        self.name = 'husky'
        self.bound = bound
        self.state_dim = 5
        self.action_dim = 3
        self.id_in_world = 2
        self.robot_space = robot_space
        self.time_step = time_step
        self.target = randomPoint(self.bound)  # TODO: remove

        print("Waiting for Gazebo.")
        rospy.wait_for_service('/gazebo/set_model_state')
        print("Found Gazebo services.")

        self.temp = rospy.Subscriber(
            '/scan', LaserScan, self._initiate_lidar, queue_size=1)

        self.lidar_subscirber = rospy.Subscriber(
            '/scan', LaserScan, self._read, queue_size=1)
        self.pose_subscriber = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self._getPose, queue_size=1)

        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)

        self.observation = None
        self.msg = Twist()
        self.lidar_lock = Lock()
        self.pose_lock = Lock()

        print("Waiting to recieve a topic message ...")
        while self.observation is None:
            sleep(.01)
        print("Topic message recived!")
        print("Initialized lidar with %i readings" % len(self.observation))

    def getPerceptionDim(self):
        return len(self.locked_observation)

    def getStateDim(self):
        return self.state_dim

    def getActionDim(self):
        return self.action_dim

    def _getPose(self, data):
        yaw = np.arctan2(data.pose[self.id_in_world].orientation.w * data.pose[
                         self.id_in_world].orientation.z, .5 - data.pose[self.id_in_world].orientation.z**2)
        x, y = data.pose[self.id_in_world].position.x, data.pose[
            self.id_in_world].position.y
        forward = data.twist[self.id_in_world].linear.x
        turn = data.twist[self.id_in_world].angular.z
        with self.pose_lock:
            self.pose = x, y, yaw, forward, turn

    def _initiate_lidar(self, data):
        self.locked_observation = np.empty(len(data.ranges))
        with self.lidar_lock:
            self.observation = data.ranges
        self.temp.unregister()

    def _read(self, data):
        with self.lidar_lock:
            self.observation = data.ranges

    def get(self):
        with self.lidar_lock:
            self.locked_observation[:] = np.clip(self.observation, 0, 20)
        with self.pose_lock:
            diff = self.target - self.pose[:2]
            direction = np.arctan2(diff[1], diff[0]) - self.pose[2]
            forward, turn = self.pose[3:]

        direction = (direction + np.pi) % (2 * np.pi) - np.pi
        self.distance = np.linalg.norm(diff)

        reward = self._reward(self.distance, self.locked_observation)
        terminated = self._isTerminated(self.locked_observation, self.distance)

        return np.copy([self.distance, np.sin(direction), np.cos(direction), forward, turn]), np.copy(self.locked_observation), reward, any(terminated), terminated

    def _reward(self, distance, readings):
        return max(1 + self.robot_space - distance, -10) + \
            min(0, (min(readings) - self.robot_space * 2))  # - self.motion_cost

    def _isTerminated(self, readings, distance):
        return any(readings < self.robot_space), distance < self.robot_space

    def changePose(self, pose=None, facing=None):
        s = ModelState()
        s.model_name = self.name

        if pose is None:
            s.pose.position.x, s.pose.position.y = randomPoint(self.bound)
        else:
            s.pose.position.x, s.pose.position.y = pose

        if facing is None:
            # removed a multiplication of 2 in angle since it cancels with a
            # devision in `sin` and `cos`
            angle = np.pi * uniform()
        else:
            angle = facing * .5

        s.pose.orientation.w = np.cos(angle)
        s.pose.orientation.z = np.sin(angle)
        s.pose.position.z = .15

        self.state(s)
        rospy.wait_for_service('/gazebo/set_model_state')
        # sleep(.1)

    def reset(self, pose=None):
        self.target = randomPoint(self.bound)
        self.changePose(pose)

        sleep(1.)
        _get = self.get()
        start = time()
        while _get[-2]:
            if time() - start > 3.:
                print("reseting position ...")
                self.changePose(pose)
            _get = self.get()

        return _get

    def act(self, action):
        if action == 0:
            self.msg.linear.x = .5
            self.msg.angular.z = 0.
        elif action == 1:
            self.msg.linear.x = 0.
            self.msg.angular.z = .5
        elif action == 2:
            self.msg.linear.x = 0.
            self.msg.angular.z = -.5
        else:
            self.msg.linear.x = 0.
            self.msg.angular.z = 0.

        self.pub_cmd.publish(self.msg)

    def step(self, action):
        self.act(action)
        sleep(self.time_step)
        return self.get()

    def getModels(self):
        self.objects = []
        done = [False]

        def _getModels(data):
            objects = []
            for name, pose, twist in zip(data.name, data.pose, data.twist):
                if name in self.protected:
                    continue
                objects.append(Model(name, pose, twist))
            self.temp.unregister()
            self.objects = objects
            done[0] = True

        self.temp = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, _getModels, queue_size=1)
        while not done[0]:
            sleep(.01)
        return self.objects


class TenderGazebo(Gazebo):

    def __init__(self, *args):
        super(TenderGazebo, self).__init__(*args)
        self.previous = 0
        self.motion_cost = 0.
        self.action_dim = 3

    def act(self, action):
        if action == 0:
            self.msg.linear.x = .5
            self.msg.angular.z = 0.
            self.motion_cost = 0.

        elif action == 1:
            self.msg.linear.x = 0.
            if self.previous == 2:
                self.msg.angular.z = 0.
                self.motion_cost = .01
            else:
                self.msg.angular.z = .5
                self.motion_cost = 0.

        elif action == 2:
            self.msg.linear.x = 0.
            if self.previous == 1:
                self.msg.angular.z = 0.
                self.motion_cost = .01
            else:
                self.msg.angular.z = -.5
                self.motion_cost = 0.

        else:
            self.msg.linear.x = 0.
            self.msg.angular.z = 0.
            self.motion_cost = .01

        self.previous = action
        self.pub_cmd.publish(self.msg)

    def _reward(self, distance, readings):
        return max(1 + self.robot_space - distance, -10) + \
            min(0, (min(readings) - self.robot_space - 1)) - self.motion_cost


def isNear(point, point_list, space):
    close = [np.linalg.norm(
        (point[0] - x, point[1] - y)) < space for x, y in point_list]
    return any(close)


class DynamicGazebo(Gazebo):

    def __init__(self, *args):
        super(DynamicGazebo, self).__init__(*args)
        self.protected = ['willowgarage', 'ground_plane_0', 'husky']
        self.spawn = rospy.ServiceProxy(
            '/gazebo/spawn_sdf_model', SpawnModel)
        self.delete = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.obstacles = self.getModels()

    def spawnSphere(self, count=1, location=None):
        with open('/home/barzin/projects/ros_learning/src/simulation/obstacles/sphere.sdf') as f:
            sdf = f.read()
        # xml = BeautifulSoup(sdf, "lxml")
        all_poses = [(o.pose.position.x, o.pose.position.y)
                     for o in self.obstacles]
        all_names = [o.model_name for o in self.obstacles]
        for _ in range(count):
            pose = location
            name = 'sphere'
            i = 0
            while name + '_%i' % i in all_names:
                i += 1
            name = name + '_%i' % i
            all_names.append(name)

            model = Model(name)

            if pose is None:
                pose = randomPoint(self.bound)
                while isNear(pose, all_poses, 2 * self.robot_space):
                    pose = randomPoint(self.bound)

            model.pose.position.x, model.pose.position.y = pose
            model.pose.position.z = 2.
            self.obstacles.append(model)

            self.spawn(name, sdf, name, model.pose, "")

        rospy.wait_for_service('/gazebo/spawn_sdf_model')

    def clearEnvrionment(self):
        for m in self.obstacles:
            self.delete(m.model_name)
        rospy.wait_for_service('/gazebo/delete_model')
        self.obstacles = []

    def kick(self, name=None):
        for obstacle in self.obstacles:
            if obstacle.model_name in self.protected:
                continue
            obstacle.twist.linear.x, obstacle.twist.linear.y = np.random.randn(
                2) * .2
            self.state(obstacle)
        rospy.wait_for_service('/gazebo/set_model_state')

    def reset(self, pose=None, vis_target=False):
        all_poses = [(o.pose.position.x, o.pose.position.y)
                     for o in self.obstacles]

        self.target = randomPoint(self.bound)
        while isNear(self.target, all_poses, 2 * self.robot_space):
            self.target = randomPoint(self.bound)

        if vis_target:
            with open('/home/barzin/projects/ros_learning/src/simulation/obstacles/target.sdf') as f:
                sdf = f.read()
            name = 'target'
            model = Model(name)
            model.pose.position.x, model.pose.position.y = self.target
            model.pose.position.z = .05
            self.spawn(name, sdf, name, model.pose, "")
            self.obstacles.append(model)

        all_poses.append(self.target)

        point = randomPoint(self.bound)
        while isNear(point, all_poses, 2 * self.robot_space):
            point = randomPoint(self.bound)

        self.changePose(point)
        sleep(1.)

        _get = self.get()
        start = time()
        while _get[-2]:
            if time() - start > 3.:
                print("reseting position ...")
                self.changePose(pose)
                sleep(1.)

            _get = self.get()

        return _get


class GazeboDiscrete(Gazebo):

    def get(self):
        with self.lidar_lock:
            self.locked_observation[:] = np.clip(self.observation, 0, 20)
        with self.pose_lock:
            self.locked_pose[:] = self.pose

        diff = self.target - self.locked_pose[:2]
        direction = np.arctan2(diff[1], diff[0]) - self.locked_pose[2]
        direction = (direction + np.pi) % (2 * np.pi) - np.pi
        distance = np.linalg.norm(diff)

        reward =  -.05 + (self.robot_space > distance) - \
            any(self.locked_observation < self.robot_space)

        if any(self.locked_observation < self.robot_space):
            start = time()
            while time() - start < 1.:
                if any(self.locked_observation < self.robot_space):
                    sleep(.1)
        terminated = any(self.locked_observation <
                         self.robot_space) or reward > 0

        return (distance, np.sin(direction), np.cos(direction)), self.locked_observation, reward, terminated


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rospy.init_node("simulation_wrapper")
    env = DynamicGazebo()
    # env.clearEnvrionment()
    # sleep(1.)
    # env.spawnSphere(5)
    env.kick()
    env.reset()
