import socket
import time
import rtde_receive
import rtde_control
import numpy as np
import struct
from datetime import datetime
import csv
# import keyboard as key
from visual_kinematics.RobotSerial import *
from math import pi
import h5py
from aloha_scripts.constants import DT, HOME_POSE, MASTER_IP, FOLLOWER_IP


"""
Class to control the actual robot (UR5e)
"""
class Follower():
    def __init__(self,ip):
        self.ip = ip
        self.receive = rtde_receive.RTDEReceiveInterface(self.ip)
        self.control = rtde_control.RTDEControlInterface(self.ip)
    
    def move2Home(self):
        print("Moving Follower to Home Pose")
        self.control.moveJ(HOME_POSE, 1.500, 0.05, False)

    def move2Pose(self,trajectoryList):
        for trajectory in trajectoryList:
            self.control.moveJ(trajectory, 3.14, 0.5, False)
    
    def getJointAngles(self):
        angles = self.receive.getActualQ()
        return angles
    
    def getTCPPosition(self):
        pose = self.receive.getActualTCPPose()
        return pose
    
    def getJointVelocity(self):
        vels = self.receive.getActualQd()
        return vels
    
    def getJointEffort(self):
        # efforts = self.receive.getJointTorques()
        return np.zeros(6)
    
    def operate(self,masterJoints):
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0/50  # 2ms can use 1/50
        lookahead_time = 0.2 # can use 0.09
        gain = 300
        # t_start = self.control.initPeriod()
        self.control.servoJ(masterJoints, velocity, acceleration, dt, lookahead_time, gain)
        # self.control.waitPeriod(t_start)

    def disconnect(self):
        # print("Disconnecting robot")
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        
    

"""
Class to control the replica robot
"""
class Master():
    def __init__(self,ip):

        self.ip = ip
        self.receiver_address = (ip, 5000)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.receiver_address)
        self.dh_params = np.array(
        [
            [0.1625, 0.0, 0.5 * pi, 0.0],
            [0.0, -0.425, 0, 0.0],
            [0.0, -0.3922, 0, 0.0],
            [0.1333, 0.0, 0.5 * pi, 0.0],
            [0.0997, 0.0, -0.5 * pi, 0.0],
            [0.0996, 0.0, 0.0, 0.0],
        ]
                                    )
        self.replica = RobotSerial(self.dh_params)
    
    def connect(self):
        self.server_socket.listen(1)
        print("Waiting for a connection...")
        self.client_socket, self.client_address = self.server_socket.accept()
        print("Connected by", self.client_address)

    def disconnect(self):
        self.client_socket.close()
        self.server_socket.close()

    def getTCPPosition(self,masterJointAngles):
        forward = self.replica.forward(masterJointAngles)
        xyz = forward.t_3_1.reshape([3,])
        rxryrz = forward.r_3
        replicaPose = np.concatenate((xyz,rxryrz)) 
        return replicaPose
    
    def getJointAngles(self):
        data = self.client_socket.recv(24)
        encoder2Angles = list(struct.unpack('6f', data))
        return encoder2Angles


"""
Class to record and save observations
"""
class Data():
    def __init__(self,fileName):
        self.fileName = fileName
    
    def write2CSV(self,dataArray):
        current_date = datetime.now().strftime('%m-%d_%H-%M')
        filename = f"{self.fileName}_{current_date}.csv"
        heading = ['Sr.no.', 'J1', ' J2', 'J3', 'J4', 'J5', 'J6', 'Timestamp']
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(heading)
            for row in dataArray:
                csv_writer.writerow(row)
    
    ## TODO:
    def write2h5py(self):
        return

## TODO:
"""
Class to check for collisions
"""
class Collision():
    def __init__(self, check,xlim,ylim,zlim,axis):
        self.check = check
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.axis = axis
    
    def detect(self,followerTCP,masterTCP):
        if self.check:
            if self.axis == "pos":
                pass
            if self.axis == "neg":
                pass
        else:
            return  


"""
Function to generate trajectory from follower to master position
"""
def generateTrajectory(masterPose, followerPose, moveTime = 1):
    steps = int(moveTime/0.02)
    waypoints = np.linspace(followerPose,masterPose,steps)
    return waypoints



def main():
    # initialize and connect master
    master = Master(MASTER_IP)
    master.connect()

    # iniialize and connect follower
    follower = Follower(FOLLOWER_IP)

    # move follower to home pose
    follower.move2Home()

    # move follower arm to same position as master
    masterJoints = master.getJointAngles()
    followerJoints = follower.getJointAngles()
    safeTrajectory = generateTrajectory(masterJoints,followerJoints)
    for joint in safeTrajectory:
        follower.operate(joint)

    # teleoperation 
    while True:
        try:
            joints2Follow = master.getJointAngles()
            follower.operate(joints2Follow)
            qPos = follower.getJointAngles()
            qVel = follower.getJointVelocity()

            # if key.is_pressed('c'):
            #     pass
            
        except (KeyboardInterrupt,BrokenPipeError,ConnectionResetError):
            print("bye bye ")
            follower.stop()
            master.disconnect()
            break
        
        # except follower.receive.RTDEException as e:
        #     print("RTDE error: {}".format(e))
        #     follower.stop()
        #     master.disconnect()
        #     break
    
if __name__ == "__main__":

    main()