import time
import sys
import math
import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
        SportClient,
        PathPoint,
        SPORT_PATH_POINT_SIZE,

)
from unitree_sdk2py.go2.video.video_client import VideoClient


class Robot:
    def __init__(self):
        # Time count
        self.t = 0.0
        self.dt = 0.01

        # Initial position and yaw
        self.px0 = 0.0
        self.py0 = 0.0
        self.yaw0 = 0.0

        # Actual position
        self.px = self.px0
        self.py = self.py0

        # Rotation angles for balanceMode
        self.yaw_rot = 0.0
        self.roll_rot = 0.0
        self.pitch_rot = 0.0
        self.angle_step = 3.5

        # Initial velocity
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        # Creating the sport client
        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        
        # Creating the video client
        self.videoClient = VideoClient()
        self.videoClient.SetTimeout(2.0)
        self.videoClient.Init()

        self.max_yaw_angle = 35
        self.max_pitch_angle = 45
        self.max_roll_angle = 40


    def getInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]

    def standUp(self):
        self.client.StandUp()
        print("Stand Up!")

    def standDown(self):
        self.client.StandDown()
        print("Stand Down!")

    def balanceStand(self):
        self.standUp()
        self.client.BalanceStand()
        print("Balance stand mode")

    def releaseMotors(self):
        self.client.Damp()
        print("Released the motors!")

    def setSpeed(self, vx: float, vy: float, vyaw: float):
        self.client.Move(vx, vy, vyaw)
        self.vx = vx
        self.vy = vy
        self.vyaw = vyaw
        print(f"Speed set to: \n\tvx: {vx}\n\tvy: {vy}\n\tvyaw: {vyaw}")

    def moveToPoint(self, p_x: float, p_y: float, p_yaw):
        """
                This method makes the robot move to the coordinates (x, y)
            and then rotates it with p_yaw radians
        """
        pass


    def rotate_rad(self, roll: float = 0, pitch: float = 0, yaw: float = 0):
        """
            Note: To make the robot rotate, put it first in balanceStand mode
        """
        code = self.client.Euler(roll, pitch, yaw)
        print(f"Robot rotated with:\n\troll: {roll}\n\tpitch: {pitch}\n\tyaw: {yaw}")
        print(f"Code returned: {code}")


    def get_front_camera_image(self):
        # Get the image data from the robot
        code, data = self.videoClient.GetImageSample()
        #Convert it to a usable cv2 image
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image

    def follow_point(self, x0, y0, x, y, precision):
        """
           This function make the robot, while in balanceMode follow a certain point
            x0, y0, represent the center of the camera
            x, y represent the target point
            Note: The robot needs to previously be but in balanceMode
        """
        if math.sqrt((x0 - x)**2 + (y0 - y)**2) >= precision:
            if x0 < x:
                if -self.max_yaw_angle < self.yaw_rot:
                    self.yaw_rot -= self.angle_step
                else:
                    self.yaw_rot = -self.max_yaw_angle
            else:
                if self.max_yaw_angle > self.yaw_rot:
                    self.yaw_rot += self.angle_step
                else:
                    self.yaw_rot = self.max_yaw_angle

            if y0 < y:
                if self.max_pitch_angle > self.pitch_rot:
                    self.pitch_rot += self.angle_step
                else:
                    self.pitch_rot = self.max_pitch_angle
            else:
                if -self.max_pitch_angle < self.pitch_rot:
                    self.pitch_rot -= self.angle_step
                else:
                    self.pitch_rot = -self.max_pitch_angle

        self.rotate_rad(0, self.pitch_rot * math.pi / 180, self.yaw_rot * math.pi / 180)



def get_mask_for_color(frame, lower_bgr, upper_bgr):
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.inRange(frame, lower_bgr, upper_bgr)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, kernel, iterations = 3)
    return mask

def detect_blobs(frame, lower_bgr, upper_bgr, min_area):
    mask = get_mask_for_color(frame, lower_bgr, upper_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = 0

    x0 = 0
    y0 = 0

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cnt += 1
            # Compute the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            #print(x, y, w, h)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Blob: {cnt}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255), -1)

            if cnt == 1:
                x0 = x + w / 2
                y0 = y + h / 2

    return frame, x0, y0



robot_state = unitree_go_msg_dds__SportModeState_()

def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg


lower_bgr = np.array([30, 50, 0])
upper_bgr = np.array([140, 160, 40])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)


    print("Start")
    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    # Initializing the robot
    robot = Robot()
    robot.getInitState(robot_state)

    robot.standDown()
    time.sleep(4)


    robot.balanceStand()



    # Main robot loop
    while True:
        robot.t += robot.dt

        # Code goes here
        image = robot.get_front_camera_image()

        height, width, channels = image.shape

        blob_image, x, y = detect_blobs(image, lower_bgr, upper_bgr, 10000)
        cv2.circle(image, (int(width / 2), int(height / 2)), 5, (0, 0, 255), -1)

        if x != 0 and y != 0:
            robot.follow_point(width / 2, height / 2, x, y, 100)
        else:
            robot.rotate_rad(0, robot.pitch_rot * math.pi / 180, robot.yaw_rot * math.pi / 180);

        cv2.imshow("Front Camera", blob_image)

        # To here

        if cv2.waitKey(20) == 27:
            break

        time.sleep(robot.dt)


    robot.rotate_rad(0, 0, 0)
    robot.standDown()
