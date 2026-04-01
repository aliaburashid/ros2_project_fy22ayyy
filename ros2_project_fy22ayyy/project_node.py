# COMP3631 Robotics Project
# project_node.py

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from math import sin, cos, radians
import time

# Waypoints are (x, y, yaw) in the map frame
# yaw in radians: 0=East, pi/2=North, pi=West, 3pi/2 (or -pi/2)=South
# WP2 from original plan moved to end as the final observation point
WAYPOINTS = [
    (0.0,   0.0,   radians(0)),      # WP 0: starting position, facing east
    (-0.5,  -0.5,  radians(0)),      # WP 1: centre of starting room
    (6.0,   -4.95, radians(0)),      # WP 2: right side of the map
    (7.24,  -12.7, radians(90)),     # WP 3: lower right corner, facing north
    (-8.7,  -4.79, radians(180)),    # WP 4: left side of the map, facing west
    (-5.23,  0.868, radians(0)),     # WP 5: upper left area
    (-10.5,  1.26,  radians(180)),   # WP 6: far left wall area, facing west
    (-4.93, -8.17, radians(270)),    # WP 7: final observation point, facing south
]


class ProjectNode(Node):

    def __init__(self):
        super().__init__('project_node')

        # cv_bridge converts ROS Image messages into OpenCV-compatible format
        # ref: https://github.com/ros-perception/vision_opencv
        self.bridge = CvBridge()

        # flags to track which colours have been detected
        # set to True on first detection, used to avoid repeated log messages
        self.red_seen   = False
        self.green_seen = False
        self.blue_found = False  # reset every frame, True only when blue is currently visible

        # x pixel position of the blue box centre in the camera image
        # used to calculate steering error during approach
        self.blue_cx = None

        # width of the camera frame in pixels, set on first frame received
        self.image_width = None

        # closest obstacle distance from laser scan in metres
        # used to stop the robot when it is within 1 m of the blue box
        self.min_distance = float('inf')

        # state machine controls the robot behaviour at a high level:
        # exploring      -> navigating through waypoints
        # searching_red  -> at final point, spinning to find red box
        # searching_blue -> red found, spinning to find blue box
        # approaching    -> driving toward the blue box
        # done           -> stopped within 1 m, task complete
        self.state = 'exploring'

        # Nav2 action client sends NavigateToPose goals to the navigation stack
        # Nav2 handles path planning and obstacle avoidance automatically
        # ref: https://navigation.ros.org/commander_api/index.html
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # publisher for direct velocity commands used during the approach phase
        # geometry_msgs/Twist: linear.x = forward speed, angular.z = turning speed
        # TurtleBot3 Burger limits: max linear 0.22 m/s, max angular 2.84 rad/s
        # ref: https://docs.ros2.org/latest/api/geometry_msgs/
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.waypoint_index  = 0      # tracks which waypoint to send next
        self.nav_goal_active = False  # prevents sending a new goal while one is active
        self.goal_handle     = None   # stored so the goal can be cancelled if needed

        # stores the latest camera frame for processing in the main loop
        # processing here instead of in the callback reduces image lag
        self.latest_image = None

        # QoS set to BEST_EFFORT and KEEP_LAST depth 1 so only the most recent
        # camera frame is kept, avoiding a backlog of old unprocessed frames
        # ref: https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # subscribe to laser scan for proximity and obstacle detection
        # ref: https://docs.ros2.org/latest/api/sensor_msgs/
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # subscribe to the camera topic using the low-lag QoS profile above
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, camera_qos)

        self.get_logger().info('='*50)
        self.get_logger().info('  ProjectNode started. Beginning exploration...')
        self.get_logger().info('='*50)

    def laser_callback(self, data):
        # filter out inf and NaN readings which indicate no return from the sensor
        # then take the minimum of the remaining values as the closest obstacle
        valid = [r for r in data.ranges if not (r == float('inf') or r != r)]
        if valid:
            self.min_distance = min(valid)

    def camera_callback(self, data):
        try:
            # convert ROS image message to OpenCV BGR format using cv_bridge
            full = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            # resize to 50% to speed up HSV conversion and contour detection
            # ref: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
            self.latest_image = cv2.resize(full, (0, 0), fx=0.5, fy=0.5)
        except CvBridgeError:
            pass

    def process_vision(self):
        if self.latest_image is None or self.state == 'done':
            return

        image = self.latest_image.copy()
        self.image_width = image.shape[1]

        # convert BGR to HSV colour space for more reliable colour detection
        # HSV separates hue from brightness, making it less sensitive to lighting changes
        # note: OpenCV uses hue range 0-180, not the standard 0-360
        # ref: https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        s = 15  # sensitivity value, widens the hue detection band for each colour

        # create binary masks for each colour using inRange
        # pixels within the HSV range become 255, everything else becomes 0
        # ref: https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html
        masks = {
            # red wraps around the hue circle so two ranges are combined with bitwise_or
            # lower red: hue 0 to s, upper red: hue 180-s to 180
            'Red': cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0,     100, 100]), np.array([s,   255, 255])),
                cv2.inRange(hsv, np.array([180-s, 100, 100]), np.array([180, 255, 255]))),
            # green is centred at hue 60 in OpenCV scale
            'Green': cv2.inRange(hsv, np.array([60-s,  100, 100]), np.array([60+s,  255, 255])),
            # blue is centred at hue 120 in OpenCV scale
            'Blue':  cv2.inRange(hsv, np.array([120-s, 100, 100]), np.array([120+s, 255, 255])),
        }

        # reset each frame so blue_found only reflects current visibility
        self.blue_found = False

        for label, mask in masks.items():
            # colour used for the circle outline drawn around each detected box
            circle_colour = (0, 0, 255) if label == 'Red' else \
                            (0, 255, 0) if label == 'Green' else \
                            (255, 0, 0)

            # find contours in the binary mask
            # RETR_LIST returns all contours, CHAIN_APPROX_SIMPLE compresses edges
            # ref: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # pick the largest contour by area, most likely the target box
                c = max(contours, key=cv2.contourArea)

                # ignore small contours under 500 px to filter out noise
                if cv2.contourArea(c) > 500:

                    # get the centre and radius of the smallest circle enclosing the contour
                    # ref: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    cx, cy = int(x), int(y)

                    # draw colour-coded circle outline around the detected box
                    cv2.circle(image, (cx, cy), int(radius), circle_colour, 3)

                    # white text label for visibility against any background colour
                    cv2.putText(image, label, (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # log each colour detection once only
                    if label == 'Red' and not self.red_seen:
                        self.red_seen = True
                        self.get_logger().info('─'*40)
                        self.get_logger().info('  DETECTED: Red Box')
                        self.get_logger().info('─'*40)

                    if label == 'Green' and not self.green_seen:
                        self.green_seen = True
                        self.get_logger().info('─'*40)
                        self.get_logger().info('  DETECTED: Green Box')
                        self.get_logger().info('─'*40)

                    if label == 'Blue':
                        self.blue_found = True
                        self.blue_cx = cx
                        # hasattr check logs the blue detection only on first occurrence
                        if not hasattr(self, '_blue_logged'):
                            self._blue_logged = True
                            self.get_logger().info('─'*40)
                            self.get_logger().info('  DETECTED: Blue Box')
                            self.get_logger().info('─'*40)

        cv2.imshow('Robot_Vision', image)
        cv2.waitKey(1)

    def stop_robot_hard(self):
        # cancel the current nav2 goal if one is active
        if self.goal_handle:
            self.goal_handle.cancel_goal_async()

        # publish zero velocity 10 times to make sure the robot fully stops
        # a single publish can be dropped if the subscriber misses it
        stop = Twist()
        for _ in range(10):
            self.vel_pub.publish(stop)
            time.sleep(0.05)

    def send_next_waypoint(self):
        # wait up to 2 seconds for nav2 to be ready before sending a goal
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            return

        x, y, yaw = WAYPOINTS[self.waypoint_index]

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id  = 'map'
        goal.pose.pose.position.x  = x
        goal.pose.pose.position.y  = y

        # convert yaw (rotation around z axis) to quaternion
        # for 2D navigation only the z and w components are needed
        # formula: z = sin(yaw/2), w = cos(yaw/2)
        # ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        goal.pose.pose.orientation.z = sin(yaw / 2)
        goal.pose.pose.orientation.w = cos(yaw / 2)
        goal.pose.header.stamp       = self.get_clock().now().to_msg()

        self.get_logger().info(
            f'  Navigating to Waypoint {self.waypoint_index + 1}/{len(WAYPOINTS)}: ({x}, {y})')

        self.nav_goal_active = True

        # send goal asynchronously, chain callbacks to handle completion
        # send_goal_async -> goal accepted -> get_result_async -> nav_done
        self.nav_client.send_goal_async(goal).add_done_callback(
            lambda f: f.result().get_result_async().add_done_callback(self.nav_done))

    def nav_done(self, future):
        self.nav_goal_active = False
        self.get_logger().info(f'  Waypoint {self.waypoint_index} reached.')
        self.waypoint_index += 1

        # once all waypoints are done switch to scanning at the final position
        if self.waypoint_index == len(WAYPOINTS):
            self.state = 'searching_red'
            self.get_logger().info('  Final observation point reached. Scanning...')


def main():
    rclpy.init()
    node = ProjectNode()

    # spin runs in a background thread so ROS callbacks are handled continuously
    # while the main loop below controls the robot behaviour
    # daemon=True means this thread exits automatically when the main thread ends
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            node.process_vision()

            if node.state == 'exploring':
                # send the next waypoint once the previous one is complete
                if not node.nav_goal_active and node.waypoint_index < len(WAYPOINTS):
                    node.send_next_waypoint()

            elif node.state == 'searching_red':
                # spin anticlockwise until the red box is detected
                # angular.z positive = anticlockwise in ROS convention
                if node.red_seen:
                    node.state = 'searching_blue'
                    node.get_logger().info('  Red found. Now searching for Blue...')
                else:
                    t = Twist()
                    t.angular.z = 0.4
                    node.vel_pub.publish(t)

            elif node.state == 'searching_blue':
                # spin clockwise until the blue box is detected
                # angular.z negative = clockwise in ROS convention
                if node.blue_found:
                    node.state = 'approaching'
                    node.get_logger().info('  Blue found. Approaching...')
                else:
                    t = Twist()
                    t.angular.z = -0.4
                    node.vel_pub.publish(t)

            elif node.state == 'approaching':
                # stop when the laser reads 1.05 m or less to the nearest obstacle
                if node.min_distance <= 1.05:
                    node.stop_robot_hard()
                    node.state = 'done'
                    node.get_logger().info('='*50)
                    node.get_logger().info(
                        f'  MISSION SUCCESS: Stopped at {round(node.min_distance, 2)} m from Blue Box')
                    node.get_logger().info('='*50)
                    break

                t = Twist()
                t.linear.x = 0.12  # forward at 0.12 m/s, within the 0.22 m/s TurtleBot3 limit

                if node.blue_cx is not None and node.image_width is not None:
                    # proportional steering based on how far the blue box is from the image centre
                    # error is positive if blue is right of centre, negative if left
                    # normalising by half the image width keeps the error in range -1 to 1
                    # multiplying by -0.5 converts to angular velocity (negative turns right)
                    error = node.blue_cx - (node.image_width / 2)
                    t.angular.z = -float(error) / (node.image_width / 2) * 0.5
                node.vel_pub.publish(t)

            time.sleep(0.05)  # ~20 Hz loop rate

    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot_hard()
        cv2.destroyAllWindows()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()