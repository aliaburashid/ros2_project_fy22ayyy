import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, LaserScan, LaserScan
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
from math import sin, cos, radians
import signal

# ── Angle reference ────────────────────────────────────────────────────────────
# 0°   = East
# 90°  = North
# 180° = West
# 270° = South  ← robot default facing direction
#
# Clockwise     = decreasing degrees (270 → 150 → 60)
# Counterclockwise = increasing degrees (60 → 210)
#
# ROS yaw uses radians with same convention.
# ──────────────────────────────────────────────────────────────────────────────

# ── Exploration waypoints (x, y, yaw_in_radians) ──────────────────────────────
#
# Step 1: Start at (-0.809, -0.783), facing 270° (South)
# Step 2: Rotate CLOCKWISE to 150° — intermediate sweep point
# Step 3: Rotate CLOCKWISE to 60°  — face NE, detect RED from (-0.50, -4.84)
# Step 4: Rotate COUNTERCLOCKWISE to 210° — face SW toward green
# Step 5: Move to GREEN box (4.47, -7.86)
# Step 6: Navigate toward BLUE box (-4.06, -9.66), stop 1 m away

WAYPOINTS = [
    # Step 1 — start position, facing 270° (South)
    (-0.81, -0.78,  radians(270 - 360)),   # 270° = -90° in ROS

    # Step 2 — rotate clockwise to 150°, stay in place
    (-0.81, -0.78,  radians(150)),

    # Step 3 — rotate clockwise to 60°, move to red detection point
    # facing 60° (North-East) from (-0.50, -4.84) to see red box
    (-0.50, -4.84,  radians(60)),

    # Step 4 — rotate counterclockwise from 60° to 210° (South-West)
    # facing toward green box area
    (-0.50, -4.84,  radians(210 - 360)),   # 210° = -150° in ROS

    # Step 5 — move to green box position
    ( 4.47, -7.86,  radians(210 - 360)),

    # Step 6 — intermediate point facing toward blue box
    ( 0.0,  -6.0,   radians(210 - 360)),

    # Step 7 — blue box area — robot will switch to approach mode
    # as soon as blue is detected before or at this point
    (-4.06, -9.66,  radians(270 - 360)),
]

# Contour area threshold — robot stops when blue box fills this many pixels
# (~1 metre away). Increase if stopping too early, decrease if too late.
BLUE_STOP_AREA = 20000


class ProjectNode(Node):

    def __init__(self):
        super().__init__('project_node')

        self.bridge      = CvBridge()
        self.sensitivity = 10

        self.red_found   = False
        self.green_found = False
        self.blue_found  = False

        self.blue_cx     = None
        self.blue_area   = 0
        self.image_width = None

        # State machine: exploring → approaching → done
        self.state = 'exploring'

        self.nav_client      = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.waypoint_index  = 0
        self.nav_goal_active = False

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.min_distance = float('inf')  # closest obstacle from laser
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10)

        # ── Laser scan subscriber (collision detection) ────────────────────
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.min_distance = float('inf')  # closest obstacle distance

        self.get_logger().info('ProjectNode started – facing South (270°).')

    # ── Laser scan callback ───────────────────────────────────────────────────
    def laser_callback(self, data):
        # Filter out inf/nan values and get the minimum distance
        ranges = [r for r in data.ranges if not (r == float('inf') or r != r)]
        if ranges:
            self.min_distance = min(ranges)
            if self.min_distance < 0.25:
                self.get_logger().warn(
                    f'WARNING: Obstacle very close! Distance: {round(self.min_distance, 2)} m')
            elif self.min_distance < 0.40:
                self.get_logger().warn(
                    f'CAUTION: Obstacle nearby. Distance: {round(self.min_distance, 2)} m')

    # ── Camera callback ────────────────────────────────────────────────────────
    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        self.image_width = image.shape[1]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s   = self.sensitivity

        green_mask = cv2.inRange(hsv,
                                 np.array([60-s,  100, 100]),
                                 np.array([60+s,  255, 255]))
        blue_mask  = cv2.inRange(hsv,
                                 np.array([120-s, 100, 100]),
                                 np.array([120+s, 255, 255]))
        red_mask   = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,     100, 100]), np.array([s,   255, 255])),
            cv2.inRange(hsv, np.array([180-s, 100, 100]), np.array([180, 255, 255])))

        self.red_found = self.green_found = self.blue_found = False
        self.blue_cx   = None
        self.blue_area = 0

        image = self.draw_largest_contour(image, red_mask,   (0,   0, 255), 'Red')
        image = self.draw_largest_contour(image, green_mask, (0, 255,   0), 'Green')
        image = self.draw_largest_contour(image, blue_mask,  (255, 0,   0), 'Blue')

        cv2.namedWindow('project_camera_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('project_camera_feed', image)
        cv2.resizeWindow('project_camera_feed', 640, 480)
        cv2.waitKey(1)

    # ── Draw contour + update flags ────────────────────────────────────────────
    def draw_largest_contour(self, image, mask, colour, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c    = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:
                (x, y), radius = cv2.minEnclosingCircle(c)
                cx = int(x)
                cv2.circle(image, (cx, int(y)), int(radius), colour, 2)
                cv2.putText(image, label, (cx - 20, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                if label == 'Red':
                    self.red_found = True
                    self.get_logger().info('RED detected! (no movement needed)')
                elif label == 'Green':
                    self.green_found = True
                    self.get_logger().info('GREEN detected! (no movement needed)')
                elif label == 'Blue':
                    self.blue_found = True
                    self.blue_cx    = cx
                    self.blue_area  = area
                    self.get_logger().info('BLUE detected! Approaching...')
        return image

    # ── Laser scan callback — collision warning ────────────────────────────────
    def laser_callback(self, data):
        # Filter out invalid (inf/nan) readings
        ranges = [r for r in data.ranges if not (r == float('inf') or r != r)]
        if ranges:
            self.min_distance = min(ranges)
            if self.min_distance < 0.25:
                self.get_logger().warn(
                    f'⚠️  COLLISION WARNING: obstacle at {self.min_distance:.2f} m!')
            elif self.min_distance < 0.40:
                self.get_logger().warn(
                    f'Close to obstacle: {self.min_distance:.2f} m')

    # ── Send next nav2 waypoint ────────────────────────────────────────────────
    def send_next_waypoint(self):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 server not available.')
            return

        x, y, yaw = WAYPOINTS[self.waypoint_index]
        self.waypoint_index = (self.waypoint_index + 1) % len(WAYPOINTS)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id    = 'map'
        goal_msg.pose.header.stamp       = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x    = x
        goal_msg.pose.pose.position.y    = y
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        # Human readable angle for logging
        deg = round((yaw * 180 / 3.14159) % 360, 1)
        self.get_logger().info(
            f'Waypoint {self.waypoint_index}/{len(WAYPOINTS)}: '
            f'({x}, {y}) facing {deg}°'
        )
        self.nav_goal_active = True
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.waypoint_response_callback)

    def waypoint_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Waypoint rejected – skipping.')
            self.nav_goal_active = False
            return
        goal_handle.get_result_async().add_done_callback(self.waypoint_result_callback)

    def waypoint_result_callback(self, future):
        self.get_logger().info('Waypoint reached.')
        self.nav_goal_active = False

    # ── Approach blue box — stop 1 m away ─────────────────────────────────────
    # Only called for BLUE. Red and green are detect-only.
    def approach_blue(self):
        twist = Twist()
        if self.blue_found and self.image_width is not None:
            if self.blue_area >= BLUE_STOP_AREA:
                self.get_logger().info('Stopped within 1 m of blue box!')
                self.vel_pub.publish(Twist())  # zero velocity
                self.state = 'done'
                return
            # Proportional steering toward centre of blue contour
            error           = self.blue_cx - (self.image_width / 2)
            twist.linear.x  = 0.15
            twist.angular.z = -float(error) / (self.image_width / 2) * 0.8
        else:
            # Blue temporarily out of view — spin slowly to relocate
            twist.angular.z = 0.3
        self.vel_pub.publish(twist)

    def cancel_nav_goal(self):
        self.nav_goal_active = False
        self.vel_pub.publish(Twist())  # stop immediately


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    def signal_handler(sig, frame):
        node.vel_pub.publish(Twist())
        rclpy.shutdown()

    rclpy.init(args=None)
    node = ProjectNode()
    signal.signal(signal.SIGINT, signal_handler)

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            if node.state == 'exploring':
                # Only blue triggers approach — red and green are detect only
                if node.blue_found:
                    node.get_logger().info('Blue detected! Switching to approach mode.')
                    node.cancel_nav_goal()
                    node.state = 'approaching'
                    continue
                if not node.nav_goal_active:
                    node.send_next_waypoint()

            elif node.state == 'approaching':
                node.approach_blue()

            elif node.state == 'done':
                break

    except ROSInterruptException:
        pass
    finally:
        node.vel_pub.publish(Twist())  # always stop robot on exit
        cv2.destroyAllWindows()
        rclpy.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()