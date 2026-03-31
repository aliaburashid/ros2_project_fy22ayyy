# Exercise 4 - following a colour (green) and stopping upon sight of another (blue).

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        # Publisher to move the robot
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Flags
        self.green_found = False
        self.blue_found = False
        self.move_forward_flag = False
        self.move_backward_flag = False

        # Sensitivity for HSV colour detection
        self.sensitivity = 10

        # Create bridge and subscribe to camera feed
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def callback(self, data):
        try:
            # Convert ROS image to OpenCV image
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # Show original camera feed
            cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('camera_Feed', image)
            cv2.resizeWindow('camera_Feed', 320, 240)

            # Convert BGR image to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Green HSV range
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

            # Blue HSV range
            hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
            hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

            # Create masks
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

            # Apply green mask for display
            filtered_image = cv2.bitwise_and(image, image, mask=green_mask)

            # Find contours for green and blue
            green_contours, _ = cv2.findContours(
                green_mask,
                mode=cv2.RETR_LIST,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            blue_contours, _ = cv2.findContours(
                blue_mask,
                mode=cv2.RETR_LIST,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            # Reset flags
            self.green_found = False
            self.blue_found = False
            self.move_forward_flag = False
            self.move_backward_flag = False

            # Check if blue is detected -> stop condition
            if len(blue_contours) > 0:
                blue_c = max(blue_contours, key=cv2.contourArea)
                if cv2.contourArea(blue_c) > 100:
                    self.blue_found = True

            # Check if green is detected -> follow condition
            if len(green_contours) > 0:
                green_c = max(green_contours, key=cv2.contourArea)
                green_area = cv2.contourArea(green_c)

                if green_area > 100:
                    self.green_found = True

                    # Draw circle around green object
                    (x, y), radius = cv2.minEnclosingCircle(green_c)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(filtered_image, center, radius, (0, 255, 255), 2)

                    # Decide movement based on object size
                    if green_area > 30000:
                        self.move_backward_flag = True
                    else:
                        self.move_forward_flag = True

            # If blue is found, override everything and stop
            if self.blue_found:
                self.move_forward_flag = False
                self.move_backward_flag = False
                print("Blue detected - stopping")

            elif self.green_found:
                print("Green detected")

            # Show filtered image
            cv2.namedWindow('threshold_Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('threshold_Feed', filtered_image)
            cv2.resizeWindow('threshold_Feed', 320, 240)

            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def walk_forward(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.2
        desired_velocity.angular.z = 0.0
        self.publisher.publish(desired_velocity)

    def walk_backward(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = -0.2
        desired_velocity.angular.z = 0.0
        self.publisher.publish(desired_velocity)

    def stop(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.0
        desired_velocity.angular.z = 0.0
        self.publisher.publish(desired_velocity)


# Create a node of your class in the main and ensure it stays up and running
def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    robot = Robot()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            # If blue is seen -> stop
            if robot.blue_found:
                robot.stop()

            # If green is seen and too close -> move backward
            elif robot.move_backward_flag:
                robot.walk_backward()

            # If green is seen and not too close -> move forward
            elif robot.move_forward_flag:
                robot.walk_forward()

            # Otherwise stop
            else:
                robot.stop()

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()