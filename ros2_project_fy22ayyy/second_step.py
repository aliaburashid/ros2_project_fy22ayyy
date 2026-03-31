# Exercise 2 - detecting two colours, and filtering out the third colour and background.

import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')

        # Sensitivity for colour range
        self.sensitivity = 10

        # Create bridge and subscribe to camera topic
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

            # Convert the BGR image into HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define HSV range for green
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

            # Define HSV range for blue
            hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
            hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

            # Create a mask for each colour
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

            # Combine the two masks
            combined_mask = cv2.bitwise_or(green_mask, blue_mask)

            # Apply the combined mask to the original image
            filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

            # Show the filtered result
            cv2.namedWindow('filtered_Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('filtered_Feed', filtered_image)
            cv2.resizeWindow('filtered_Feed', 320, 240)

            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    # Initialise ROS and create node
    rclpy.init(args=None)
    cI = colourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Destroy OpenCV windows before closing
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main()