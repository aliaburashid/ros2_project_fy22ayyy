# Exercise 3 - If green object is detected, and above a certain size, then send a message (print or use lab2)

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

        # Flag to show whether green has been detected
        self.green_found = False

        # Sensitivity for colour detection
        self.sensitivity = 10

        # Initialise CvBridge and subscribe to camera topic
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
            # Convert the received ROS image into an OpenCV image
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # Show original camera feed
            cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('camera_Feed', image)
            cv2.resizeWindow('camera_Feed', 320, 240)

            # Set HSV range for green
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

            # Convert BGR image to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create green mask
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)

            # Apply the mask to the original image
            filtered_image = cv2.bitwise_and(image, image, mask=green_mask)

            # Find contours in the green mask
            contours, _ = cv2.findContours(
                green_mask,
                mode=cv2.RETR_LIST,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            # Reset flag before checking
            self.green_found = False

            if len(contours) > 0:
                # Find largest contour
                c = max(contours, key=cv2.contourArea)

                # Only consider it if the contour area is large enough
                if cv2.contourArea(c) > 100:
                    # Find centre and radius of enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    radius = int(radius)

                    # Draw circle around detected green object
                    cv2.circle(filtered_image, center, radius, (0, 255, 255), 2)

                    # Set flag
                    self.green_found = True

            # Print message if green was found
            if self.green_found:
                print("Green detected")

            # Show filtered result
            cv2.namedWindow('threshold_Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('threshold_Feed', filtered_image)
            cv2.resizeWindow('threshold_Feed', 320, 240)

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

    # Destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main()