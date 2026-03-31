#from __future__ import division
import cv2
import threading
import numpy as np
import rclpy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')

        # Create bridge for ROS image -> OpenCV image conversion
        self.bridge = CvBridge()

        # Sensitivity for colour detection range
        self.sensitivity = 10

        # Subscribe to the camera image topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )

        self.subscription  # prevent unused variable warning

    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # Convert BGR image to HSV
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define green colour range in HSV
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

            # Create mask for green colour
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)

            # Apply mask to original image
            filtered_img = cv2.bitwise_and(cv_image, cv_image, mask=green_mask)

            # Show filtered image
            cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('camera_Feed', filtered_img)
            cv2.resizeWindow('camera_Feed', 320, 240)
            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


def main():

    def signal_handler(sig, frame):
        rclpy.shutdown()

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

    cv2.destroyAllWindows()
    cI.destroy_node()


if __name__ == '__main__':
    main()