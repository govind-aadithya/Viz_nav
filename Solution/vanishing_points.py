import cv2
import math
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Suppress all warnings of type RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

class VanishingPointDetector:
    REJECT_DEGREE_TH = 4.0

    def __init__(self):
        self.image = None
        self.lines = None

    def _filter_lines(self, lines):
        final_lines = []

        for line in lines:
            [[x1, y1, x2, y2]] = line

            # Calculating the equation of the line: y = mx + c
            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
            else:
                m = 100000000
            c = y2 - m * x2
            # theta will contain values between -90 -> +90.
            theta = math.degrees(math.atan(m))

            # Rejecting lines of slope near to 0 degree or 90 degrees and storing others
            if self.REJECT_DEGREE_TH <= abs(theta) <= (90 - self.REJECT_DEGREE_TH):
                # Calculate the length of the line with handling the case where points are the same
                l = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) if (x1, y1) != (x2, y2) else 0
                final_lines.append([x1, y1, x2, y2, m, c, l])

        # Removing extra lines
        # (we might get many lines, so we are going to take only the longest 15 lines
        # for further computation because more than this number of lines will only
        # contribute towards slowing down of our algorithm.)
        if len(final_lines) > 15:
            final_lines = sorted(final_lines, key=lambda x: x[-1], reverse=True)
            final_lines = final_lines[:15]

        return final_lines

    def _get_lines(self):
        # Converting to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Blurring image to reduce noise.
        blur_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
        # Generating Edge image
        edge_image = cv2.Canny(blur_gray_image, 40, 255)

        # Finding Lines in the image
        lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, 50, 10, 15)

        # Check if lines found
        if lines is None or len(lines) == 0:
            #print("Not enough lines found in the image for Vanishing Point detection.")
            return None

        # Filtering Lines wrt angle
        self.lines = self._filter_lines(lines)

    def _get_vanishing_point(self):
        # We will apply an algorithm inspired by RANSAC for this. We will take a combination
        # of 2 lines one by one, find their intersection point, and calculate the
        # total error (loss) of that point. Error of the point means the root of the sum of
        # squares of the distance of that point from each line.
        vanishing_point = None
        min_error = 100000000000

        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                m1, c1 = self.lines[i][4], self.lines[i][5]
                m2, c2 = self.lines[j][4], self.lines[j][5]

                if m1 != m2:
                    x0 = (c1 - c2) / (m2 - m1)
                    y0 = m1 * x0 + c1

                    err = 0
                    for k in range(len(self.lines)):
                        m, c = self.lines[k][4], self.lines[k][5]
                        m_ = (-1 / m)
                        c_ = y0 - m_ * x0

                        x_ = (c - c_) / (m_ - m)
                        y_ = m_ * x_ + c_

                        l = math.sqrt((y_ - y0) ** 2 + (x_ - x0) ** 2)

                        err += l ** 2

                    err = math.sqrt(err)

                    if min_error > err:
                        min_error = err
                        vanishing_point = [x0, y0]

        return vanishing_point

    def detect_vanishing_point(self, image):
        # Get vanishing point
        self.image = image
        self._get_lines()
        
        # Check if lines are found
        if self.lines is None:
            return

        vanishing_point = self._get_vanishing_point()

        # Checking if the vanishing point is found
        if vanishing_point is None:
            #print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")
            return 10000
        else:
            # Calculate the actual midpoint of the image
            mid_pt = [self.image.shape[1] // 2, self.image.shape[0] // 2]

            # Draw lines and vanishing point
            #for line in self.lines:
                #cv2.line(self.image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            #cv2.circle(self.image, (int(vanishing_point[0]), int(vanishing_point[1])), 10, (0, 0, 255), -1)

            # Draw a circle at the midpoint with a different color
            #cv2.circle(self.image, (mid_pt[0], mid_pt[1]), 5, (255, 0, 255), -1)

            # Calculate and print the distance between the vanishing point and midpoint
            distance = math.sqrt((vanishing_point[0] - mid_pt[0]) ** 2 + (vanishing_point[1] - mid_pt[1]) ** 2)
            distance = vanishing_point[0] - mid_pt[0]
            #print(f"Distance between Vanishing Point and Midpoint: {distance}")
            return distance
            # Show the final image
            #cv2.imshow("OutputImage", self.image)
            #cv2.waitKey(0)


if __name__ == "__main__":
    input_image_path = "images/fpv_x250_y250.png"
    image = cv2.imread(input_image_path)
    vpd = VanishingPointDetector()
    vpd.detect_vanishing_point(image)
