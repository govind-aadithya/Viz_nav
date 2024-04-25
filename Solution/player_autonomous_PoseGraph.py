from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import PathOptimization as path_planner
import vanishing_points


class KeyboardPlayerPyGame(Player):
    folder_path = 'images'
    target_image_path = 'target.png'

    def __init__(self):
        # Default Initialize
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        
        # Map param initialize
        self.map_screen = None
        self.goal_position = None  
        self.start_point = (250, 250) 
        self.map = np.zeros((500, 500), dtype=int)  
        
        # Mode Params initialize
        self.exit_flag = False
        self.fpv_active = True
        self.interrupt_frame = 0
        self.current_phase = "exploration"
        self.enable_auto = False               # True - Auto
        self.pre_nav_status = False
        self.initiate = False
        
        # Nav Params initialize
        self.position = (0, 0)  
        self.orientation = 0  
        self.shortest_path = [(250, 250, 0), (251, 250, 0.0), (252, 250, 0.0), (253, 250, 0.0), (254, 250, 0.0), (255, 250, 0.0), (256, 250, 0.0), (257, 250, 0.0), (258, 250, 0.0), (259, 250, 0.0), (260, 250, 0.0), (261, 250, 0.0), (262, 250, 0.0), (263, 250, 0.0), (264, 250, 0.0), (265, 250, 0.0), (266, 250, 0.0), (267, 250, 0.0), (268, 250, 0.0), (269, 250, 0.0), (270, 250, 0.0), (271, 250, 0.0), (272, 250, 0.0), (273, 250, 0.0), (274, 250, 0.0), (275, 250, 0.0), (276, 250, 0.0), (277, 250, 0.0), (278, 250, 0.0), (279, 250, 0.0), (280, 250, 0.0), (281, 250, 0.0), (282, 250, 0.0), (283, 250, 0.0), (284, 250, 0.0), (285, 250, 0.0), (286, 250, 0.0), (287, 250, 0.0), (288, 250, 0.0), (289, 250, 0.0), (290, 250, 0.0), (291, 250, 0.0), (292, 250, 0.0), (293, 250, 0.0), (294, 250, 0.0), (295, 250, 0.0), (296, 250, 0.0), (297, 250, 0.0), (298, 250, 0.0), (299, 250, 0.0), (300, 250, 0.0), (301, 250, 0.0), (302, 250, 0.0), (303, 250, 0.0), (303, 251, 90.0), (303, 252, 90.0), (303, 253, 90.0), (303, 254, 90.0), (303, 255, 90.0), (303, 256, 90.0), (303, 257, 90.0), (303, 258, 90.0), (303, 259, 90.0), (303, 260, 90.0), (303, 261, 90.0), (303, 262, 90.0), (304, 262, 0.0), (305, 262, 0.0), (306, 262, 0.0), (307, 262, 0.0), (308, 262, 0.0), (309, 262, 0.0), (310, 262, 0.0), (311, 262, 0.0), (312, 262, 0.0), (313, 262, 0.0), (314, 262, 0.0), (315, 262, 0.0), (316, 262, 0.0), (317, 262, 0.0), (318, 262, 0.0), (318, 263, 90.0), (318, 264, 90.0), (318, 265, 90.0), (318, 266, 90.0), (318, 267, 90.0), (318, 268, 90.0), (318, 269, 90.0), (318, 270, 90.0), (318, 271, 90.0), (318, 272, 90.0), (318, 273, 90.0), (318, 274, 90.0), (318, 275, 90.0), (318, 276, 90.0), (317, 276, 180.0), (316, 276, 180.0), (315, 276, 180.0), (314, 276, 180.0), (313, 276, 180.0), (312, 276, 180.0), (311, 276, 180.0), (310, 276, 180.0), (309, 276, 180.0), (308, 276, 180.0), (307, 276, 180.0), (306, 276, 180.0), (305, 276, 180.0), (304, 276, 180.0), (303, 276, 180.0), (302, 276, 180.0), (301, 276, 180.0), (301, 277, 90.0), (301, 278, 90.0), (301, 279, 90.0), (301, 280, 90.0), (301, 281, 90.0), (301, 282, 90.0), (301, 283, 90.0), (301, 284, 90.0), (301, 285, 90.0), (301, 286, 90.0), (301, 287, 90.0), (301, 288, 90.0), (301, 289, 90.0), (301, 290, 90.0), (301, 291, 90.0), (301, 292, 90.0), (301, 293, 90.0), (301, 294, 90.0), (301, 295, 90.0), (301, 296, 90.0), (301, 297, 90.0), (301, 298, 90.0), (301, 299, 90.0), (301, 300, 90.0), (301, 301, 90.0), (301, 302, 90.0), (301, 303, 90.0), (301, 304, 90.0)]            
        self.path_index = 0
        self.curr_pnts = 0
        self.curr_waypoint = self.shortest_path[self.curr_pnts]
        
        # Drive correction Params
        self.short_flag = False
        self.v = vanishing_points.VanishingPointDetector()
        self.err = 0
        self.frame_count = 0
        
        # Feature params
        self.keypoint_list = []
        self.descriptor_list = []
        self.feature_pos = []
        
        super().__init__()  

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        
        self.orientation = 0
        # if self.current_phase != "pre-navigation":
        self.map.fill(0)
        self.position = (250, 250)
        self.curr_pnts = 0
        self.curr_waypoint = self.shortest_path[self.curr_pnts]
        pygame.init()
        self.enable_auto = False  # Autonomous movement is off by default
        # Add 's' key to toggle autonomous movement
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_s: 'toggle_autonomous',
            pygame.K_a: 'autonomous_left',
            pygame.K_d: 'autonomous_right',
            pygame.K_ESCAPE: Action.QUIT
        }

    
    def show_target_images(self):

        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow('KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target_img.png',concat_img)
        
        self.target_feature = feature_extraction(concat_img)
        
        cv2.waitKey(1)
        
        
    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()
        
    
    def match_finder(self, target_image_path, method='SIFT'):
        # Extract features from the target image
        image = cv2.imread(target_image_path)
        target_keypoints, target_descriptors = feature_extraction(image, method)

        # Dictionary to hold images and their match counts
        image_match_counts = {}

        # Loop over the images in the folder
        for index, _  in enumerate(self.keypoint_list):
            
            # Match the features with the target
            good_matches = feature_matching(target_keypoints, self.keypoint_list[index], target_descriptors, self.descriptor_list[index])
            
            # Store the number of good matches
            image_match_counts[index] = good_matches
        
        # Rank the images by the number of good matches
        best_match = sorted(image_match_counts.items(), key=lambda item: item[1], reverse=True)
        
        #print(best_match)
        
        return best_match
    

    def pose_update(self, action):
        
        x, y = self.position
        new_position = self.position  
        
        #print(action)

        if action == Action.FORWARD or action == 'toggle_autonomous':
            if self.orientation == 0:  # North
                new_position = (x + 1, y)
            elif self.orientation == 90:  # East
                new_position = (x, y + 1)
            elif self.orientation == 180:  # South
                new_position = (x - 1, y)
            else:  # West
                new_position = (x, y - 1)
                
        elif action == Action.BACKWARD:
            if self.orientation == 0:  # South
                new_position = (x - 1, y)
            elif self.orientation == 90:  # West
                new_position = (x, y - 1)
            elif self.orientation == 180:  # North
                new_position = (x + 1, y)
            else:  # East
                new_position = (x, y + 1)
                
        elif action == 'autonomous_left' or (self.current_phase == "navigation" and action == Action.LEFT):
            self.orientation = (self.orientation - 90) % 360
            return  # No need to check bounds or update position
        
        elif action == 'autonomous_right' or (self.current_phase == "navigation" and action == Action.RIGHT):
            self.orientation = (self.orientation + 90) % 360
            return  # No need to check bounds or update position

        # Check if the new position is within the bounds of the map and not a wall
        if 0 <= new_position[0] < self.map.shape[1] and 0 <= new_position[1] < self.map.shape[0]:
            self.position = new_position
        
        #print(new_position, self.orientation)

    def map_update(self):
        x, y = self.position
        if self.map[y, x] == 0:
            # Update visited position
            self.map[y, x] = 1  
    
    
    def draw_map(self, save_image=True):

        cell_size = 10
        window_width = self.map.shape[1] * cell_size
        window_height = self.map.shape[0] * cell_size

        # Create a new surface to draw the map on
        map_surface = pygame.Surface((window_width, window_height))

        # Define colors
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GOLD = (255, 215, 0)
        
        # Clear the surface with a white background
        map_surface.fill(WHITE)

        # Draw the map
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if (x, y) == self.start_point:
                    pygame.draw.rect(map_surface, GOLD, rect)
                    
                elif self.map[y, x] == 1:  # Visited cell
                    pygame.draw.rect(map_surface, GREEN, rect)
                
                elif self.map[y, x] == 2:  # Wall
                    pygame.draw.rect(map_surface, RED, rect)
                    
        start_x, start_y = 250, 250  # Start location coordinates
        start_rect = pygame.Rect(start_x * cell_size, start_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(map_surface, RED, start_rect)
        # Draw the robot
        robot_x, robot_y = self.position
        robot_rect = pygame.Rect(robot_x * cell_size, robot_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(map_surface, BLUE, robot_rect)

        if self.goal_position:
            goal_x, goal_y = self.goal_position
            goal_rect = pygame.Rect(goal_x * cell_size, goal_y * cell_size, cell_size, cell_size)
            BLACK = (0, 0, 0)  # Define a black color for the goal
            pygame.draw.rect(map_surface, BLACK, goal_rect)  # Draw the goal location in black

        # Optionally save the image
        if save_image:
            # Save the surface to a file
            desktop_path = "./"
            pygame.image.save(map_surface, 'map.png')

        coordinates_array = []  # Initialize an empty list to store coordinates

        
        # Iterate over the map and collect coordinates
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == 1:  # Visited cell                    
                    coordinates_array.append((x,y))
        self.shortest_path = path_planner.calculate_path(coordinates_array, self.goal_position)
        coordinates_file_path = os.path.join(desktop_path, 'map_coordinates.txt')
        with open(coordinates_file_path, 'w') as file:
            # Write all coordinates to the file
            
            file.write(f'array={coordinates_array}')

        return coordinates_array  # Return the array containing all coordinates
    
    def act(self):
        global i,turn
        action = Action.IDLE
        mapped_action = Action.IDLE


        if not self.short_flag:
            #print("Inside first condiiton")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_flag = True
                    action = Action.QUIT
                elif event.type == pygame.KEYDOWN:
                    # Check if the key is in the keymap and set the corresponding action
                    if event.key in self.keymap:
                        mapped_action = self.keymap[event.key]
                        if mapped_action == 'toggle_autonomous':
                            self.enable_auto = not self.enable_auto
                        elif event.key == pygame.K_ESCAPE:
                            # If pre-navigation hasn't been done, start it
                            if not self.pre_nav_status and self.current_phase == "exploration":
                                self.current_phase = "pre-navigation"
                                self.pre_navigation()
                                self.pre_nav_status = True  # Set the flag to True
                            else:
                                # If pre-navigation is done, proceed to quit
                                self.exit_flag = True
                                action = Action.QUIT
                                self.short_flag = True
                                self.position = (250, 250)
                                self.orientation = 0
                                self.curr_pnts = 0
                                
                        elif mapped_action == 'autonomous_left':
                                turn=1
                                i = 0
                        elif mapped_action == 'autonomous_right':
                                turn=-1
                                i = 0                
                        else:
                                action = mapped_action
            
            
            if turn==1:
                if i < 37:
                    action = Action.LEFT
                    i = i + 1
                else:
                    turn = 0
                    action = Action.IDLE
            if turn==-1:
                if i < 37:
                    action = Action.RIGHT
                    i = i + 1
                else:
                    turn = 0
                    action = Action.IDLE

            if self.pre_nav_status and not self.exit_flag:
                self.current_phase = "navigation"

            # If autonomous movement is enabled, move forward automatically
          
            if self.enable_auto:
                mapped_action = 'toggle_autonomous'
                action = Action.FORWARD
                pygame.time.delay(70)  # Tune this value for mapping drive speed
                
            # Additional code for current_phase transition
            if self.current_phase == "pre-navigation":
                self.pre_navigation()
                self.pre_nav_status = True
                
            self.pose_update(mapped_action)
            self.map_update()

        
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_flag = True
                    action = Action.QUIT
                
            if self.orientation == self.curr_waypoint[2] and turn == 0 : # this check is the oientaion criteria is met
                # Check if the key is in the keymap and set the corresponding action
                if self.position == (self.curr_waypoint[0],self.curr_waypoint[1]):
                    self.enable_auto = False # the waypoint reached
                    self.curr_pnts += 2
                    if self.curr_pnts > len(self.shortest_path)-3:
                        # If pre-navigation hasn't been done, start it
                        print("Goal Reached!")
                        if not self.pre_nav_status and self.current_phase == "exploration":
                            self.current_phase = "pre-navigation"
                            self.pre_navigation()
                            self.pre_nav_status = True  # Set the flag to True
                        else:
                            # If pre-navigation is done, proceed to quit
                            self.exit_flag = True
                            self.short_flag = False
                            #action = Action.QUIT
                    self.curr_waypoint = self.shortest_path[self.curr_pnts]
                else:
                    #print("moving forward")
                    self.enable_auto = True  # the waypoint is yet to reached
                    
            elif i >= 37 or i == 0:  
                if self.orientation - self.curr_waypoint[2] > 0:
                            turn = +1
                            i = 0
                elif self.orientation - self.curr_waypoint[2] < 0:
                            turn = -1
                            i = 0                
            update = True
            
            if turn==1:
                if i < 37:
                    action = Action.LEFT
                    i = i + 1
                else:
                    turn = 0
                    action = Action.IDLE
            elif turn==-1:
                if i < 37:
                    action = Action.RIGHT
                    i = i + 1
                else:
                    turn = 0
                    action = Action.IDLE
            else:
                #self.err = 0

                if self.err > 5 and self.err < 15:
                    action = Action.RIGHT
                    #print("Turning right for correction")
                    self.err = 0
                    update = False

                elif self.err < -5 and self.err > -15:
                    action = Action.LEFT
                    #print("Turning left for correction")
                    self.err = 0
                    update = False


            if self.pre_nav_status and not self.exit_flag:
                self.current_phase = "navigation"

            # If autonomous movement is enabled, move forward automatically
            if self.enable_auto and update:
                action = Action.FORWARD
                
            # Additional code for current_phase transition
            if self.current_phase == "pre-navigation":
                self.pre_navigation()

            if update and ( i <= 0 or i >= 37):
                self.pose_update(action)
                self.map_update()     
        
        
        return action
    
    
    def draw_and_wait(self):
        self.draw_map(True)
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False


    def auto_advance(self, target_position):
        # Move towards the target position
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]

        if dx > 0:
            return Action.RIGHT
        elif dx < 0:
            return Action.LEFT
        elif dy > 0:
            return Action.FORWARD
        elif dy < 0:
            return Action.BACKWARD
        else:
            return Action.IDLE

    def see(self, fpv):
        if not self.fpv_active or fpv is None or len(fpv.shape) < 3:
            return 
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        rgb = self.convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

        if self.frame_count != 0:
            self.err = self.v.detect_vanishing_point(fpv)
        else:
            self.frame_count += 1
        # if self.initiate:
        if self.get_state():
            _, current_phase, _, _, _, _ = self.get_state()

        self.interrupt_frame  = (self.interrupt_frame + 1) % 2

        # Save the FPV image according to the mapping
        if self.interrupt_frame  % 2 == 0:
            self.build_feature_map(fpv)
        self.initiate = True

    def build_feature_map(self, fpv_image):
        
        kp, des = feature_extraction(fpv_image)
        self.keypoint_list.append(kp)
        self.descriptor_list.append(des)
        self.feature_pos.append(self.position)
        
        
    def convert_opencv_img_to_pygame(self, opencv_image):
        """
        Convert OpenCV images for Pygame.
        """
        opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
        shape = opencv_image.shape[1::-1]  # (height, width)
        pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
        return pygame_image
    
    def pre_navigation(self):
        if self.current_phase == "pre-navigation":
            print('Pre-navigation current_phase started.')
            
            
            best_match = self.match_finder(self.target_image_path, 'SIFT')

            if best_match:
                best_image_index, _ = best_match[0]
                goal_location = self.feature_pos[best_image_index]

                if goal_location:
                    self.goal_position = goal_location
                    print(f"The goal location is: {self.goal_position}")
                    self.draw_map() 
                else:
                    print("No coordinates could be extracted from the image name.")
            else:
                print("No ranked images available to extract the goal location.")

            self.current_phase = "navigation"


def feature_extraction(image, method='SIFT'):
       
    if image is None:
        raise FileNotFoundError("Image could not be read. Ensure the path is correct and the file is accessible.")
    # Convert it to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect features and compute descriptors.
    if method == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    elif method == 'SURF':
        # Note: Your OpenCV needs to be compiled with nonfree modules to use SURF
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(gray_image, None)
    
    return keypoints, descriptors


def feature_matching(kp1, kp2, descriptors1, descriptors2):
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Keep good matches: ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    similarity = len(good_matches) / len(kp1)
    
    #print(len(good_matches))
    
    return similarity


if __name__ == "__main__":
    import vis_nav_game
    #import shutil
    
    i = 0
    turn=0
    
    '''
    images_folder_path = 'images'
    
    # Ensure the 'images' folder exists
    if os.path.exists(images_folder_path):    
       shutil.rmtree(images_folder_path) 
       print('Image folder deleted and ready to go')
    '''   
       
    player = KeyboardPlayerPyGame()
    vis_nav_game.play(the_player=player)