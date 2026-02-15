import csv
import cv2
import time
import mediapipe as mp
import numpy as np
import random
import tkinter as tk
from tkinter import ttk, messagebox, Scale, Label, Toplevel
from PIL import Image, ImageTk, ImageDraw, ImageFont
import speech_recognition as sr
import pygame
import pyaudio
import threading
import os
from screeninfo import get_monitors
from concurrent.futures import ThreadPoolExecutor
# from picamera.array import PiRGBArray
# from picamera import PiCamera

# Global variables for MediaPipe resources to avoid reinitializing them multiple times
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose

pygame.mixer.init()
score = 0
joint_to_monitor = ''
is_flexing = False  # Para detectar a mudança de estado
csv_filename = './game_scores.csv'
current_level = 1
level_target_scores = {1: 10, 2: 10, 3: 10, 4: 300}  # Targets for each level
level_complete = False
level_complete_message = ""
lives = 3  # Start with 3 lives
game_over = False
game_over_time = 2
global ball_position, ball_velocity, obstacle_position, target_positions, goal_positions
ball_position = [640, 0]  # Initial ball position at the top center of the screen
ball_velocity = [0, 5]  # Initial ball velocity
obstacle_position = [random.randint(100, 1820), random.randint(100, 980)]  # Initial obstacle position
target_positions = []  # List of target positions for level 4
goal_positions = []  # List of goal positions for levels
game_over_time_right = None
# Terminar jogo
button_area = (1500, 50, 1800, 200)  # (x1, y1, x2, y2)
# Global flag to control speech recognition
speech_recognition_enabled = True
first_it = True
level_4_duration = 45  # Duration of Level 4 in seconds
level_4_start_time = None  # To track the start time of Level 4
successful_catches = 0
level_3_obstacle_positions = []
global speech
pygame.mixer.init()
# Initialize global variables
score_positions = {}
# # Load the pre-trained MobileNet SSD model and the prototxt file
net = cv2.dnn.readNetFromCaffe('./deploy.prototxt', './mobilenet_iter_73000.caffemodel')
root = None

monitor = get_monitors()[0] 
screen_width = monitor.width
screen_height = monitor.height
print(f"width = {screen_width}")
print(f"height = {screen_height}")

# Carrega as imagens para cotovelo
elbowL_images = [
    cv2.imread('./corpo/cot1L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/cot2L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/cot3L.png', cv2.IMREAD_UNCHANGED)
]

# Carrega as imagens para cotovelo
elbowR_images = [
    cv2.imread('./corpo/cot3R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/cot1R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/cot2R.png', cv2.IMREAD_UNCHANGED)
]

# Carrega as imagens para neck
neck_images = [
    cv2.imread('./corpo/neckreal1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/neckreal2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/neckreal4.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/neckreal3.png', cv2.IMREAD_UNCHANGED)
]

# Carrega as imagens para neck
wristL_images = [
    cv2.imread('./corpo/wrist1L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/wrist2L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/wrist3L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/wrist4L.png', cv2.IMREAD_UNCHANGED)
]

# Carrega as imagens para neck
wristR_images = [
    cv2.imread('./corpo/wrist1R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/wrist2R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/wrist3R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/wrist4R.png', cv2.IMREAD_UNCHANGED)
]

# Carrega as imagens para joelho
kneeL_images = [
    cv2.imread('./corpo/knee1L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/knee2L.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/knee3L.png', cv2.IMREAD_UNCHANGED)
]

# Carrega as imagens para joelho
kneeR_images = [
    cv2.imread('./corpo/knee1R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/knee2R.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('./corpo/knee3R.png', cv2.IMREAD_UNCHANGED)
]


def show_instructions(image, level):
    instructions = {
        1: "Level 1: Touch the moving goals with your hands to score points!",
        2: "Level 2: Catch the falling ball with your hands. Avoid letting it hit the bottom!",
        3: "Level 3: Avoid the obstacles while catching the ball!",
        4: "Level 4: Touch as many targets as possible within the time limit!"
    }
    instruction_text = instructions.get(level, "Good luck!")
    text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, instruction_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def complete_level(image, cap, sound_on=True):
    global current_level, level_complete, level_complete_message, sound, game_over
    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))
    level_complete = True

    if current_level >= len(level_target_scores):
        level_complete_message = "Congratulations! You've completed all levels!"
        current_level = 1  # Reset to start over
        current_time_plus = time.time() + 2
        if time.time() > current_time_plus: 
            text_size = cv2.getTextSize(level_complete_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (image.shape[0] + text_size[1]) // 2
            cv2.putText(image, level_complete_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Single Player', image)
            cv2.waitKey(1)

        if sound:
            sound.stop()  # Stop the music
        exit_game(cap, sound_on)
    else:
        level_complete_message = f"Level {current_level} complete! Prepare for the next level..."
        current_level += 1
        print(f"Advancing to level {current_level}")  # Debugging statement
        setup_level(current_level, screen_width, screen_height)
        game_over = False  # Ensure game_over is reset

    print(level_complete_message)  # Debugging statement

def level_1(image, results, goal_positions, width, height, cap, sound_on=True):
    global score, game_over_time_right, game_over

    if game_over_time_right is None:
        game_over_time_right = 0

    if not game_over and score < level_target_scores[current_level]:
        draw_and_check_goals(image, results.multi_hand_landmarks, goal_positions, width, height)
            
        # Check for the circle end condition
        is_sound_on = draw_circle_and_check_end(cap, height, image, results, is_right_half=False, is_single_player=True, sound_on=True, fisio = False)

    
        # Drawing the power-off button
        draw_power_off(image, width)

    else:
        cv2.putText(image, "Game Over! Restarting the level...", (width // 2 - 300, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        if time.time() > (game_over_time_right + 2):
            reset_game_state(width, height)
            game_over = False
            game_over_time_right = None
    
    cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return is_sound_on

def level_2(image, results, goal_positions, width, height, cap, sound_on = True):
    global ball_position, ball_velocity, score, lives, game_over, game_over_time_right
    
    # Initialize game_over_time_right if it's None
    if game_over_time_right is None:
        game_over_time_right = 0
    
    if not game_over:
        
        ball_position[1] += ball_velocity[1]
        draw_lives(image, lives, width)
        sound_on = draw_circle_and_check_end(cap, height, image, results, is_right_half=False, is_single_player=True, sound_on=True, fisio = False)

        draw_power_off(image, width)
        
        # Check if there are any detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Check for catches with each hand
                if detect_catch(ball_position, hand_landmarks, width, height):
                    score += 1  # Increment the score
                    reset_ball_position(ball_position, width)  # Reset the ball position
                    ball_velocity[1] = min(ball_velocity[1] + 1, 100)  # Increase ball speed

        cv2.circle(image, tuple(ball_position), 30, (0, 255, 255), -1)  # Draw the ball

    # Handle game over logic and ball position resets
    if ball_position[1] >= height:
        lives -= 1
        draw_lives(image, lives, width)
        if lives == 0:
            game_over_time_right = time.time()
            game_over = True
        else:
            reset_ball_position(ball_position, width)
    
    if not game_over:
        cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # So passa para o proximo nivel quando passar este!
    else:
        cv2.putText(image, "Game Over! Restarting the level...", (width // 2 - 300, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        if time.time() > (game_over_time_right + 2):
            print("Restarting level...")
            reset_game_state(width, height)
            game_over = False
            game_over_time_right = None
            sound_on = True
            
    return sound_on

def level_3(image, results, goal_positions, width, height, cap, sound_on=True):
    global ball_position, ball_velocity, level_3_obstacle_positions, score, lives, game_over, game_over_time_right, successful_catches, current_level

    if game_over_time_right is None:
        game_over_time_right = 0

    if not game_over:
        ball_position[1] += ball_velocity[1]
        draw_lives(image, lives, width)
        is_sound_on = draw_circle_and_check_end(cap, height, image, results, is_right_half=False, is_single_player=True, sound_on=sound_on, fisio = False)
        draw_power_off(image, width)
        draw_obstacles(image, level_3_obstacle_positions)  # Draw all obstacles

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if detect_catch(ball_position, hand_landmarks, width, height):
                    score += 1
                    successful_catches += 1  # Increment the successful catches counter
                    reset_ball_position(ball_position, width)
                    ball_velocity[1] = min(ball_velocity[1] + 1, 100)
                    if successful_catches % 3 == 0:  # Add a new obstacle every 3 successful catches
                        add_new_obstacle(level_3_obstacle_positions, width, height)

                if detect_collision(hand_landmarks, level_3_obstacle_positions, width, height):
                    lives -= 1
                    if lives == 0:
                        game_over_time_right = time.time()
                        game_over = True
                    else:
                        reset_obstacle_position(level_3_obstacle_positions, width, height)

        cv2.circle(image, tuple(ball_position), 30, (0, 255, 255), -1)

        if ball_position[1] >= height:
            lives -= 1
            if lives == 0:
                game_over_time_right = time.time()
                game_over = True
            else:
                reset_ball_position(ball_position, width)

    if not game_over:
        cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "Game Over! Restarting the level...", (width // 2 - 300, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        is_sound_on = True
        if time.time() > (game_over_time_right + 2):
            print("Restarting level...")
            reset_game_state(width, height)
            game_over = False
            game_over_time_right = None
        elif score >= level_target_scores[current_level]:
            complete_level()

    return is_sound_on
    
def level_4(image, results, goal_positions, width, height, cap, sound_on=True):
    global target_positions, score, level_4_start_time, game_over, is_sound_on
    
    if len(target_positions) == 0:
        target_positions = init_target_positions(width, height)
    
    if level_4_start_time is None:
        level_4_start_time = time.time()
    
    current_time = time.time()
    elapsed_time = current_time - level_4_start_time
    remaining_time = max(0, level_4_duration - elapsed_time)  # Ensure time does not go negative
    
    print(f"Remaining time: {remaining_time}")  # Debugging statement
    cv2.putText(image, f"Time: {int(remaining_time)}s", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    if remaining_time > 0:
        draw_targets(image, target_positions)
        # Ensure is_sound_on is always returned
        is_sound_on = draw_circle_and_check_end(cap, height, image, results, is_right_half=False, is_single_player=True, sound_on=sound_on, fisio = False)
        draw_power_off(image, width)
        print("Level 4 running.")
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for target in target_positions[:]:  # Iterate over a copy of the list
                    if detect_touch(hand_landmarks, target, width, height):
                        score += 1
                        target_positions.remove(target)
                
                    if not target_positions:  # If all targets are touched, generate new ones
                        target_positions = init_target_positions(width, height)
        
        cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    if remaining_time <= 0:
        complete_level(image, cap, sound_on=True)
        game_over = True
        is_sound_on = exit_game(cap, sound_on)
        print("Level 4 completed.")  # Debugging statement
        #leaves the while loop

    return is_sound_on

# Dictionary to map level number to function
level_functions = {
    1: level_1,
    2: level_2,
    3: level_3,
    4: level_4,
}

def reset_game_state(width, height):
    global ball_position, ball_velocity, score, lives, game_over
    # Reset all game parameters
    ball_position = [640, 0]
    ball_velocity = [0, 5]
    score = 0
    lives = 3
   
def draw_lives(image, lives, width):
    # Load the heart icon and ensure it has the same number of channels as the main image
    heart_icon = cv2.imread('./vidas.png', cv2.IMREAD_COLOR)  # Assuming your icon is in color
    if heart_icon is None:
        raise FileNotFoundError("The heart icon image file could not be loaded.")

    # Ensure heart icon is resized to a size appropriate for display
    heart_icon = cv2.resize(heart_icon, (40, 40))  # Adjust dimensions as needed

    # Define the positions for the lives (hearts) starting from the left upper corner, a bit lower down
    heart_width = heart_icon.shape[1]
    start_x = 10  # Start 10 pixels from the left edge of the image
    start_y = 50  # Start 50 pixels from the top edge of the image, instead of 10
    heart_positions = [(start_x + i * (heart_width + 10), start_y) for i in range(lives)]  # Adjust spacing as needed
    
    # Draw the current number of lives
    for pos in heart_positions:
        x, y = pos
        # Make sure we're not going out of the image bounds
        if y + heart_icon.shape[0] <= image.shape[0] and x + heart_icon.shape[1] <= image.shape[1]:
            image[y:y+heart_icon.shape[0], x:x+heart_icon.shape[1]] = heart_icon

def detect_catch(ball_position, hand_landmarks, width, height, ball_radius=20):
    # Check if any hand landmark is close enough to the ball to be considered a catch
    if hand_landmarks:
        for landmark in hand_landmarks.landmark:
            # Transform the landmark position to match the image coordinates
            landmark_x = int(landmark.x * width)
            landmark_y = int(landmark.y * height)
            # Calculate the distance from the landmark to the ball
            distance = np.sqrt((landmark_x - ball_position[0]) ** 2 + (landmark_y - ball_position[1]) ** 2)
            if distance < ball_radius:  # If close enough, it's a catch
                return True
    return False

def reset_ball_position(ball_position, width):
    ball_position[0] = random.randint(0, width)  # Random x-coordinate at the top
    ball_position[1] = 0  # Start from the top

def init_goal_positions(width, height):
    return [(random.randint(50, width - 50), random.randint(50, height - 50)), (random.randint(50, width - 50), random.randint(50, height - 50))]

def init_goal_positions_multiplayer(screen_width, screen_height):
    player_goals = {
        'left': [(random.randint(50, screen_width // 2 - 50), random.randint(50, screen_height - 50)) for _ in range(2)],
        'right': [(random.randint(screen_width // 2 + 50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(2)]
    }
    return player_goals

def init_target_positions(width, height):
    return [(random.randint(50, width - 50), random.randint(50, height - 50)) for _ in range(10)]

def draw_power_off(image, width):
    power_off_icon = cv2.imread('./poweroff.png', cv2.IMREAD_COLOR)  # Assuming your icon is in color
    if power_off_icon is None:
        raise FileNotFoundError("The power off icon image file could not be loaded.")

    power_off_icon = cv2.resize(power_off_icon, (30, 30))  # Adjust dimensions as needed
    start_x = width - power_off_icon.shape[1] - 10  # Start 10 pixels from the right edge of the image

    # Ensure we're not going out of the image bounds
    if start_x >= 0:
        # Draw the power off icon on the image
        image[10:10+power_off_icon.shape[0], start_x:start_x+power_off_icon.shape[1]] = power_off_icon
        
def draw_circle_and_check_end(cap, height, image, results, is_right_half=False, is_single_player=False, sound_on = True, fisio = False):
    
    if fisio == False:
        radius = 18
        circle_color = (0, 0, 0)  # Black color in BGR
    else:
        radius = 18
        circle_color = (0, 0, 0)
        
    if not is_single_player and fisio == False:
        if is_right_half == True:
             x_position = image.shape[1] - radius - 8
        else:
            x_position = image.shape[1] - radius - 8
    
    if is_single_player == True:
        x_position = image.shape[1] - radius - 8
    
    if fisio == True:
        x_position = (image.shape[1] + 3)//2
    
    if fisio == False:
        y_position = 6 + radius  # Fixed y position, adjusted from top
    else:
        y_position = int(0.9 * image.shape[0] - 30)
    
    x_position = int(x_position)
    y_position = int(y_position)
    
    if fisio == False:
        cv2.circle(image, (x_position, y_position), radius, circle_color, -1)
        # Verifique se há landmarks de mãos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Para cada landmark, calcule as coordenadas (x, y) na imagem
                for landmark in hand_landmarks.landmark:
                    lx = int(landmark.x * image.shape[1])
                    ly = int(landmark.y * image.shape[0])
                    # Calcule a distância entre o landmark e o centro do círculo
                    distance = np.linalg.norm(np.array([lx - x_position, ly - y_position]))
                    # Se a distância for menor que o raio do círculo, saia do jogo
                    if distance < (radius - 4):
                        sound_on = exit_game(cap, sound_on)
                        break
    else:
        
        # Drawing a rectangle
        # top_left = (int(x_position - radius), int(y_position - radius))
        # bottom_right = (int(x_position + radius), int(y_position + radius))
        # cv2.rectangle(image, top_left, bottom_right, rectangle_color, -1)
        cv2.circle(image, (x_position, y_position), radius, circle_color, -1)
        # The parameter results is passed as: results.pose_landmarks.landmark
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                # For each landmark, calculate the coordinates (x, y) in the image
                lx = int(landmark.x * image.shape[1])
                ly = int(landmark.y * image.shape[0])
                # Check if the landmark is within the rectangleº
                distance = np.linalg.norm(np.array([lx - x_position, ly - y_position]))
                if distance < (radius - 4):
                    sound_on = exit_game(cap, sound_on)
                    break

    return sound_on

def draw_end_game(image, width):
    end_button = cv2.imread('./DISLIKE_teste.png', cv2.IMREAD_COLOR)
    restart_button = cv2.imread('./LIKE_teste.png', cv2.IMREAD_COLOR)
    if end_button is None or restart_button is None:
        raise FileNotFoundError("The icon image file could not be loaded.")

    icon_height, icon_width = 48, 48
    end_button = cv2.resize(end_button, (icon_width, icon_height))
    restart_button = cv2.resize(restart_button, (icon_width, icon_height))

    # Define the x positions for the buttons to be centered above the circles
    circle_radius = 30
    circle_padding = 20
    start_x1 = image.shape[1] // 2 + circle_radius - icon_width // 2 + circle_padding // 2
    start_x2 = image.shape[1] // 2 - circle_radius - icon_width // 2 - circle_padding // 2
    
    # Define the y_position to be above the circles
    text_y_position = image.shape[0] // 2 + 50 + 50  # Adjust based on your text position
    circle_y_position = text_y_position + 50 + circle_radius + 20 # Add some padding below the text
    y_position = circle_y_position - 25  # Adjust to position above the circles

    if start_x1 >= 0 and start_x1 + icon_width <= image.shape[1]:
        image[y_position:y_position+icon_height, start_x1:start_x1+icon_width] = end_button
    if start_x2 >= 0 and start_x2 + icon_width <= image.shape[1]:
        image[y_position:y_position+icon_height, start_x2:start_x2+icon_width] = restart_button

def draw_circles_v2(cap, image, results, sound_on=True, like = False, dislike = False):
    radius = 30
    circle_color = (255, 255, 255)  # White color in BGR
    
    # Define the x positions for the circles to be centered
    circle_padding = 20
    x_position_1 = image.shape[1] // 2 - radius - circle_padding // 2
    x_position_2 = image.shape[1] // 2 + radius + circle_padding // 2

    text_y_position = image.shape[0] // 2 + 50 + 50  # Adjust based on your text position
    y_position = text_y_position + 50 + radius + 18  # Add some padding below the text and buttons

    cv2.circle(image, (x_position_1, y_position), radius, circle_color, -1)
    cv2.circle(image, (x_position_2, y_position), radius, circle_color, -1)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lx = int(landmark.x * image.shape[1])
                ly = int(landmark.y * image.shape[0])
                distance1 = np.linalg.norm(np.array([lx - x_position_1, ly - y_position]))
                distance2 = np.linalg.norm(np.array([lx - x_position_2, ly - y_position]))
                if distance2 < (radius - 4):
                    sound_on = exit_game(cap, sound_on)
                    dislike = True
                    break
                elif distance1 < (radius - 4):
                    sound_on = False
                    like = True
                    break
                
    return like, dislike

# Draw goals and check if they are hit
def draw_and_check_goals(image, landmarks, goal_positions, width, height):
    global score
    goal_radius = 20
    colors = [(0, 255, 0), (0, 0, 255)]


    for pos, color in zip(goal_positions, colors):
        cv2.circle(image, pos, goal_radius, color, -1)
    
    if landmarks:
        for hand_landmark in landmarks:
            for landmark in hand_landmark.landmark:
                lx, ly = int(landmark.x * width), int(landmark.y * height)
                for index, (gx, gy) in enumerate(goal_positions):
                    if np.linalg.norm(np.array([lx - gx, ly - gy])) < goal_radius:
                        score += 1
                        goal_positions[index] = (
                            random.randint(goal_radius, width - goal_radius),
                            random.randint(goal_radius, height - goal_radius)
                        )
                        break


def process_and_draw(image, results, player_side, screen_width, screen_height, scores, player_index, ball_positions, ball_radius, illuminated_ball, illuminated_start_time):
    global score_positions

    if results.multi_hand_landmarks:
        hand_landmarks_list = list(results.multi_hand_landmarks)
        hand_landmark_coordinates = [
            (int(landmark.x * screen_width), int(landmark.y * screen_height))
            for hand_landmarks in hand_landmarks_list
            for landmark in hand_landmarks.landmark
        ]

        for (lx, ly) in hand_landmark_coordinates:
            if illuminated_ball is not None:
                gx, gy = ball_positions[illuminated_ball]
                if np.linalg.norm([lx - gx, ly - gy]) < ball_radius:
                    scores[player_index] += 1
                    illuminated_ball = None
                    illuminated_start_time = None
                    score_positions[player_side][(gx, gy)] = time.time()
                    break

        for hand_landmarks in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    for i, (gx, gy) in enumerate(ball_positions):
        color = (0, 255, 0) if i == illuminated_ball else (255, 0, 0)
        cv2.circle(image, (gx, gy), ball_radius, color, -1)

    current_time = time.time()
    for (gx, gy), score_time in list(score_positions[player_side].items()):
        if current_time - score_time < 2:
            cv2.putText(image, "+1", (gx, gy - ball_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            del score_positions[player_side][(gx, gy)]

    score_position = (11, 30)
    text = f"Points: {scores[player_index]} / 3"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
    rect_start = (score_position[0] - 5, score_position[1] - text_height - 5)
    rect_end = (score_position[0] + text_width + 5, score_position[1] + 5)

    cv2.rectangle(image, rect_start, rect_end, (0, 0, 0), -1)
    cv2.putText(image, text, score_position, font, font_scale, (255, 255, 255), thickness)

    return illuminated_ball, illuminated_start_time

def detect_collision(hand_landmarks, obstacle_positions, width, height, obstacle_radius=30):
    for landmark in hand_landmarks.landmark:
        landmark_x = int(landmark.x * width)
        landmark_y = int(landmark.y * height)
        for obstacle_position in obstacle_positions:
            distance = np.sqrt((landmark_x - obstacle_position[0]) ** 2 + (landmark_y - obstacle_position[1]) ** 2)
            if distance < obstacle_radius:
                return True
    return False

def reset_obstacle_position(obstacle_positions, width, height):
    for obstacle_position in obstacle_positions:
        obstacle_position[0] = random.randint(100, width - 100)
        obstacle_position[1] = random.randint(100, height - 100)

def draw_obstacles(image, obstacle_positions):
    for obstacle_position in obstacle_positions:
        cv2.circle(image, tuple(obstacle_position), 30, (0, 0, 255), -1)

def add_new_obstacle(obstacle_positions, width, height):
    new_obstacle_position = [random.randint(100, width - 100), random.randint(100, height - 100)]
    obstacle_positions.append(new_obstacle_position)


def detect_touch(hand_landmarks, target_position, width, height, target_radius=20):
    for landmark in hand_landmarks.landmark:
        landmark_x = int(landmark.x * width)
        landmark_y = int(landmark.y * height)
        distance = np.sqrt((landmark_x - target_position[0]) ** 2 + (landmark_y - target_position[1]) ** 2)
        if distance < target_radius:
            return True
    return False

def draw_targets(image, target_positions):
    for pos in target_positions:
        cv2.circle(image, pos, 20, (0, 255, 0), -1)

def setup_level(level, screen_width, screen_height):
    global score, goal_positions, ball_position, ball_velocity, lives, successful_catches, level_3_obstacle_positions, target_positions, level_4_start_time
    score = 0
    width = 1920
    height = 1080
    lives = 3
    ball_position = [640, 0]
    ball_velocity = [0, 5]
    successful_catches = 0
    level_3_obstacle_positions = [[random.randint(100, width - 100), random.randint(100, height - 100)]]
    target_positions = []
    level_4_start_time = None
    if level == 1:
        goal_positions = init_goal_positions(screen_width, screen_height)
    elif level == 2:
        goal_positions = init_goal_positions(screen_width, screen_height)
    elif level == 3:
        goal_positions = init_goal_positions(screen_width, screen_height)
    elif level == 4:
        target_positions = init_target_positions(screen_width, screen_height)
    print(f"Level {level} setup complete.")  # Debugging statement


def load_and_start_game(mode, loading_window):

    def prepare_resources():
        # Aqui você pode colocar a lógica de inicialização e carregamento dos recursos do seu jogo.
        # Isso inclui carregar modelos do MediaPipe, inicializar o Pygame mixer, carregar sons, etc.
        # pygame.mixer.init()
        # Simulação de carregamento. Substitua isso pelo seu código de carregamento real.
        time.sleep(2)  # Simula o tempo de carregamento com um sleep. Remova isso na implementação final.
    
    def start_game():
        # Fechar a janela de carregamento
        loading_window.destroy()

        # Após o carregamento, inicie o jogo baseado no modo selecionado.
        if mode == "singleplayer":
            start_singleplayer()
        elif mode == "multiplayer":
            start_multiplayer()
        elif mode == "physio":
            Fisio()
        else:
            print(f"Mode {mode} not recognized.")

    # Carregar recursos e iniciar o jogo na thread de fundo para não bloquear a UI
    threading.Thread(target=lambda: [prepare_resources(), tk._default_root.after(0, start_game)], daemon=True).start()

def show_loading_and_start_game(mode):

    loading_window = tk.Toplevel()
    loading_window.title("Loading...")
    
    # Configura a janela de carregamento para ser centralizada na tela
    loading_window.geometry("300x100")  # Define o tamanho da janela
    screen_width = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (300 / 2))
    y_coordinate = int((screen_height / 2) - (100 / 2))
    loading_window.geometry(f"+{x_coordinate}+{y_coordinate}")
    
    loading_label = tk.Label(loading_window, text="Loading mode... Please wait!")
    loading_label.pack(padx=20, pady=20)

    # Desabilita o fechamento manual da janela de carregamento
    loading_window.protocol("WM_DELETE_WINDOW", lambda: None)
    
    # Iniciar carregamento e jogo
    load_and_start_game(mode, loading_window)


def start_singleplayer():
    global level_4_start_time, sound, game_over, goal_positions, current_level, instructions_completed, level_message_completed, score, level_complete_message, speech_recognition_enabled, speech_thread
    speech_recognition_enabled = False
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Single Player", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Single Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Retrieve monitor information and set the window size for the primary monitor
    monitors = get_monitors()
    if monitors:
        primary_monitor = monitors[0]  # Assuming the first monitor is the primary one
        cv2.resizeWindow("Single Player", primary_monitor.width, primary_monitor.height)
        cv2.moveWindow("Single Player", primary_monitor.x, primary_monitor.y)

    cv2.setWindowProperty("Single Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))
    setup_level(current_level, screen_width, screen_height)  # Setup initial level parameters
    instruction_display_time = 5
    level_message_display_time = 2  # Seconds to display the level message
    level_message_last_shown = time.time()
    # Define a variable for the congratulations message display time
    congrats_message_display_time = 2  # At least 1 second for the congratulations message
    last_congrats_message_time = 0  # Initialize with 0
    level_completed = False  # Flag to indicate level completion
    game_over = False

    sound = pygame.mixer.Sound("./Wii_Wonderland.mp3")
    sound.set_volume(0.1)
    sound.play(loops=-1)
  
    if current_level in [3, 4]:
        level_4_start_time = time.time()  # Initialize start time for Level 3 and 4

    is_sound_on = True  # Initialize is_sound_on variable

    level_message_completed = 0
    instructions_completed = 0
  
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Flip the captured image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height, width, _ = image.shape
            current_time = time.time()
            
            # Display the level message for a few seconds
            if current_time < level_message_last_shown + level_message_display_time:
                text_size = cv2.getTextSize(f"Level {current_level}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = (image.shape[0] + text_size[1]) // 2
                cv2.putText(image, f"Level {current_level}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                level_message_completed = level_message_last_shown + level_message_display_time
                cv2.imshow('Single Player', image)
                cv2.waitKey(1)
                continue

            # Display the instructions for a few seconds after the level message disappears
            if current_time < level_message_completed + instruction_display_time:
                show_instructions(image, current_level)
                instructions_completed = level_message_completed + instruction_display_time
                first_attempt = False
                cv2.imshow('Single Player', image)
                cv2.waitKey(1)
                continue

            # Call the appropriate level function based on the current level after the instructions disappear
            level_function = level_functions.get(current_level)
            if level_function and current_time > instructions_completed:
                is_sound_on = level_function(image, results, goal_positions, width, height, cap, sound_on=is_sound_on)

            # Check for level completion
            if score >= level_target_scores[current_level] and not level_completed:
                complete_level(image, cap, sound_on=True)
                last_congrats_message_time = current_time
                level_completed = True
            
            # Display congrats message
            if level_completed and current_time - last_congrats_message_time < congrats_message_display_time:
                text_size = cv2.getTextSize(level_complete_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = (image.shape[0] + text_size[1]) // 2
                cv2.putText(image, level_complete_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Single Player', image)
                cv2.waitKey(1)
                continue
            elif level_completed and current_time - last_congrats_message_time >= congrats_message_display_time: # Reset the level after the congrats message is displayed
                level_message_last_shown = current_time + 1  # Add a delay before starting the next level
                level_completed = False
                setup_level(current_level, screen_width, screen_height)  # Reset the level
            
            if not is_sound_on:
                sound.stop()
            
            if current_level > len(level_target_scores):
                break

            cv2.imshow('Single Player', image)
            if cv2.waitKey(10) & 0xFF == ord('r'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    speech_recognition_enabled = True
    print(f"Speech_flag_a_sair_do_multiplayer = {speech_recognition_enabled}")
    # if speech_thread is None or not speech_thread.is_alive():
    #     recognize_speech_commands(root, command_queue)

    # process_speech_commands(root, command_queue)
    
def exit_game(cap, sound_on):
    # Release the video capture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    sound_on = False
    return sound_on

def get_person_bounding_boxes(frame, draw=False):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for person
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))
                if draw:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    print(f"Drawn box: ({startX}, {startY}), ({endX}, {endY})")
    return boxes

def check_intersection(boxes, screen_width, is_left_screen, is_right_screen):
    crossed = False

    for box in boxes:
        startX, startY, endX, endY = box
        if is_left_screen:
            # Check if any part of the box is beyond the right edge of the left screen
            if endX > screen_width:
                crossed = True
        if is_right_screen:
            # Check if any part of the box is beyond the right edge of the left screen
            if startX < 0:
                crossed = True
    return crossed

def both_hands_in_zone(results, x1, y1, x2, y2, img_width, img_height):
    if results.multi_hand_landmarks:
        hand_in_zone_count = 0
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img_width), int(lm.y * img_height)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hand_in_zone_count += 1
                    break
        return hand_in_zone_count >= 2
    return False

def reset_score(scores):
    scores[0] = 0
    scores[1] = 0
    
def draw_text_with_outline(image, text, position, font, font_scale, text_color, outline_color, thickness, outline_thickness):
    cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, text, position, font, font_scale,  text_color, thickness, cv2.LINE_AA)

def draw_scoreboard(image, game_scores, position, font, font_scale, color, outline_color, thickness, line_type):
    text = f"{game_scores[0]} - {game_scores[1]}"
    # Get the size of the text box
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    
    # Calculate the top-left and bottom-right corners of the rectangle
    top_left = (position[0] - 10, position[1] - text_height - 10)
    bottom_right = (position[0] + text_width + 10, position[1] + 10)
    
    # Draw the rectangle (placard)
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)  # White background
    cv2.rectangle(image, top_left, bottom_right, outline_color, 2)  # Black border

    # Draw the text with outline
    draw_text_with_outline(image, text, position, font, font_scale, color, outline_color, thickness, line_type)

def process_frame(image, hands, screen_width, screen_height, ball_positions, ball_radius, illuminated_ball, illuminated_start_time, scores, start):
    image = cv2.flip(image, 1)
    left_image = image[:, :image.shape[1] // 2]
    right_image = image[:, image.shape[1] // 2:]

    left_results = hands.process(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    right_results = hands.process(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if start:
        illuminated_ball, illuminated_start_time = process_and_draw(left_image, left_results, 'left', left_image.shape[1], left_image.shape[0], scores, 0, ball_positions, ball_radius, illuminated_ball, illuminated_start_time)
        illuminated_ball, illuminated_start_time = process_and_draw(right_image, right_results, 'right', right_image.shape[1], right_image.shape[0], scores, 1, ball_positions, ball_radius, illuminated_ball, illuminated_start_time)

    combined_image = np.hstack((left_image, right_image))
    return combined_image, illuminated_ball, illuminated_start_time, left_results, right_results, results, left_image, right_image

def start_multiplayer():
    global score_positions, speech_recognition_enabled, speech_thread, start
    speech_recognition_enabled = False
    print(f"FLAGGGG = {speech_recognition_enabled}")

    score_positions = {'left': {}, 'right': {}}

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Multiplayer Mode", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Multiplayer Mode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    monitors = get_monitors()
    if monitors:
        primary_monitor = monitors[0]
        cv2.resizeWindow("Multiplayer Mode", primary_monitor.width, primary_monitor.height)
        cv2.moveWindow("Multiplayer Mode", primary_monitor.x, primary_monitor.y)

    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))

    print(f"screen_width = {screen_width}")
    print(f"screen_height = {screen_height}")

    goal_positions = init_goal_positions_multiplayer(screen_width, screen_height)
    scores = [0, 0]
    set_scores = [0, 0]
    game_scores = [0, 0]

    goal_positions = {'left': [(100, 200), (200, 300)], 'right': [(400, 500), (500, 600)]}
    ball_positions = [(random.randint(50, screen_width // 2 - 50), random.randint(50, screen_height - 50)) for _ in range(6)]
    ball_radius = 30
    illuminated_ball = None
    illuminated_start_time = None

    sound = pygame.mixer.Sound("./Elderly_Fun.mp3")
    sound.set_volume(0.1)

    countdown_time = 60
    win_message_start_time = None
    penalty = False
    start = False
    start_game = False
    left_ready = False
    right_ready = False

    sound_flag = True
    is_sound_on = True

    left_ready_start_time = None
    right_ready_start_time = None
    ready_time = None

    game_number = 1
    game_start_time = time.time()
    game_message_displayed = True
    start_time = None
    dislike = False
    like = False
    winner = False

    drawn = False
    overlay_static = None
    frame_counter = 0
    frame_interval=0.05
    random_flag = False

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=4) as hands:
        with ThreadPoolExecutor(max_workers=5) as executor:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                frame_counter += 1

                if frame_counter % 1 == 0:
                    current_time = time.time()
                    
                    future_processed_frame = executor.submit(process_frame, image, hands, screen_width, screen_height, ball_positions, ball_radius, illuminated_ball, illuminated_start_time, scores, start)
                    combined_image, illuminated_ball, illuminated_start_time, left_results, right_results, results, left_image, right_image = future_processed_frame.result()

                    if not start:
                        reset_score(game_scores)
                        reset_score(set_scores)
                        reset_score(scores)
                        game_number = 1
                        left_zone_x1, left_zone_y1, left_zone_x2, left_zone_y2 = 0, 0, left_image.shape[1], left_image.shape[0]
                        right_zone_x1, right_zone_y1, right_zone_x2, right_zone_y2 = 0, 0, right_image.shape[1], right_image.shape[0]

                        left_hands_in_zone = both_hands_in_zone(left_results, left_zone_x1, left_zone_y1, left_zone_x2, left_zone_y2, left_image.shape[1], left_image.shape[0])
                        right_hands_in_zone = both_hands_in_zone(right_results, right_zone_x1, right_zone_y1, right_zone_x2, right_zone_y2, right_image.shape[1], right_image.shape[0])

                        if left_hands_in_zone:
                            if left_ready_start_time is None:
                                left_ready_start_time = time.time()
                            elif time.time() - left_ready_start_time >= 2:
                                left_ready = True
                        else:
                            left_ready_start_time = None

                        if right_hands_in_zone:
                            if right_ready_start_time is None:
                                right_ready_start_time = time.time()
                            elif time.time() - right_ready_start_time >= 2:
                                right_ready = True
                        else:
                            right_ready_start_time = None

                        left_overlay = left_image.copy()
                        right_overlay = right_image.copy()

                        if left_ready:
                            cv2.rectangle(left_overlay, (0, 0), (left_image.shape[1], left_image.shape[0]), (144, 238, 144), -1)
                            left_image = cv2.addWeighted(left_overlay, 0.5, left_image, 0.5, 0)
                            cv2.putText(left_image, "Ready!", (left_image.shape[1] // 2 - 100, left_image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                            left_color = (0, 100, 0)
                        else:
                            cv2.rectangle(left_overlay, (0, 0), (left_image.shape[1], left_image.shape[0]), (50, 50, 255), -1)
                            left_image = cv2.addWeighted(left_overlay, 0.5, left_image, 0.5, 0)
                            cv2.putText(left_image, "Not Ready", (left_image.shape[1] // 2 - 130, left_image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 5, cv2.LINE_AA)
                            left_color = (0, 100, 0) if int(time.time() * 2) % 2 == 0 else (0, 255, 0)

                        if right_ready:
                            cv2.rectangle(right_overlay, (0, 0), (right_image.shape[1], right_image.shape[0]), (144, 238, 144), -1)
                            right_image = cv2.addWeighted(right_overlay, 0.5, right_image, 0.5, 0)
                            cv2.putText(right_image, "Ready!", (right_image.shape[1] // 2 - 100, right_image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                            right_color = (0, 0, 139)
                        else:
                            cv2.rectangle(right_overlay, (0, 0), (right_image.shape[1], right_image.shape[0]), (50, 50, 255), -1)
                            right_image = cv2.addWeighted(right_overlay, 0.5, right_image, 0.5, 0)
                            cv2.putText(right_image, "Not Ready", (right_image.shape[1] // 2 - 130, right_image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 5, cv2.LINE_AA)
                            right_color = (0, 0, 139) if int(time.time() * 2) % 2 == 0 else (0, 0, 255)

                        cv2.rectangle(left_image, (0, 0), (left_image.shape[1], left_image.shape[0]), left_color, 10)
                        cv2.rectangle(right_image, (0, 0), (right_image.shape[1], right_image.shape[0]), right_color, 10)

                        cv2.putText(left_image, "Player 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(right_image, "Player 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

                        combined_image = np.hstack((left_image, right_image))

                        if not left_ready or not right_ready:
                            text_box_x1, text_box_y1 = screen_width // 2 - 125, screen_height - 100
                            text_box_x2, text_box_y2 = screen_width // 2 + 125, screen_height - 50
                            cv2.rectangle(combined_image, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (0, 0, 0), -1)
                            cv2.putText(combined_image, "The first to 3, wins! ", (text_box_x1 + 10, text_box_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            # cv2.putText(combined_image, "", (text_box_x1 + 10, text_box_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        if left_ready and right_ready:
                            if ready_time is None:
                                ready_time = time.time()
                            elif time.time() - ready_time >= 1:
                                start = True
                                game_start_time = time.time()

                        cv2.imshow('Multiplayer Mode', combined_image)
                        if cv2.waitKey(5) & 0xFF == ord('r'):
                            break
                        continue
                    image = cv2.flip(image, 1)
                    if game_message_displayed:
                        combined_image = np.hstack((image[:, :image.shape[1] // 2], image[:, image.shape[1] // 2:]))

                        if game_scores[0] == 3 or game_scores[1] == 3:
                            draw_text_with_outline(combined_image, "GAME ended", (screen_width // 2 - 155, screen_height // 2 - 100),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), (0, 0, 0), 5, 8)
                        else:
                            draw_text_with_outline(combined_image, f"GAME {game_number}", (screen_width // 2 - 110, screen_height // 2 - 100),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), (0, 0, 0), 5, 8)

                        position = (screen_width // 2 - 100, screen_height // 2 + 50)
                        draw_scoreboard(combined_image, game_scores, position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), (0, 0, 0), 5, 8)

                        if game_scores[0] == 3 or game_scores[1] == 3:
                            draw_text_with_outline(combined_image, "Do you want to play again?", (screen_width // 2 - 350, screen_height // 2 + 125),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255, 0, 0), (0, 0, 0), 5, 8)

                            like, dislike = draw_circles_v2(cap, combined_image, results)
                            draw_end_game(combined_image, screen_width)
                            if dislike and not like:
                                is_sound_on = exit_game(cap, is_sound_on)
                                break
                            elif like and not dislike:
                                start = False
                                left_ready = False
                                right_ready = False
                                continue
                        else:
                            current_time = time.time()
                            if current_time >= game_start_time + 2:
                                game_message_displayed = False
                                start_game = True
                                start_time = time.time()
                        cv2.imshow('Multiplayer Mode', combined_image)
                        if cv2.waitKey(5) & 0xFF == ord('r'):
                            break
                        continue

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    remaining_time = max(0, int(countdown_time - elapsed_time))
                    progress = (elapsed_time / countdown_time) * 360

                    if sound_flag:
                        sound.play(loops=-1)
                        sound_flag = False

                    is_sound_on_left_side = draw_circle_and_check_end(cap, screen_height, left_image, left_results, is_right_half=False, is_single_player=False, sound_on=True, fisio = False)

                    is_sound_on_right_side = draw_circle_and_check_end(cap, screen_height, right_image, right_results, is_right_half=True, is_single_player=False, sound_on=True, fisio = False)


                    left_boxes = get_person_bounding_boxes(left_image, draw=False)
                    right_boxes = get_person_bounding_boxes(right_image, draw=False)

                    illuminated_ball, illuminated_start_time = process_and_draw(left_image, left_results, 'left', left_image.shape[1], left_image.shape[0], scores, 0, ball_positions, ball_radius, illuminated_ball, illuminated_start_time)
                    illuminated_ball, illuminated_start_time = process_and_draw(right_image, right_results, 'right', right_image.shape[1], right_image.shape[0], scores, 1, ball_positions, ball_radius, illuminated_ball, illuminated_start_time)
                    combined_image = np.hstack((left_image, right_image))

                    if not drawn:
                        overlay_static = np.zeros_like(combined_image)
                        draw_power_off(overlay_static, screen_width // 2)
                        draw_power_off(overlay_static[:, screen_width // 2:], screen_width // 2)
                        line_height = screen_height - 101
                        cv2.line(overlay_static, (screen_width // 2, 0), (screen_width // 2, line_height), (255, 255, 255), 3)
                        box_width = screen_width // 3
                        box_height = 100
                        box_x = (screen_width - box_width) // 2
                        box_y = screen_height - box_height
                        cv2.rectangle(overlay_static, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)

                        cv2.putText(overlay_static, "Player 1", (box_x + 5, box_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                        text_width_p1, _ = cv2.getTextSize("Player 1    ", cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                        text_width_p1 = text_width_p1[0]
                        cv2.line(overlay_static, (box_x + 5, box_y + 45), (box_x + 5 + text_width_p1, box_y + 45), (255, 255, 255), 2)

                        cv2.putText(overlay_static, "Player 2", (box_x + box_width - 160, box_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                        text_width, _ = cv2.getTextSize("Player 2  ", cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                        text_width = text_width[0]
                        cv2.line(overlay_static, (box_x + box_width - 160, box_y + 45), (box_x + box_width - 160 + text_width, box_y + 45), (255, 255, 255), 2)

                        center_coordinates = (box_x + box_width // 2, box_y + box_height // 2)
                        radius = 40
                        thickness = 5
                        cv2.circle(overlay_static, center_coordinates, radius, (255, 255, 255), thickness)

                        drawn = True

                    combined_image = cv2.addWeighted(combined_image, 1, overlay_static, 1, 0)
                    cv2.ellipse(combined_image, center_coordinates, (radius, radius), 0, 0, -progress, (0, 255, 0), thickness)
                    player1_crossed = check_intersection(left_boxes, image.shape[1] // 2, is_left_screen=True, is_right_screen=False)
                    player2_crossed = check_intersection(right_boxes, image.shape[1] // 2, is_left_screen=False, is_right_screen=True)

                    if player1_crossed or player2_crossed:
                        penalty = True
                        cv2.putText(combined_image, "Get to your positions!", (screen_width // 2 - 200, screen_height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

                    if (scores[1] < 3) and (scores[0] == 3):
                        set_scores[0] += 1
                        reset_score(scores)
                        if not random_flag:
                            ball_positions = [(random.randint(50, screen_width // 2 - 50), random.randint(50, screen_height - 50)) for _ in range(6)]
                            random_flag = True
                    elif (scores[1] == 3) and (scores[0] < 3):
                        set_scores[1] += 1
                        reset_score(scores)
                        if not random_flag:
                            ball_positions = [(random.randint(50, screen_width // 2 - 50), random.randint(50, screen_height - 50)) for _ in range(6)]
                            random_flag = True

                    random_flag = False
                    cv2.putText(combined_image, f"{set_scores[0]}", (box_x + 5, box_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(combined_image, f"{set_scores[1]}", (box_x + box_width - 160, box_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if remaining_time >= 10:
                        cv2.putText(combined_image, f"{remaining_time}s", (box_x + box_width // 2 - 31, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(combined_image, f"{remaining_time}s", (box_x + box_width // 2 - 18, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

                    if set_scores[0] == 3 and set_scores[1] < 3:
                        winner = True
                        print(f"Jogador 1 = {game_scores[0]}")
                        if win_message_start_time is None:
                            win_message_start_time = time.time()
                        cv2.putText(combined_image, "Player 1 won!", (screen_width // 4 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                        if time.time() > (win_message_start_time + 2):
                            win_message_start_time = None
                            game_message_displayed = True
                            game_start_time = time.time()
                            game_scores[0] += 1
                            game_number += 1
                            reset_score(set_scores)
                            reset_score(scores)
                            countdown_time = 60
                            start_time = time.time()

                    elif set_scores[1] == 3 and set_scores[0] < 3:
                        winner = True
                        print(f"Jogador 2 = {game_scores[1]}")
                        if win_message_start_time is None:
                            win_message_start_time = time.time()
                        cv2.putText(combined_image, "Player 2 won!", (3 * screen_width // 4 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                        if time.time() > (win_message_start_time + 2):
                            win_message_start_time = None
                            game_message_displayed = True
                            game_start_time = time.time()
                            game_scores[1] += 1
                            game_number += 1
                            reset_score(set_scores)
                            reset_score(scores)
                            countdown_time = 60
                            start_time = time.time()

                    if remaining_time == 0 and not winner:
                        if set_scores[1] > set_scores[0]:
                            if win_message_start_time is None:
                                win_message_start_time = time.time()
                            cv2.putText(combined_image, "Player 2 won!", (3 * screen_width // 4 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                            if time.time() > (win_message_start_time + 2):
                                win_message_start_time = None
                                game_message_displayed = True
                                game_start_time = time.time()
                                game_scores[1] += 1
                                game_number += 1
                                reset_score(set_scores)
                                reset_score(scores)
                                countdown_time = 60
                                start_time = time.time()

                        elif set_scores[1] < set_scores[0]:
                            if win_message_start_time is None:
                                win_message_start_time = time.time()
                            cv2.putText(combined_image, "Player 1 won!", (screen_width // 4 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                            if time.time() > (win_message_start_time + 2):
                                win_message_start_time = None
                                game_message_displayed = True
                                game_start_time = time.time()
                                game_scores[0] += 1
                                game_number += 1
                                reset_score(set_scores)
                                reset_score(scores)
                                countdown_time = 60
                                start_time = time.time()
                        else:
                            if win_message_start_time is None:
                                win_message_start_time = time.time()
                            if scores[0] > scores[1]:
                                draw = False
                                flag = 0
                                cv2.putText(combined_image, "Player 1 won!", (screen_width // 4 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            elif scores[0] < scores[1]:
                                draw = False
                                flag = 1
                                cv2.putText(combined_image, "Player 2 won!", (3 * screen_width // 4 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            else:
                                draw = True
                                draw_text_with_outline(combined_image, "Draw!", (screen_width // 4 - 100, screen_height // 2),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1.75, (128, 128, 128), (0, 0, 0), 4, 6)
                                draw_text_with_outline(combined_image, "Draw!", (3 * screen_width // 4 - 100, screen_height // 2),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1.75, (128, 128, 128), (0, 0, 0), 4, 6)

                            if time.time() > (win_message_start_time + 2):
                                win_message_start_time = None
                                game_message_displayed = True
                                game_start_time = time.time()
                                if not draw:
                                    game_scores[flag] += 1
                                game_number += 1
                                reset_score(set_scores)
                                reset_score(scores)
                                countdown_time = 60
                                start_time = time.time()

                    current_time = time.time()
                    if illuminated_ball is None and (current_time - start_time) % 2 < 0.1:
                        illuminated_ball = random.randint(0, len(ball_positions) - 1)
                        illuminated_start_time = current_time

                    if illuminated_ball is not None and current_time - illuminated_start_time > 2:
                        illuminated_ball = None

                    cv2.imshow('Multiplayer Mode', combined_image)

                    if not is_sound_on_left_side or not is_sound_on_right_side or not is_sound_on:
                        sound.stop()

                    if cv2.waitKey(5) & 0xFF == ord('r'):
                        break

    cap.release()
    cv2.destroyAllWindows()

    speech_recognition_enabled = True
    print(f"Speech_flag_a_sair_do_multiplayer = {speech_recognition_enabled}")
    # if speech_thread is None or not speech_thread.is_alive():
    #     recognize_speech_commands(root, command_queue)

    # process_speech_commands(root, command_queue)
    # recognize_speech_commands(root)   
    
def load_and_resize_image(path, size=(150, 150)):
    # Load the image
    image = Image.open(path)
    # Resize to the specified size using the high-quality Lanczos filter
    image = image.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)

def play_hover_sound(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.set_volume(1)  # Adjust volume
    pygame.mixer.music.play()



def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1+radius, y1,
              x1+radius, y1,
              x2-radius, y1,
              x2-radius, y1,
              x2, y1,
              x2, y1+radius,
              x2, y1+radius,
              x2, y2-radius,
              x2, y2-radius,
              x2, y2,
              x2-radius, y2,
              x2-radius, y2,
              x1+radius, y2,
              x1+radius, y2,
              x1, y2,
              x1, y2-radius,
              x1, y2-radius,
              x1, y1+radius,
              x1, y1+radius,
              x1, y1]

    return canvas.create_polygon(points, **kwargs, smooth=True)

def create_rounded_rectangle_image(width, height, radius, fill_color, outline_color):
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width, height), radius, fill=fill_color, outline=outline_color, width=5)
    return ImageTk.PhotoImage(image)

#PHYSIO-----------------------------------------------------------------------------------------
# Função para sobrepor uma imagem com transparência
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

def save_score_to_csv(score, joint):
    csv_filename = 'scores.csv'
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([joint, score])
    print(f'Score for {joint} saved to {csv_filename}')

# Função para carregar os scores do arquivo CSV
def load_scores_from_csv():
    scores = {}
    if os.path.isfile(csv_filename):
        with open(csv_filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)  # Pula o cabeçalho
            for row in reader:
                joint, score = row
                scores[joint] = int(score)
    return scores

# Função para obter o último número de repetições de um exercício específico
def get_last_reps(joint):
    scores = load_scores_from_csv()
    return scores.get(joint, 0)

# Função para inicializar o arquivo CSV se ele não existir
def initialize_csv():
    if not os.path.isfile(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Joint', 'Score'])  # Adiciona cabeçalhos
    print(f'CSV file initialized: {csv_filename}')

# Função para calcular o ângulo
def calculate_angle(a, b, c):
    a = np.array(a)  # Ponto A
    b = np.array(b)  # Ponto B
    c = np.array(c)  # Ponto C
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# Função para desenhar a barra de progresso circular
def draw_circular_progress_bar(image, progress, center, radius=100, thickness=20):
    angle = int(360 * progress / 100)
    cv2.circle(image, center, radius, (0, 0, 255), thickness + 10)
    cv2.circle(image, center, radius, (255, 255, 255), thickness)
    if progress > 0:
        cv2.ellipse(image, center, (radius, radius), 0, 0, angle, (0, 255, 0), thickness)
    font_scale = 5
    thickness = 4
    text = str(score)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

# Função para desenhar a barra de progresso vertical
def draw_vertical_progress_bar(image, progress, position, size=(30, 400), thickness=20, segments=10):
    x, y = position
    w, h = size
    segment_height = h // segments
    progress = np.clip(progress, 0, 100)
    filled_segments = int(segments * (progress / 100))
    cv2.rectangle(image, (x, y), (x + thickness, y + h), (255, 255, 255), -1)
    for i in range(filled_segments):
        segment_y = y + h - (i + 1) * segment_height
        cv2.rectangle(image, (x, segment_y), (x + thickness, segment_y + segment_height), (0, 255, 0), -1)
        cv2.rectangle(image, (x, segment_y), (x + thickness, segment_y + segment_height), (0, 0, 255), 2)
    for i in range(filled_segments, segments):
        segment_y = y + h - (i + 1) * segment_height
        cv2.rectangle(image, (x, segment_y), (x + thickness, segment_y + segment_height), (255, 255, 255), -1)
        cv2.rectangle(image, (x, segment_y), (x + thickness, segment_y + segment_height), (0, 0, 255), 2)

# Função para desenhar o retângulo arredondado com texto
def draw_rounded_rectangle_with_text(image, text, rect_position, rect_size, radius=20, color=(128, 128, 128), alpha=0.5):
    overlay = image.copy()
    x, y = rect_position
    w, h = rect_size
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h ), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    font_scale = 2
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + text_size[1] + 20
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# Função para desenhar fotos na tela com retângulos vermelhos semitransparentes
def draw_photos(image, photo_images, centers):
    photo_size = (125, 175)  # Altere o tamanho das imagens aqui
    rect_padding = 20  # Tamanho do padding para o retângulo maior
    for center, photo_img in zip(centers, photo_images):
        x, y = center
        overlay_img = cv2.resize(photo_img, photo_size)  # Redimensione conforme necessário

        # Desenha o retângulo vermelho semitransparente maior
        overlay = image.copy()
        cv2.rectangle(overlay, (x - photo_size[0] // 2 - rect_padding, y - photo_size[1] // 2 - rect_padding),
                      (x + photo_size[0] // 2 + rect_padding, y + photo_size[1] // 2 + rect_padding), (255, 255, 0), -1)
        cv2.rectangle(overlay, (x - photo_size[0] // 2 - rect_padding, y - photo_size[1] // 2 - rect_padding),
                      (x + photo_size[0] // 2 + rect_padding, y + photo_size[1] // 2 + rect_padding), (0, 0, 0), 3)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        if overlay_img.shape[2] == 4:
            alpha_channel = overlay_img[:, :, 3] / 255.0
            overlay_image_alpha(image, overlay_img[:, :, :3], (x - photo_size[0] // 2, y - photo_size[1] // 2), alpha_channel)
        else:
            image[y-photo_size[1]//2:y+photo_size[1]//2, x-photo_size[0]//2:x+photo_size[0]//2] = overlay_img

# Função para verificar se o dedo está dentro de uma área da foto
def check_finger_in_photo(finger_pos, centers, photo_size=150):  # Certifique-se de que o tamanho esteja sincronizado
    radius = photo_size // 2
    for i, center in enumerate(centers):
        distance = np.linalg.norm(finger_pos - np.array(center))
        if distance < radius:
            return i
    return -1


def draw_menu(image, width, height):
    icon = cv2.imread('./poweroff.png', cv2.IMREAD_COLOR)  # Assuming your icon is in color
    if icon is None:
        raise FileNotFoundError("The power off icon image file could not be loaded.")

    icon = cv2.resize(icon, (30, 30))  # Adjust dimensions as needed

    # Calculate the starting x coordinate (centered horizontally)
    start_x = (width - icon.shape[1]) // 2 + 1
    print(f" shape 1= {icon.shape[1]}")
    # Calculate the starting y coordinate (slightly towards the bottom)
    # Assuming we want to place it 10% from the bottom
    start_y = int(image.shape[0] * 0.9 - 45)
    
    print(f" shape = {icon.shape[0]}")
    
    # Ensure we're not going out of the image bounds
    if start_x >= 0 and start_y >= 0:
        # Draw the power off icon on the image
        image[start_y:start_y+icon.shape[0], start_x:start_x+icon.shape[1]] = icon
    else:
        raise ValueError("The icon cannot be placed within the bounds of the image.")

def Fisio():
    global joint_to_monitor, score, is_flexing, speech_recognition_enabled
    speech_recognition_enabled = False
    
    print(f"fisio_ = {speech_recognition_enabled}")
    # Carrega as imagens dos exercícios
    photo_images = [
        cv2.imread('./corpo/CotD.png', cv2.IMREAD_UNCHANGED),
        cv2.imread('./corpo/CotE.png', cv2.IMREAD_UNCHANGED),
        cv2.imread('./corpo/JoelhoE.png', cv2.IMREAD_UNCHANGED),
        cv2.imread('./corpo/JoelhoD.png', cv2.IMREAD_UNCHANGED),
        cv2.imread('./corpo/MaoE.png', cv2.IMREAD_UNCHANGED),
        cv2.imread('./corpo/MaoD.png', cv2.IMREAD_UNCHANGED),
        cv2.imread('./corpo/Pescoco.png', cv2.IMREAD_UNCHANGED)
    ]
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Exercise", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Exercise", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Retrieve monitor information and set the window size for the primary monitor
    monitors = get_monitors()
    if monitors:
        primary_monitor = monitors[0]
        cv2.resizeWindow("Exercise", primary_monitor.width, primary_monitor.height)
        cv2.moveWindow("Exercise", primary_monitor.x, primary_monitor.y)
    
    # cv2.setWindowProperty("Exercise", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Define o tempo de transição entre as fases (em segundos)
    # selection_duration = 2  # Tempo para selecionar o exercício
    start_time = time.time()
    good_job_time = 0
    display_good_job = False
    scores = load_scores_from_csv()
    joint_to_monitor = None
    
    score = 0
    image_index = 0
    
    # Obtém o último número de repetições
    rect_x, rect_y = 100, 0
    rect_w, rect_h = 700, 0
    
    # Variáveis de controle
    selecting_exercise = True
    good_job_time = 2
    first = True
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            height, width, _ = image.shape
            
            current_time = time.time()
            
            if selecting_exercise:
                # Mostrar tela de seleção de exercício
                num_photos = len(photo_images)
                space_between_photos = (width - num_photos * 150) // (num_photos + 1)  # Ajuste o espaço conforme o tamanho das fotos
                centers = [(space_between_photos * (i + 1) + 150 * i + 75, height // 4) for i in range(num_photos)]
                draw_photos(image, photo_images, centers)
    
                text_scale = 2
                text_thickness = 6
                cv2.putText(image, "SELECT EXERCISE", (width // 2 - 300, height // 8 - 60), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness + 3, cv2.LINE_AA)
                cv2.putText(image, "SELECT EXERCISE", (width // 2 - 300, height // 8 - 60), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 0), text_thickness, cv2.LINE_AA)
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                is_sound_on = draw_circle_and_check_end(cap, height, image, results, is_right_half=False, is_single_player=False, sound_on = False, fisio = True)
                if first == True:
                    overlay_static = np.zeros_like(image)
                    draw_menu(overlay_static, width, height)
                    first = False
                    
                image = cv2.addWeighted(image, 1, overlay_static, 1, 0)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    finger_pos = np.multiply([landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                                              landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y],
                                             [width, height]).astype(int)
                    cv2.circle(image, tuple(finger_pos), 10, (255, 0, 0), -1)
                    
                    selected_photo = check_finger_in_photo(finger_pos, centers, photo_size=150)
                    if selected_photo != -1:
                        if selected_photo == 0:
                            joint_to_monitor = 'right_elbow'
                        elif selected_photo == 1:
                            joint_to_monitor = 'left_elbow'
                        elif selected_photo == 2:
                            joint_to_monitor = 'right_knee'
                        elif selected_photo == 3:
                            joint_to_monitor = 'left_knee'
                        elif selected_photo == 4:
                            joint_to_monitor = 'right_wrist'
                        elif selected_photo == 5:
                            joint_to_monitor = 'left_wrist'
                        elif selected_photo == 6:
                            joint_to_monitor = 'neck'
                        selecting_exercise = False
                        start_time = current_time
            else:
                if results.pose_landmarks and score != 10:
                    landmarks = results.pose_landmarks.landmark
                    height, width, _ = image.shape
                    
                    last_reps = get_last_reps(joint_to_monitor)
                    
                    #botao off - fazer if

                    rect_y = height // 4 - 200
                    rect_h = height // 2 + 400

                    if joint_to_monitor == 'right_elbow':
                        shoulder = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y], [width, height]).astype(int)
                        elbow = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y], [width, height]).astype(int)
                        wrist = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y], [width, height]).astype(int)
                        angle = calculate_angle(shoulder, elbow, wrist)
                        desired_angle = 15
                        joint_text = "LEFT ELBOW"
                        current_images = elbowR_images

                    elif joint_to_monitor == 'left_elbow':
                        shoulder = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y], [width, height]).astype(int)
                        elbow = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y], [width, height]).astype(int)
                        wrist = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y], [width, height]).astype(int)
                        angle = calculate_angle(shoulder, elbow, wrist)
                        desired_angle = 15
                        joint_text = "RIGHTLBOW"
                        current_images = elbowL_images

                    elif joint_to_monitor == 'left_wrist':
                        finger = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].y], [width, height]).astype(int)
                        wrist = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y], [width, height]).astype(int)
                        elbow = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y], [width, height]).astype(int)
                        angle = calculate_angle(finger, wrist, elbow)
                        desired_angle = 130
                        joint_text = "RIGHT WRIST"
                        current_images = wristR_images # Você pode mudar para um conjunto diferente de imagens para exercícios de pulso se tiver

                    elif joint_to_monitor == 'right_wrist':
                        finger = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y], [width, height]).astype(int)
                        wrist = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y], [width, height]).astype(int)
                        elbow = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y], [width, height]).astype(int)
                        
                        angle = calculate_angle(finger, wrist, elbow)
                        desired_angle = 130
                        joint_text = "LEFT WRIST"
                        current_images = wristL_images  # Você pode mudar para um conjunto diferente de imagens para exercícios de pulso se tiver


                    elif joint_to_monitor == 'right_knee':
                        hip = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x,
                                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y], [width, height]).astype(int)
                        knee = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x,
                                            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y], [width, height]).astype(int)
                        ankle = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y], [width, height]).astype(int)
                        angle = calculate_angle(hip, knee, ankle)
                        desired_angle = 90
                        joint_text = "LEFT KNEE"
                        current_images = kneeL_images

                    elif joint_to_monitor == 'left_knee':
                        hip = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y], [width, height]).astype(int)
                        knee = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y], [width, height]).astype(int)
                        ankle = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y], [width, height]).astype(int)
                        angle = calculate_angle(hip, knee, ankle)
                        desired_angle = 90
                        joint_text = "RIGHT KNEE"
                        current_images = kneeL_images

                    elif joint_to_monitor == 'neck':
                        left_ear = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].x,
                                                landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].y], [width, height]).astype(int)
                        right_ear = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].x,
                                                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].y], [width, height]).astype(int)
                        left_shoulder = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y], [width, height]).astype(int)
                        right_shoulder = np.multiply([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y], [width, height]).astype(int)
                        shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0).astype(int)
                        joint_text = "NECK"
                        current_images = neck_images  # Você pode mudar para um conjunto diferente de imagens para exercícios de pescoço se tiver

                        # Calcula o ângulo entre a orelha, o ombro correspondente e o ombro oposto
                        left_angle = calculate_angle(left_ear, left_shoulder, right_shoulder)
                        right_angle = calculate_angle(right_ear, right_shoulder, left_shoulder)

                        # Considera a menor dos dois ângulos calculados
                        angle = min(left_angle, right_angle)

                        # Print dos valores para depuração
                        #print(f"Left ear: {left_ear}, Right ear: {right_ear}")
                        #print(f"Left shoulder: {left_shoulder}, Right shoulder: {right_shoulder}")
                        #print(f"Left angle: {left_angle}, Right angle: {right_angle}")
                        #print(f"Selected angle: {angle}")

                        if angle < 40:  # Threshold para considerar uma repetição
                            if not is_flexing:
                                score += 1
                                is_flexing = True
                        else:
                            is_flexing = False

                    if joint_to_monitor != 'neck':
                        if angle < desired_angle and not is_flexing:
                            score += 1
                            is_flexing = True
                        elif angle >= desired_angle:
                            is_flexing = False
                    
                    progress = (180 - angle) / (180 - desired_angle) * 100 if joint_to_monitor != 'neck' else (30 - angle) / 30 * 100
                    progress = max(0, min(100, progress))
                    
                    if joint_to_monitor in ['right_elbow', 'left_elbow']:
                        cv2.circle(image, tuple(shoulder), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(elbow), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(wrist), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(shoulder), 20, (255, 0, 0), 2)
                        cv2.circle(image, tuple(elbow), 20, (255, 0, 0), 2)
                        cv2.circle(image, tuple(wrist), 20, (255, 0, 0), 2)
                        cv2.line(image, tuple(shoulder), tuple(elbow), (255, 255, 255), 2)
                        cv2.line(image, tuple(elbow), tuple(wrist), (255, 255, 255), 2)
                    elif joint_to_monitor in ['right_knee', 'left_knee']:
                        cv2.circle(image, tuple(hip), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(knee), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(ankle), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(hip), 20, (255, 0, 0), 2)
                        cv2.circle(image, tuple(knee), 20, (255, 0, 0), 2)
                        cv2.circle(image, tuple(ankle), 20, (255, 0, 0), 2)
                        cv2.line(image, tuple(hip), tuple(knee), (255, 255, 255), 2)
                        cv2.line(image, tuple(knee), tuple(ankle), (255, 255, 255), 2)
                    elif joint_to_monitor == 'neck':
                        cv2.circle(image, tuple(left_ear), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(right_ear), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(left_shoulder), 10, (0, 255, 0), -1)
                        cv2.circle(image, tuple(right_shoulder), 10, (0, 255, 0), -1)
                        # Desenha linhas para visualização
                        cv2.line(image, tuple(left_ear), tuple(left_shoulder), (255, 255, 255), 2)
                        cv2.line(image, tuple(right_ear), tuple(right_shoulder), (255, 255, 255), 2)
                        cv2.line(image, tuple(left_shoulder), tuple(right_shoulder), (255, 255, 255), 2)
                    elif joint_to_monitor in ['right_wrist', 'left_wrist']:
                        cv2.circle(image, tuple(finger), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(wrist), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(elbow), 10, (255, 0, 0), -1)
                        cv2.circle(image, tuple(finger), 20, (255, 0, 0), 2)
                        cv2.circle(image, tuple(wrist), 20, (255, 0, 0), 2)
                        cv2.circle(image, tuple(elbow), 20, (255, 0, 0), 2)
                        cv2.line(image, tuple(finger), tuple(wrist), (255, 255, 255), 2)
                        cv2.line(image, tuple(wrist), tuple(elbow), (255, 255, 255), 2)

                    
                    draw_rounded_rectangle_with_text(image, joint_text, (rect_x, rect_y), (rect_w, rect_h), radius=20, color=(128, 128, 128), alpha=0.5)
                    
                    # Adiciona o texto "Last nr of reps: {last_reps}" logo abaixo do nome da parte do corpo
                    cv2.putText(image, f"Last nr of reps: {last_reps}", (rect_x + 20, rect_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    vertical_progress = min(score / 10 * 100, 100)
                    
                    circle_center_x = rect_x + rect_w // 3
                    circle_center_y = rect_y + rect_h // 2
                    vertical_bar_x = rect_x + 2 * rect_w // 3 + 45
                    vertical_bar_y = rect_y + 150
                    vertical_bar_h = rect_h - 300

                    current_time = time.time()
                    if current_time - start_time >= 1:
                        image_index = (image_index + 1) % len(current_images)
                        start_time = current_time
                    
                    overlay_img = current_images[image_index]
                    overlay_img = cv2.resize(overlay_img, (rect_w - 50, rect_h // 3 + 150))
                    img_x, img_y = rect_x + 50, rect_y + rect_h - overlay_img.shape[0]

                    if overlay_img.shape[2] == 4:
                        alpha_channel = overlay_img[:, :, 3] / 255.0
                        overlay_image_alpha(image, overlay_img[:, :, :3], (img_x, img_y), alpha_channel)
                    else:
                        image[img_y:img_y + overlay_img.shape[0], img_x:img_x + overlay_img.shape[1]] = overlay_img

                    draw_circular_progress_bar(image, progress, center=(circle_center_x, circle_center_y - 140), radius=150, thickness=45)
                    draw_vertical_progress_bar(image, vertical_progress, position=(vertical_bar_x, vertical_bar_y), size=(90, 350), thickness=60)
                
                elif score == 10:
                    if not display_good_job:
                        save_score_to_csv(score, joint_to_monitor)
                        display_good_job = True
                        good_job_time = time.time()
                
                    current_time = time.time()
                    if display_good_job:
                        if current_time - good_job_time < 2:
                            # Define the text parameters
                            text = "Good job!"
                            position = (width // 2 - 100, height // 2)  # Adjust as needed
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.75
                            text_color = (0, 0, 255)  # Red
                            outline_color = (255, 255, 255)  # White
                            thickness = 4
                            outline_thickness = 8
                            
                            # Draw the text with outline
                            draw_text_with_outline(image, text, position, font, font_scale, text_color, outline_color, thickness, outline_thickness)
                        else:
                            display_good_job = False
                            score = 0
                            selecting_exercise = True
                        
            cv2.imshow('Exercise', image)
            if cv2.waitKey(10) & 0xFF == ord('r'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    speech_recognition_enabled = True
    print(f"fisio_fim = {speech_recognition_enabled}")
    # Atualiza o score no dicionário se for maior que o valor existente
    if joint_to_monitor in scores:
        if score > scores[joint_to_monitor]:
            scores[joint_to_monitor] = score
    else:
        scores[joint_to_monitor] = score

    # Salva os scores atualizados no arquivo CSV
    save_score_to_csv(scores)

def recognize_speech_commands(root):
    """Recognizes speech commands in the background and triggers button actions."""
    global speech_recognition_enabled, speech
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    recognizer.energy_threshold = 50  # Lower value increases sensitivity
    recognizer.dynamic_energy_threshold = True  # Adjusts dynamically
    
    def listen_and_recognize():
        global speech_recognition_enabled
        
        with microphone as source:
            
            recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    # recognizer.adjust_for_ambient_noise(source)
                    print("Listening for commands...")
                    # root.after(0, lambda: pulse_border(root, True))
                    audio = recognizer.listen(source)
                    # root.after(0, lambda: pulse_border(root, False))
                    command = recognizer.recognize_google(audio).lower()
                    print(f"Recognized command: {command}")
                    
                    if any(keyword in command for keyword in ["single player", "single", "singleplayer", "first mode", "first button", "primeiro botão", "primeiro"]):
                        root.after(0, lambda: show_loading_and_start_game("singleplayer"))
                    elif any(keyword in command for keyword in ["multi player", "multiplayer", "multiplier" ,"second mode", "second button", "segundo botão"]):
                        root.after(0, lambda: show_loading_and_start_game("multiplayer"))
                    elif any(keyword in command for keyword in ["visio", "Phisio", "Physio", "Phisyo" ,"fisio", "teraphy button", " therapy button", "physiotherapy", "physio"]):
                        root.after(0, lambda: show_loading_and_start_game("physio"))
                    elif any(keyword in command for keyword in ["quit", "exit", "close", "desliga"]):
                        root.after(0, lambda: on_close(root))
                        break
                        
                except sr.UnknownValueError:
                    print("Could not understand the audio.")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                except Exception as e:
                    print(f"Error during speech recognition: {e}")

    # Start speech recognition in a background thread
    speech = threading.Thread(target=listen_and_recognize, daemon=True)
    speech.start()

# Define the window close handler
def on_close(root):
    global speech_recognition_enabled 
    pygame.mixer.music.stop()  # Stop music playback
    pygame.mixer.quit()  # Quit the mixer to clean up resources
    speech_recognition_enabled = False
    
    # Signal the speech thread to stop
    root.destroy()  # Destroy the Tkinter window

def pulse_border(widget, listening):
    """Pulses the border of the widget to indicate listening state."""
    if listening:
        color_cycle = ["red", "orange", "yellow", "green", "blue", "purple"]
        cycle_length = 1000
        def animate(i=0):
            color = color_cycle[i % len(color_cycle)]
            widget.config(highlightbackground=color, highlightcolor=color, highlightthickness=10)
            if speech_recognition_enabled:
                widget.after(cycle_length // len(color_cycle), animate, i + 1)
        animate()
    else:
        widget.config(highlightthickness=0)

def blink_image(label, interval=500):
    """Makes the given label blink."""
    def toggle_visibility():
        current_state = label.winfo_viewable()
        if current_state:
            label.place_forget()
        else:
            label.place(x=screen_width // 2, y=screen_height // 1.8, anchor='center')
        label.after(interval, toggle_visibility)
    toggle_visibility()

def main():
    global first_it, speech_recognition_enabled
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    print(f"Speech_flag_main_1_It = {speech_recognition_enabled}")
    print("Tou na hub")

    if first_it:
        play_hover_sound("./welcome.mp3")
        first_it = False
        
    recognize_speech_commands(root)    
    background_image = Image.open("./Experimento4.png")
    new_size = (screen_width, screen_height)
    resized_background_image = background_image.resize(new_size, Image.LANCZOS)
    background_photo = ImageTk.PhotoImage(resized_background_image)
    background_label = tk.Label(root, image=background_photo)
    background_label.place(relwidth=1, relheight=1)
    
    games_bg_image = create_rounded_rectangle_image(450, 150, 30, (255, 255, 255, 153), (0, 0, 0, 255))
    physio_bg_image = create_rounded_rectangle_image(500, 150, 30, (255, 255, 255, 153), (0, 0, 0, 255))

    games_bg_label = tk.Label(root, image=games_bg_image, bd=0, bg='white')
    games_bg_label.place(x=screen_width // 4.2, y=screen_height // 2.2, anchor='center')

    games_label = tk.Label(root, text="Games", font=("Helvetica", 25), bg='white')
    games_label.place(x=screen_width // 4.2, y=screen_height // 2.55, anchor='center')

    singleplayer_button = ttk.Button(root, text="Singleplayer", command=lambda: show_loading_and_start_game("singleplayer"), style="my.TButton")
    singleplayer_button.place(x=screen_width // 5.4, y=screen_height // 2.2, anchor="center")
    # singleplayer_button.bind("<Enter>", lambda e: play_hover_sound("./home/elderly/Desktop/elderlyactive-game/digital-beeping-151921.mp3"))

    multiplayer_button = ttk.Button(root, text="Multiplayer", command=lambda: show_loading_and_start_game("multiplayer"), style="my.TButton")
    multiplayer_button.place(x=screen_width // 3.4, y=screen_height // 2.2, anchor="center")
    # multiplayer_button.bind("<Enter>", lambda e: play_hover_sound("./digital-beeping-151921.mp3"))

    physio_bg_label = tk.Label(root, image=physio_bg_image, bd=0, bg='white')
    physio_bg_label.place(x=screen_width // 1.3, y=screen_height // 2.2, anchor='center')

    physio_label = tk.Label(root, text="Therapy", font=("Helvetica", 25), bg='white')
    physio_label.place(x=screen_width // 1.3, y=screen_height // 2.55, anchor='center')

    physiotherapy_button = ttk.Button(root, text="Physiotherapy", command=lambda: show_loading_and_start_game("physio"), style="my.TButton")
    physiotherapy_button.place(x=screen_width // 1.3, y=screen_height // 2.2, anchor="center")
    # physiotherapy_button.bind("<Enter>", lambda e: play_hover_sound("./home/elderly/Desktop/elderlyactive-game/digital-beeping-151921.mp3"))

    quit_button = ttk.Button(root, text="✖", command=lambda: on_close(root), style="Quit.TButton")
    quit_button.place(relx=0.95, rely=0.05, anchor="center", width=30, height=25)
    # quit_button.bind("<Enter>", lambda e: play_hover_sound("./home/elderly/Desktop/elderlyactive-game/digital-beeping-151921.mp3"))

    print(f"Speech_flag_depois_do_recognize = {speech_recognition_enabled}")

    style = ttk.Style()
    style.configure("my.TButton", font=("Helvetica", 15))
    style.configure("Quit.TButton", font=("Arial", 10), background="red", foreground="red", padding=1)
    
    # # Load and place the blinking image with transparency
    # blinking_image = Image.open("SPEAK.png")  # Replace with your image path
    # blinking_image = blinking_image.convert("RGBA")  # Convert to RGBA mode
    # data = blinking_image.getdata()

    # new_data = []
    # for item in data:
    #     # Replace white pixels with transparent pixels
    #     if item[0] == 255 and item[1] == 255 and item[2] == 255:
    #         new_data.append((255, 255, 255, 0))  # Fully transparent
    #     else:
    #         new_data.append(item)

    # blinking_image.putdata(new_data)
    # blinking_image = blinking_image.resize((50, 50), Image.LANCZOS)  # Resize the image

    # blinking_photo = ImageTk.PhotoImage(blinking_image)
    # blinking_label = tk.Label(root, image=blinking_photo, bd=0)
    # blinking_label.place(x=screen_width // 2, y=screen_height // 1.2, anchor='center')
    
    # blink_image(blinking_label)

    root.mainloop()


if __name__ == "__main__":
    main()
