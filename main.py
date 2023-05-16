import cv2
import mediapipe as mp
import pyautogui
from screen_brightness_control import get_brightness, set_brightness

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

initial_brightness = get_brightness()[0]
if initial_brightness < 40:
    initial_brightness = 40
elif initial_brightness > 60:
    initial_brightness = 50

target_brightness_increase = 100
target_brightness_decrease = 0

brightness_factor = 2.5

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (255, 255, 255)
brightness_text_position = (10, 30)
coordinates_text_position = (10, 60)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    output = hand_detector.process(frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            x0, y0 = None, None
            x5, y5 = None, None
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 0:
                    x0, y0 = x, y
                elif id == 5:
                    x5, y5 = x, y
            if x0 is not None and y0 is not None and x5 is not None and y5 is not None:
                distance = ((x5 - x0) ** 2 + (y5 - y0) ** 2) ** 0.5
                print()
                brightness_change = target_brightness_increase - target_brightness_decrease
                brightness_ratio = (distance - 50) / (frame_width - 50)
                brightness = initial_brightness + brightness_factor * brightness_change * brightness_ratio
                set_brightness(brightness)
                index_x = int(screen_width / frame_width * x5)
                index_y = int(screen_height / frame_height * y5)
                pyautogui.moveTo(index_x, index_y)
                brightness_text = f'Brightness: {get_brightness()[0]}'
                coordinates_text = f'Coordinates: ({index_x}, {index_y})'
                cv2.putText(frame, brightness_text, brightness_text_position, font, font_scale, font_color, 1)
                cv2.putText(frame, coordinates_text, coordinates_text_position, font, font_scale, font_color, 1)
    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) == 27:
        break

set_brightness(100)

cap.release()
cv2.destroyAllWindows()
