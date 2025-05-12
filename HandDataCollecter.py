import cv2
import mediapipe as mp
import math
import pickle
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

Position_Layers = [
    [0],
    [1, 5, 9, 13, 17],
    [2, 6, 10, 14, 18],
    [3, 7, 11, 15, 19],
    [4, 8, 12, 16, 20],
]


def GetPositionLayer(HandIndex):
    for layernum, line in enumerate(Position_Layers):
        if HandIndex in line:
            return {
                'layer': layernum,
                'index': line.index(HandIndex)
            }


def GetPointsDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def GetRelativeDistance(StandardLength, p1, p2):
    PointDistance = GetPointsDistance(p1, p2)
    return PointDistance / StandardLength


def ImageToDistanceData(image, hands):
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    Frame_Layer_Data = [
        [],
        [],
        [],
        [],
    ]

    Hand_Frame_Data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for index, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                z = landmark.y

                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                Hand_Frame_Data.append([relative_x, relative_y])

            break

    return {
        'Hand-Landmarks': Hand_Frame_Data,
        'image': image
    }


if __name__ == '__main__':
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            ImageData = ImageToDistanceData(image, hands)
            Hand_Landmarks = ImageData['Hand-Landmarks']
            image = ImageData['image']

            if Hand_Landmarks:
                for point in Hand_Landmarks:
                    cv2.circle(image, (point[0], point[1]), 5, (0, 255, 0), -1)

            cv2.imshow("MediaPipe Hands", image)
            if cv2.waitKey(5) & 0xFF == 27:  # press escape to break
                break

    cap.release()
    cv2.destroyAllWindows()