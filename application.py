import cv2
import numpy as np
import time
import PoseModule as pm
import argparse

def workoutFunc(workout):
    if workout in ["bicepCurls", "shoulderPress", "tricepPushdown", "cableRow", "latPulldown"]:
        p1=16
        p2=14
        p3=12
        p4=11
        p5=13
        p6=15
        if workout=="bicepCurls":
            lim1, lim2 = 210, 310
        elif workout=="shoulderPress":
            lim1, lim2 = 70, 130
        elif workout=="tricepPushdown":
            lim1, lim2 = 220, 280
        elif workout=="cableRow":
            p1,p2,p3,p4,p5,p6 = p4, p5, p6, p1, p2, p3
            lim1, lim2 = 210, 280
        elif workout=="latPulldown":
            lim1, lim2 = 210, 280
            
    elif workout in ["legPress", "squat"]:
        p1, p2, p3, p4, p5, p6 = 23, 25, 27, 24, 26, 28
        if workout == "legPress":
            lim1, lim2 = 100, 140
        elif workout == "squat":
            p1, p3=p3,p1
            lim1,lim2=195, 265
        
    return p1, p2, p3, p4, p5, p6, lim1, lim2

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='WORKOUT')

    parser.add_argument('--workout', type=str, required=True, help='workout')
    args = parser.parse_args()
    
    p1, p2, p3, p4, p5, p6, l1, l2 = workoutFunc(args.workout)
    cap = cv2.VideoCapture(f"{args.workout}.mp4")
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        if img is None:
            continue
        img = cv2.resize(img, (1280, 720))

        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            # first angle
            angle = detector.findAngle(img, p1, p2, p3)
            # second angle
            if args.workout=="squat":
                twoP = detector.twoPointComparison(img, 11, 26)
            # angle2 = detector.findAngle(img, p4, p5, p6)

            per = np.interp(angle, (l1, l2), (0, 100))
            bar = np.interp(angle, (l1, l2), (650, 100))
            if args.workout=="legPress" and angle>165:
                cv2.putText(img, "Warning!", (145, 145), cv2.FONT_HERSHEY_PLAIN, 5,
                        (0, 0, 255), 5)
            if args.workout=="squat":
                if twoP and angle>250:
                    cv2.putText(img, "Incorrect Form. Shoulders ahead of knees.", (245, 245), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 5)

            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)

            # Draw Curl Count
            # cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
