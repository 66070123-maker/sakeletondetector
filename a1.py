import cv2
import mediapipe as mp
import math

# ------------------- ฟังก์ชันคำนวณมุม -------------------
def calculate_angle(a, b, c):
    """ a,b,c = (x, y) """
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    angle_rad = math.acos(dot / (mag_ba * mag_bc + 1e-7))
    return math.degrees(angle_rad)

# ------------------- ตั้งค่า MediaPipe -------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(1)  # กล้องด้านข้าง

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถเปิดกล้องได้")
            break

        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = pose.process(img_rgb)
        img_rgb.flags.writeable = True
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # ดึงพิกัดที่จำเป็น
            def get_point(idx):
                return [landmarks[idx].x, landmarks[idx].y]

            # ------------------- คำนวณมุม -------------------
            knee_angle = calculate_angle(get_point(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                         get_point(mp_pose.PoseLandmark.KNEE_RIGHT.value),
                                         get_point(mp_pose.PoseLandmark.ANKLE_RIGHT.value))

            hip_angle = calculate_angle(get_point(mp_pose.PoseLandmark.SHOULDER_RIGHT.value),
                                        get_point(mp_pose.PoseLandmark.HIP_RIGHT.value),
                                        get_point(mp_pose.PoseLandmark.KNEE_RIGHT.value))

            elbow_angle = calculate_angle(get_point(mp_pose.PoseLandmark.SHOULDER_RIGHT.value),
                                          get_point(mp_pose.PoseLandmark.ELBOW_RIGHT.value),
                                          get_point(mp_pose.PoseLandmark.WRIST_RIGHT.value))

            wrist_angle = calculate_angle(get_point(mp_pose.PoseLandmark.ELBOW_RIGHT.value),
                                          get_point(mp_pose.PoseLandmark.WRIST_RIGHT.value),
                                          get_point(mp_pose.PoseLandmark.WRIST_RIGHT.value))  # ใช้เป็น reference

            shoulder_angle = calculate_angle(get_point(mp_pose.PoseLandmark.ELBOW_RIGHT.value),
                                            get_point(mp_pose.PoseLandmark.SHOULDER_RIGHT.value),
                                            get_point(mp_pose.PoseLandmark.HIP_RIGHT.value))

            neck_angle = calculate_angle(get_point(mp_pose.PoseLandmark.NOSE.value),
                                         get_point(mp_pose.PoseLandmark.SHOULDER_RIGHT.value),
                                         get_point(mp_pose.PoseLandmark.HIP_RIGHT.value))

            torso_angle = calculate_angle(get_point(mp_pose.PoseLandmark.SHOULDER_RIGHT.value),
                                          get_point(mp_pose.PoseLandmark.HIP_RIGHT.value),
                                          get_point(mp_pose.PoseLandmark.KNEE_RIGHT.value))

            # ------------------- ตรวจเกณฑ์ -------------------
            status = []

            if 90 <= knee_angle <= 100:
                status.append("Knee OK")
            else:
                status.append("Knee Bad")

            if 90 <= hip_angle <= 100:
                status.append("Hip OK")
            else:
                status.append("Hip Bad")

            if 90 <= elbow_angle <= 100:
                status.append("Elbow OK")
            else:
                status.append("Elbow Bad")

            if 9 <= wrist_angle <= 10 :  # ข้อมือ ±10
                status.append("Wrist OK")
            else:
                status.append("Wrist Bad")

            if neck_angle <= 15:
                status.append("Neck OK")
            else:
                status.append("Neck Bad")

            if shoulder_angle <= 20:
                status.append("Shoulder OK")
            else:
                status.append("Shoulder Bad")

            if 100 <= torso_angle <= 110:
                status.append("Torso OK")
            else:
                status.append("Torso Bad")

            # วาด skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # แสดงผลบนหน้าจอ
            y0 = 30
            for i, s in enumerate(status):
                cv2.putText(frame, s, (10, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if "OK" in s else (0,0,255), 2)

        # แสดงภาพ
        cv2.imshow("Ergonomics Pose Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
