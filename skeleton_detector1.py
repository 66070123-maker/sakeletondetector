import cv2
import mediapipe as mp
import time

# 1. การตั้งค่า MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 2. การเริ่มต้นกล้องเว็บแคม
# ลองเปลี่ยนเลข 0 เป็น 1 หรือ 2 หากกล้องหลักของคุณไม่ทำงาน
cap = cv2.VideoCapture(2)

# *** เพิ่มการตรวจสอบการเปิดกล้อง ***
if not cap.isOpened():
    print("❌ ข้อผิดพลาด: ไม่สามารถเปิดกล้องได้ ตรวจสอบว่ากล้องไม่ได้ถูกใช้งานโดยโปรแกรมอื่น")
    exit()

# เริ่มต้นโมเดล Pose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    print("--- ระบบเริ่มต้นแล้ว (กด 'q' เพื่อออก) ---")
    
    # ลูปหลัก: อ่านภาพจากกล้องทีละเฟรม
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ไม่สามารถรับเฟรมจากกล้องได้")
            break

        # 1. การประมวลผลภาพสำหรับ MediaPipe (BGR -> RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False 
        
        results = pose.process(image) # ประมวลผลภาพเพื่อหาจุด (Landmarks)
        
        # 2. การแสดงผล (RGB -> BGR)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # วาดผลลัพธ์ (โครงกระดูก)
        if results.pose_landmarks:
            # วาดจุด (landmarks) และเส้นเชื่อมโยง (connections)
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                # สไตล์สำหรับจุด (สีฟ้า)
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                # สไตล์สำหรับเส้นเชื่อมโยง (สีเขียว)
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            # โมเดลจะตรวจจับ 33 จุดทั่วร่างกาย
            

        # แสดงภาพในหน้าต่าง
        cv2.imshow('MediaPipe Pose Skeleton Detector', image)
        
        # กด 'q' เพื่อออกจากลูป
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 3. การปิดระบบ
cap.release()
cv2.destroyAllWindows()
print("--- ปิดโปรแกรมแล้ว ---")
