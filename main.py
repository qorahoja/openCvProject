#boshlandi 20/04/2023

#tugallandi 27/04/2023

#Yaratuvchi QorahojayevM



import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
glasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')


cap = cv2.VideoCapture(0)

while True:
    # kadrlarni olish
    ret, frame = cap.read()

    # kadrga gri ranga o'tkazib berish
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # yuzlarni aniqlash
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # har bir yuzni ajratish va chizish
    for (x, y, w, h) in faces:
        # yuzni ajratish
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Yuz', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


        # ko'zlar va burunni aniqlash
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)




        # yuz ichidagi ko'zoynaklarni aniqlash
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # har bir ko'zoynakni ajratish va chizish
        for (ex, ey, ew, eh) in glasses:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(frame, 'Ko\'z', (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),2)

        # yuz ichidagi tabassumni aniqlash
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)

        # har bir tabassumni ajratish va chizish
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            cv2.putText(roi_color, 'Tabassum', (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if len(faces) == 0:
        cv2.putText(frame, 'Yuz topilmadi', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # kadrlarni korsatish
    cv2.imshow('Face Detection', frame)

    # agar 'q' tugmasini bosilgan bo'lsa, dasturni to'xtatamiz
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# barcha ochiq konsol va oynalarni yopish
cap.release()
cv2.destroyAllWindows()