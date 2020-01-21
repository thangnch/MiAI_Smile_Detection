import cv2

def load_model():
    cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
    cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
    return cascade_face, cascade_eye,cascade_smile

def draw_smile(img, center, radius):
    axes = (radius, radius//2)
    color = (255,0,0)
    cv2.ellipse(img, center, axes, 0, 0, 180, color,thickness=5,lineType=cv2.LINE_AA)
    return

def draw_eye(img, center,radius):
    axes = (radius, radius // 2)
    color = (0, 0, 255)
    cv2.ellipse(img, center, axes, 0, 180, 360, color, thickness=5, lineType=cv2.LINE_AA)

def detection(grayscale, img, face_detect, eye_detect, smile_detect):
    # Detect face trong anh
    face = face_detect.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:

        # Tach ROI
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face]

        # Detect cac smile trong ROI
        smile_lst = smile_detect.detectMultiScale(ri_grayscale, 1.7, 35)
        smiled = False
        for (x_smile, y_smile, w_smile, h_smile) in smile_lst:
            # Ve mat cuoi
            draw_smile(ri_color, (x_smile + w_smile // 2, y_smile + h_smile // 4), radius=w_smile // 3)
            smiled =  True

        # Neu co smile thi
        if smiled:
            eye = eye_detect.detectMultiScale(ri_grayscale, 1.3, 9)
            for (x_eye, y_eye, w_eye, h_eye) in eye:
                    draw_eye(ri_color,(x_eye+w_eye//2, y_eye+h_eye//2),radius=w_eye//3)
            cv2.putText(img,"HAPPY NEW YEAR!!",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),5)

    return img

face_detect, eye_detect, smile_detect = load_model()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    frame = cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = detection(gray, frame,face_detect, eye_detect, smile_detect)

    cv2.imshow('WC', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()