import cv2
from tensorflow.keras.models import load_model
model=load_model("models/best_model.h5")
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
face_clip=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

if ret:
    print("ret true")
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces=face_clip.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        print("hi")
        face_roi=gray[y:y+h,x:x+w]
        face_resized=cv2.resize(face_roi,(48,48))
        face_nor=face_resized/255.0
        face_input=face_nor.reshape(1,48,48,1)
        prediction=model.predict(face_input)
        print(prediction)
        pred=np.arg
        # cv2.imshow("web",face_roi)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # else:
        #     print("failed")
cap.release()
