import cv2

img =cv2.imread('D:\me.jpg')


def imgDetector(img,cascade):
    
    # 영상 압축
    img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
    # 그레이 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # cascade 얼굴 탐지 알고리즘 
    results = cascade.detectMultiScale(gray,scaleFactor= 1.5,minNeighbors=5, minSize=(20,20))        
        
    for box in results:
            
        x, y, w, h = box
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
    
    # 사진 출력        
    cv2.imshow('facenet',img)  
    cv2.waitKey(10000)

cascade_filename = 'D:\haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

imgDetector(img,cascade)