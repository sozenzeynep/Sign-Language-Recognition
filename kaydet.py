import cv2
import numpy as np

kamera = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

isim = "Dort5"

while True:
    ret, kare = kamera.read() #Frame alma
    Kesilmis_Kare = kare[70:250,70:250]
    Kes_Gri = cv2.cvtColor(Kesilmis_Kare,cv2.COLOR_BGR2GRAY)
    Kesilmis_Kare_HSV = cv2.cvtColor(Kesilmis_Kare,cv2.COLOR_BGR2HSV)

    alt_degerler= np.array([0,48,80])
    üst_degerler = np.array([90,255,255])

    Renk_filtresi_sonucu = cv2.inRange(Kesilmis_Kare_HSV,alt_degerler,üst_degerler)
    Renk_filtresi_sonucu = cv2.morphologyEx(Renk_filtresi_sonucu, cv2.MORPH_CLOSE, kernel)

    cnts, hierarchy =cv2.findContours(Renk_filtresi_sonucu,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_genislik = 0
    max_uzunluk =0
    max_ındex= -1

    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)
        if(w>max_genislik and h>max_uzunluk):
            max_uzunluk = h
            max_genislik= w
            max_ındex = t

    if(len(cnts)> 0):
        x,y,w,h = cv2.boundingRect(cnts[max_ındex])
        cv2.rectangle(Kesilmis_Kare,(x,y),(x+w,y+h),(0,255,0),2)
        El_resim = Renk_filtresi_sonucu[y:y+h,x:x+w]
        cv2.imshow("El resim",El_resim)

    cv2.imshow("Kare", kare)
    cv2.imshow("Kesilmiş Kare",Kesilmis_Kare)
    cv2.imshow("Renk Filtresi",Renk_filtresi_sonucu)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite("Veri/"+ isim+".jpg",El_resim)

kamera.release()
cv2.destroyAllWindows()

