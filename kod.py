import cv2
import numpy as np
import os

kamera = cv2.VideoCapture(0)
kernel = np.ones((4,4),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,30)
fontScale = 1
fontColor = (0,0,255)
lineType = 2

def ResimFarkBul(Resim1,Resim2):
    Resim2 = cv2.resize(Resim2, (Resim1.shape[1], Resim1.shape[0]))
    Fark_Resim= cv2.absdiff(Resim1, Resim2)
    Fark_Sayi = cv2.countNonZero(Fark_Resim)
    return Fark_Sayi

def VeriYükle():
    Veri_İsimler = []
    Veri_Resimler = []

    Dosyalar= os.listdir("Veri/")
    for Dosya in Dosyalar:
        Veri_İsimler.append(Dosya.replace(".jpg", ""))
        Veri_Resimler.append(cv2.imread("Veri/"+Dosya, 0))

    return Veri_İsimler,Veri_Resimler

def Sınıflandır(Resim, Veri_isimler, Veri_Resimler): #Hangisinin farklılık değeri az ise ona ait olur
    min_index = 0
    min_değer = ResimFarkBul(Resim, Veri_Resimler[0])
    for t in range(len(Veri_isimler)):
        Fark_Değer= ResimFarkBul(Resim, Veri_Resimler[t])
        if(Fark_Değer < min_değer):
            min_değer = Fark_Değer
            min_index = t
            #print(min_değer)
    return Veri_isimler[min_index]

Veri_isimler, Veri_Resimler = VeriYükle()
print(Veri_isimler)

while True:
    ret, kare = kamera.read()
    Kesilmis_Kare = kare[70:250, 70:250]
    Kesilmis_Kare_HSV = cv2.cvtColor(Kesilmis_Kare,cv2.COLOR_BGR2HSV)
#HSV renk uzayı = Hue, Saturation ve Value terimleri ile rengi tanımlar

    alt_degerler= np.array([0,48,80])
    üst_degerler = np.array([90,255,255])

    Renk_filtresi_sonucu = cv2.inRange(Kesilmis_Kare_HSV,alt_degerler,üst_degerler)
    Renk_filtresi_sonucu = cv2.morphologyEx(Renk_filtresi_sonucu, cv2.MORPH_CLOSE, kernel)
    Renk_filtresi_sonucu = cv2.morphologyEx(Renk_filtresi_sonucu, cv2.MORPH_OPEN, kernel2)

    cnts, hierarchy =cv2.findContours(Renk_filtresi_sonucu,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_genislik = 0
    max_uzunluk = 0
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

        yazi = Sınıflandır(El_resim, Veri_isimler, Veri_Resimler)
        yazi = yazi.replace("0", " ")
        yazi = yazi.replace("1", " ")
        yazi = yazi.replace("2", " ")
        yazi = yazi.replace("3", " ")
        yazi = yazi.replace("4", " ")
        yazi = yazi.replace("5", " ")
        yazi = yazi.replace("6", " ")
        yazi = yazi.replace("7", " ")
        yazi = yazi.replace("8", " ")
        yazi = yazi.replace("9", " ")
        cv2.putText(Kesilmis_Kare, yazi, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    cv2.imshow("Kare", kare)
    cv2.imshow("Renk Filtresi",Renk_filtresi_sonucu)
    cv2.imshow("Sonuc",Kesilmis_Kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()

