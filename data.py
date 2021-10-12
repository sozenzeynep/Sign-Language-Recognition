import cv2
import numpy as np
import os

kernel = np.ones((5, 5), np.uint8)

my_directory = os.listdir('C:\\Users\zeyne\OneDrive\Masaüstü\İşaret Dili Tanıma ve Yazdırma Projesi\datasetler\ASL-Finger-Spelling-Recognition-master\ASL-Finger-Spelling-Recognition-master\asl_dataset')

def filtre(path,path2):
    while True:
        o = cv2.imread(path)
        p = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
        pHSV = cv2.cvtColor(o, cv2.COLOR_BGR2HSV)

        AltDegerler = np.array([0, 48, 80])
        UstDegerler = np.array([90, 255, 255])

        FiltreSonucu = cv2.inRange(pHSV, AltDegerler, UstDegerler)
        FiltreSonucu = cv2.morphologyEx(FiltreSonucu, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("Veri/"+path2, FiltreSonucu)
        cv2.imshow("Sonuc", FiltreSonucu)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for file in my_directory:
    my_new_directory = os.listdir('C:\\Users\\zeyne\OneDrive\Masaüstü\\İşaret Dili Tanıma ve Yazdırma Projesi\datasetler\\asl-alphabet\\asl_alphabet_train\\asl_alphabet_train\\'+ file)
    for new_file in my_new_directory:
        filtre('C:\\Users\\zeyne\OneDrive\Masaüstü\\İşaret Dili Tanıma ve Yazdırma Projesi\datasetler\\asl-alphabet\\asl_alphabet_train\\asl_alphabet_train\\'+file+'\\'+new_file, new_file)

cv2.destroyAllWindows()
