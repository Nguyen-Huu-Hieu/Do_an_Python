
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single


# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# start chương: dùng WPOD-NET phat hien bien so

# Đường dẫn ảnh
img_path = "test/test4.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
# anh goc
Ivehicle = cv2.imread(img_path)  

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

# cv2.imshow("hieu", LpImg[0])  anh da crop


# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')

if (len(LpImg)):
    # cvtColor : chuyển không gian màu
    cv2.imshow("Bien so duoc crop", cv2.cvtColor(LpImg[0],cv2.COLOR_RGB2BGR ))
    # cv2.waitKey()

    # end chương : dùng WPOD-NET phat hien bien so

    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

    roi = LpImg[0]
    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)

    # Ap dung threshold de phan tach so va nen, chuyen thanh anh nhi phan
    # nguong la 127
    # pixel trắng: 255, pixel đen : 0 , pixel nào > 127 --> chuyển về 255
    # THRESH_BINARY_INV : la loại ngưỡng
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]


    cv2.imshow("Anh nhi phan", binary)
    # cv2.waitKey()

    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # result biển số cuối cùng
    plate_info = ""

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio h/w
            if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                # Ve khung chu nhat quanh so
                # cv2.rectangle (hình ảnh, start_point, end_point, màu sắc, độ dày)
                # link : https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
    
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Tach so va predict
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                curr_num = np.array(curr_num,dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dua vao model SVM
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result<=9: # Neu la so thi hien thi luon
                    result = str(result)  # method str: đưa về dạng chuỗi
                else: #Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                plate_info +=result

    cv2.imshow("Cac contour tim duoc", roi)
    # cv2.waitKey() 

    # Viet bien so len anh
    # cv2.putText (hình ảnh, văn bản, vị trí, phông chữ, hệ số ..., màu , độ dày, lineType)
    cv2.putText(Ivehicle,fine_tune(plate_info),(10, 40), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)


    # Hien thi anh
    print("Bien so=", plate_info)
    cv2.imshow("Hinh anh output",Ivehicle)
    cv2.waitKey()


# đong cac cua so 
cv2.destroyAllWindows() 
