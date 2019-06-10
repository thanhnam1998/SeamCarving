import cv2 as cv 
import numpy as np
import sys
import time

path = 'seam.jpg'
pic_name = 'check_img'
#scale_ver=50
#scale_hor=50
scale=50

check_img = cv.imread(path)
check_window_name = pic_name
cv.namedWindow(check_window_name)

origin_img = cv.imread(path)
emap_mask_ver = np.zeros((origin_img.shape[0], origin_img.shape[1]), dtype = np.uint8)
#emap_mask_hor = np.zeros((origin_img.shape[1], origin_img.shape[0]), dtype = np.uint8) 


drawing = False
def mouse_event (event, x, y, flags, param):
    global drawing, emap_mask_ver

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            for i in range(y-5,y+5):
                for j in range(x-5,x+5):
                    check_img[i][j] = [0, 255, 0] 
                    emap_mask_ver[i][j] = 1
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        for i in range(y-5,y+5):
            for j in range(x-5,x+5):
                check_img[i][j] = [0, 255, 0]
                emap_mask_ver[i][j] = 1

cv.setMouseCallback(check_window_name,mouse_event)

while (1):
    cv.imshow(check_window_name,check_img)
    k = cv.waitKey(1) 
    if k == 27:  # bam nut ESC thi dung lai ### co the thay the bang nut khac bang cach thay doi so 27 neu muon
        break
emap_mask_hor=np.rot90(emap_mask_ver,3,(0,1))


def emap_generate_gray(img):
    sohang,socot=img.shape[0:2]
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sobel_x=cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
    sobel_y=cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)
    #cv.imshow('sobel_x',sobel_x)
    #cv.imshow('sobel_y',sobel_y)
    abs_sobelx=cv.convertScaleAbs(sobel_x)
    abs_sobely=cv.convertScaleAbs(sobel_y)
    #cv.imshow('absx',abs_sobelx)
    #cv.imshow('absy',abs_sobely)
    result=cv.addWeighted(abs_sobelx,0.5,abs_sobely,0.5,0)
    #cv.imshow('absx+absy',result)
    #cv.waitKey()
    if(sohang==emap_mask_ver.shape[0]):
        for i in range(sohang):
            for j in range(socot):
                if emap_mask_ver[i][j] == 1:
                    result[i][j] = 255
    else:
        for i in range(sohang):
            for j in range(socot):
                if emap_mask_hor[i][j] == 1:
                    result[i][j] = 255
    return result

def emap_generate_RGB(img):  #return type = uint64
    global emap_mask_ver,emap_mask_hor

    sohang = img.shape[0]
    socot = img.shape[1]
    b, g ,r = cv.split(img)

    bx = cv.Sobel(b,-1,0,1)
    by = cv.Sobel(b,-1,1,0)

    gx = cv.Sobel(g,-1,0,1)
    gy = cv.Sobel(g,-1,1,0)

    rx = cv.Sobel(r,-1,0,1)
    ry = cv.Sobel(r,-1,1,0)

    #cv.imshow('sobel_blue_x',bx)
    #cv.imshow('sobel_blue_y',by)
    #cv.imshow('sobel_green_x',gx)
    #cv.imshow('sobel_green_y',gy)
    #cv.imshow('sobel_red_x',rx)
    #cv.imshow('sobel_red_y',ry)
    result = np.zeros((sohang, socot), dtype = np.uint64)
    if(sohang==emap_mask_ver.shape[0]):
        for i in range(0, sohang):
            for j in range(0, socot):
                if emap_mask_ver[i][j] == 0:
                    result[i][j] = int(bx[i][j]) + int(by[i][j]) + int(gx[i][j]) + int(gy[i][j]) + int(rx[i][j]) + int(ry[i][j])
                else:
                    result[i][j] = 100000000
    else:
        for i in range(0, sohang):
            for j in range(0, socot):
                if emap_mask_hor[i][j] == 0:
                    result[i][j] = int(bx[i][j]) + int(by[i][j]) + int(gx[i][j]) + int(gy[i][j]) + int(rx[i][j]) + int(ry[i][j])
                else:
                    result[i][j] = 100000000
    #cv.imshow('emap_generate_RGB',result)
    #cv.waitKey()
    return result

def seam_line_generate(emap):
    sohang = emap.shape[0]
    socot = emap.shape[1]
    seam_line = np.ones((sohang), dtype = np.uint64)

    etable = np.ones((sohang, socot), dtype=np.uint64)
    etable[0] = emap[0]
    for i in range(1, sohang):
        for j in range(0, socot):
            if j == 0:
                etable[i][j] = emap[i][j] + min([ etable[i-1][j], etable[i-1][j+1] ])
            elif j == socot - 1:
                etable[i][j] = emap[i][j] + min([ etable[i-1][j], etable[i-1][j-1] ])
            else:
                etable[i][j] = emap[i][j] + min([ etable[i-1][j], etable[i-1][j-1], etable[i-1][j+1] ])
    y = 0
    x = sohang - 1
    mi = etable[x][y]
    for i in range(1, socot):
        if etable[x][i] < mi:
            mi = etable[x][i]
            y = i
    seam_line[x] = y

    for i in reversed(range(0,sohang-1)):
        if y == 0:
            if etable[i][y+1] < etable[i][y]:
                y = y + 1
        elif y == socot-1:
            if etable[i][y-1] < etable[i][y]:
                y = y - 1
        else:
            cong = etable[i][y+1]
            tru = etable[i][y-1]
            bang = etable[i][y]
            if cong < tru and cong < bang:
                y = y + 1
            elif tru < cong and tru < bang:
                y = y - 1
        seam_line[i] = y    
    return seam_line

def seam_delete(seam, img):
    global emap_mask_ver,emap_mask_hor
    sohang = img.shape[0]
    socot = img.shape[1]
    if(sohang==emap_mask_ver.shape[0]):
        for i in range(sohang):
            for j in range(int(seam[i]), socot-1):
                img[i,j] = img[i, j+1]
                emap_mask_ver[i,j] = emap_mask_ver[i, j+1]
    else:
        for i in range(sohang):
            for j in range(int(seam[i]), socot-1):
                img[i,j] = img[i, j+1]
                emap_mask_hor[i,j] = emap_mask_hor[i, j+1]
    return img[:, 0:socot-1]

def seam_add(img,seam):
    sohang=img.shape[0]
    socot=img.shape[1]
    zeros=np.zeros((sohang,1,3),dtype=np.uint8)
    img_ext=np.hstack((img,zeros))
    for i in range(sohang):
        #print(i,' ',int(seam[i]))
        #print(img_ext.shape[:2],socot)
        for j in range(socot,int(seam[i]),-1):
            img_ext[i,j]=img[i,j-1]
        for k in range(3):
            if(int(seam[i])==0):
                v1=img_ext[i,int(seam[i])+2,k]
            else:
                v1=img_ext[i,int(seam[i])-1,k]
            v2=img_ext[i,int(seam[i])+1,k]
            img_ext[i,int(seam[i]),k]=(int(v1)+int(v2))/2
            #print('a')
    return img_ext 

def seam_carving_gray(img, scale):
    tmp = img.copy()
    for i in range(scale):
        emap=emap_generate_gray(img)
        s = seam_line_generate(emap)
        for b in range(img.shape[0]):
            img[b][int(s[b])] = [0,0,255]
        cv.imshow("seamlinegray",img)
        img = seam_delete(s,img)
        #tmp=seam_add(tmp,s)
        #cv.imshow('tmp', tmp)
        cv.waitKey(1)
    return img#,tmp

def seam_carving_RGB(img, scale):
    tmp = img.copy()
    for i in range(scale):
        emap = emap_generate_RGB(img)
        s = seam_line_generate(emap)
        for b in range(img.shape[0]):
            img[b][int(s[b])] = [0,0,255]
        #cv.imshow("seamlineRGB",img)
        img = seam_delete(s,img)
        #tmp=seam_add(tmp,s)
        #cv.imshow('tmp', tmp)
        cv.waitKey(1)
    return img#,tmp

img1=cv.imread(path)
img2=cv.imread(path)
img3=cv.imread(path)
img4=cv.imread(path)
start_gray=time.time()
img_gray_ver=seam_carving_gray(img1,scale)
end_gray=time.time()
print(end_gray-start_gray)
img3 = np.rot90(img_gray_ver, 3, (0,1)) #xoay ảnh 90 độ để tìm đường seam ngang
start_gray=time.time()
img_gray_hor=seam_carving_gray(img3,scale)
img_gray_hor=np.rot90(img_gray_hor,1,(0,1)) #xoay ảnh lại
end_gray=time.time()
print(end_gray-start_gray)


"""
start_RGB=time.time()
img_RGB_ver=seam_carving_RGB(img2,scale_ver)
end_RGB=time.time()
print(end_RGB-start_RGB)
img4 = np.rot90(img_RGB_ver, 1, (0,1)) #xoay ảnh 90 độ để tìm đường seam ngang
start_RGB=time.time()
img_RGB_hor=seam_carving_RGB(img4,scale_hor)
img_RGB_hor=np.rot90(img_RGB_hor,3,(0,1)) #xoay ảnh lại
end_RGB=time.time()
print(end_RGB-start_RGB)
"""

#cv.imshow('gray_ver',img_gray_ver)
cv.imshow('gray_result_'+str(scale),img_gray_hor)
#cv.imwrite('aaa.jpg',img_gray_hor)
#cv.imshow('RGB_ver',img_RGB_ver)
#cv.imshow('RGB_result_'+str(scale),img_RGB_hor)
cv.waitKey()
cv.destroyAllWindows()