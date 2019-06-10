import cv2 as cv 
import numpy as np

def emap_generate(img):  #return type = uint64
    sohang = img.shape[0]
    socot = img.shape[1]
    b, g ,r = cv.split(img)
    #cv.imshow('b', b)
    #cv.imshow('g', g)
    #cv.imshow('r', r)
    bx = cv.Sobel(b,-1,0,1)
    by = cv.Sobel(b,-1,1,0)

    gx = cv.Sobel(g,-1,0,1)
    gy = cv.Sobel(g,-1,1,0)

    rx = cv.Sobel(r,-1,0,1)
    ry = cv.Sobel(r,-1,1,0)

    result = np.zeros((sohang, socot), dtype = np.uint64)
    #print("cho nay co loi")
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            result[i][j] = int(bx[i][j]) + int(by[i][j]) + int(gx[i][j]) + int(gy[i][j]) + int(rx[i][j]) + int(ry[i][j])
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

def all_seam_generate(emap, scale):
    sohang = emap.shape[0]
    socot = emap.shape[1]
    seam_lines = np.zeros((scale, sohang), dtype = np.uint64)

    etable = np.ones((sohang, socot), dtype = np.uint64)
    etable[0] = emap[0]
    for i in range(1, sohang):
        for j in range(0, socot):
            if j == 0:
                etable[i][j] = emap[i][j] + min([ etable[i-1][j], etable[i-1][j+1] ])
            elif j == socot - 1:
                etable[i][j] = emap[i][j] + min([ etable[i-1][j], etable[i-1][j-1] ])
            else:
                etable[i][j] = emap[i][j] + min([ etable[i-1][j], etable[i-1][j-1], etable[i-1][j+1] ])

    for i in range(scale):
        seam_line = np.ones((sohang), dtype = np.uint64)
        y = 0
        x = sohang - 1
        mi = etable[x][y]
        for j in range(1, socot):
            if etable[x][j] < mi:
                mi = etable[x][j]
                y = j
        seam_line[x] = y

        for j in reversed(range(0,sohang-1)):
            if y == 0:
                if etable[j][y+1] < etable[j][y]:
                    y = y + 1
            elif y == socot-1:
                if etable[j][y-1] < etable[j][y]:
                    y = y - 1
            else:
                cong = etable[j][y+1]
                tru = etable[j][y-1]
                bang = etable[j][y]
                if cong < tru and cong < bang:
                    y = y + 1
                elif tru < cong and tru < bang:
                    y = y - 1
            seam_line[j] = y    


        for j in range(sohang):
            etable[j][int(seam_line[j])] = 999999999999999999
        #print ('tim duoc ', i, ' duong')
        seam_lines[i] = seam_line
    return seam_lines



def seam_delete(seam, img):
    a = img.shape[0]
    b = img.shape[1]
    for i in range(a):
        for j in range(int(seam[i]),b-1):
            img[i,j]=img[i,j+1]
    return img[:,0:b-1]

def all_seam_delete(seam_lines, img, scale):
    sohang = img.shape[0]
    socot = img.shape[1]
    result = np.zeros((sohang, socot, 3), dtype = np.uint8)
    
    jj = 0
    for i in range(sohang):
        jj = 0
        for j in range(socot):
            #print(type(seam_lines[:,i]))          
            if not(j in seam_lines[:,i]):
                result[i][jj] = img[i][j]
                jj+=1
                #fprint(i,' ',j)
        #print('xong ', i, ' hang')

    tmp = img.copy()
    for i in range(len(seam_lines)):
        for j in range(sohang):
            tmp[j][seam_lines[i][j]] = [255,0,0]
    
    cv.imshow('seamline', tmp)
    return result[:,0:socot-scale]
            

        

    
def seam_carving(img, scale):
    for i in range(scale):
        #tmp = img.copy()
        emap = emap_generate(img)
        s = seam_line_generate(emap)
        for b in range(img.shape[0]):
            img[b][int(s[b])] = [0,0,255]
        cv.imshow("seamline",img)
        #cv.waitKey(0)
        img = seam_delete(s,img)
        #cv.imshow('ahaha', img)
        cv.waitKey(1)

def all_seam_carving(img, scale): #tim tat ca duong seam va xoa 1 luc
    #print('tim tat ca duong seam va xoa 1 luc')
    emap = emap_generate(img)
    seam_lines = all_seam_generate(emap,scale)
    result = all_seam_delete(seam_lines, img, scale)
    cv.imshow('result', result)
    return seam_lines,result

img = cv.imread('seam.jpg')
#grayimg = cv.imread('seam.jpg',0)

cv.imshow('img', img)
#cv.imshow('grayimg', grayimg)
it=img.copy()

t,it=all_seam_carving(it,50)
#cv.imwrite('aaa.jpg',it)

cv.waitKey(0)
cv.destroyAllWindows()