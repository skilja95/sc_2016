import cv2
from scipy import ndimage
import time
import math
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
from skimage import color
import numpy as np
import Tkinter as tk
import tkFileDialog as filedialog
from sklearn.datasets import fetch_mldata

root = tk.Tk()
root.withdraw()
print("Please choose video for analysing. File chooser is opened.")
videoName = filedialog.askopenfilename(filetypes=[("Video Files","*.avi;*.mp4")])
cap = cv2.VideoCapture(videoName)
mnist = fetch_mldata('MNIST original')
cap1 = cv2.VideoCapture(videoName) # da bih mogao da unistim prozor nakog uzimanja prs
i=0
gray="grayFrame"
frame1="frame"
if i==0:
    i=i+1
    while(cap1.isOpened()):
        print("Video is opened.")
        ret, frame = cap1.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame1=frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            break

    print("Showing the video.")
    cap1.release()
    cv2.destroyAllWindows()


frame = frame1
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,30)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
    y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
    x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
    y2 = int(y0 - 1000*(a))
    #cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
lines = cv2.HoughLinesP(edges,1,np.pi/180,30, minLineLength = 200, maxLineGap = 10)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(frame,(x1,y1),(x2,y2),(255,255,0),1)
cv2.imwrite('FirstFrameLine.jpg',frame)
line = [(x1, y1), (x2, y2)]
cc = -1
sum = 0

def nextId():
    global cc
    cc += 1
    return cc


new_mnist_set=[]
def transformMnist(mnist):

    i=0;
    while i < 70000:
        mnist_img=mnist.data[i].reshape(28,28)
        mnist_img_BW=((color.rgb2gray(mnist_img)/255.0)>0.88).astype('uint8')
        l = label(mnist_img_BW.reshape(28,28))
        r = regionprops(l)
        min_x = r[0].bbox[0]
        min_y = r[0].bbox[1]
    
        for j in range(1,len(r)):
            if(r[j].bbox[0]<min_x):
                min_x = r[j].bbox[0]
            if(r[j].bbox[1]<min_y):
                min_y = r[j].bbox[1]
        img = np.zeros((28,28))
        img[:(28-min_x),:(28-min_y)] = mnist_img_BW[min_x:,min_y:]
        new_mnist_img = img
        new_mnist_set.append(new_mnist_img)
        i=i+1
        
def move(image):
    l = label(image.reshape(28,28))
    r = regionprops(l)
    min_x = r[0].bbox[0]
    min_y = r[0].bbox[1]

    for j in range(1,len(r)):
        if(r[j].bbox[0]<min_x):
            min_x = r[j].bbox[0]
        if(r[j].bbox[1]<min_y):
            min_y = r[j].bbox[1]
    img = np.zeros((28,28))
    img[:(28-min_x),:(28-min_y)] = image[min_x:,min_y:]

    return img

def main():
    print("Transform mnist started. Please wait...")
    transformMnist(mnist)
    print("Mnist transformated successfuly.")
    kernel = np.ones((2,2),np.uint8)
    boundaries = [
        ([230, 230, 230], [255, 255, 255])
    ]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))

    elements = []
    t =0
    counter = 0
    times = []

    while (1):
        start_time = time.time()
        ret, img = cap.read()
        if not ret:
            break
        (lower, upper) = boundaries[0]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)
        img0 = cv2.dilate(img0, kernel)
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if (dxc > 11 or dyc > 11):
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                
                r = 20
                item = elem
                items = elements
                retVal = []
                for obj in items:
                    x,y = item['center']
                    X,Y = obj['center']
                    a,b = X-x, Y-y
                    vrati = math.sqrt(a*a + b*b)
                    mdist = vrati
                    if(mdist<r):
                        retVal.append(obj)
                        
                
                lst = retVal
                nn = len(lst)
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['pass'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []
                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                pnt = el['center']
                start = line[0]
                end = line[1]

                x,y = start
                X,Y = end
                line_vec = (X-x, Y-y)
                X,Y = pnt
                pnt_vec = (X-x, Y-y)
                a,b = line_vec
                line_len = math.sqrt(a*a + b*b)
                x,y = line_vec
                mag = math.sqrt(x*x + y*y)               
                line_unitvec = (x/mag, y/mag)
                x,y = pnt_vec
                pnt_vec_scaled = (x * 1.0/line_len, y * 1.0/line_len)
                x,y = line_unitvec
                X,Y = pnt_vec_scaled
                t = x*X + y*Y   
                r = 1
                if t < 0.0:
                    t = 0.0
                    r = -1
                elif t > 1.0:
                    t = 1.0
                    r = -1
                x,y = line_vec
                nearest = (x * t, y * t)
                
                x,y = nearest
                X,Y = pnt_vec
                asi,asi1 = X-x, Y-y
                duz = math.sqrt(asi*asi + asi1*asi1)
                dist = duz
                nearest = (x+X, y+Y)
                pnt = (int(nearest[0]), int(nearest[1]))
                if r > 0:
                    c = (25, 25, 255)
                    if (dist < 9):
                        c = (0, 255, 160)
                        if el['pass'] == False:
                            el['pass'] = True
                            counter += 1
                            (x,y)=el['center']
                            xLijevo=x-14
                            xDesno=x+14
                            (sx,sy)=el['size']
                            yDole=y-14
                            yGore=y+14
                            slika = img[yDole:yGore,xLijevo:xDesno]
                            global sum
                            img_BW=color.rgb2gray(slika) >= 0.88
                            img_BW=(img_BW).astype('uint8')
                            plt.imshow(img_BW,'gray')
                            plt.show()
                            
                            newImg = move(img_BW)
                            
                            
                            
                            plt.imshow(newImg, 'gray')
                            plt.show()
                            i=0;
                            
                            i=0;
                            ret = 0
                            while i<70000:
                                asd=0
                                mnist_img=new_mnist_set[i]
                                asd=np.sum(mnist_img!=newImg)
                                if asd<20:
                                    ret = mnist.target[i]
                                    break
                                i=i+1
        
                            rez = ret
                            print("Broj je prepoznat kao: " + format(rez))
                            sum += rez



        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Sum: ' + str(sum), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
        # print nr_objects
        t += 1
        if t % 10 == 0:
            print t
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        out.write(img)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    et = np.array(times)
    print("Video koji je ucitan je: " + videoName)
    print("Rezultat je: " + format(sum))
    print 'mean %.2f ms' % (np.mean(et))

main()