import numpy as np
import scipy as sc
import cv2
import math
import matplotlib.pyplot as plt
from random import sample
import operator
import os
from PIL import Image
from scipy import optimize
import sys

def findhomography(matchingpoints):
    homoandinliers = []
    numinliers=0
    for pp in range(420):
        samp = sample(matchingpoints,4) # list of lists of lists
        A=[]
        for n,v in enumerate(samp):
            A.append([-samp[n][0][0], -samp[n][0][1], -1, 0, 0, 0, samp[n][1][0]*samp[n][0][0], samp[n][1][0]*samp[n][0][1], samp[n][1][0]])
            A.append([0, 0, 0, -samp[n][0][0], -samp[n][0][1], -1, samp[n][1][1]*samp[n][0][0], samp[n][1][1]*samp[n][0][1], samp[n][1][1]])
        u, s, vh = np.linalg.svd(np.asarray(A), full_matrices=True)
        h = vh[-1]
        
        gmatch = []
        for jj in matchingpoints:
            x2 = -jj[0][0]*h[0] -jj[0][1]*h[1] -1*h[2]+ 0+ 0+ 0+ jj[1][0]*jj[0][0]*h[6]+ jj[1][0]*jj[0][1]*h[7]+ jj[1][0]*h[8]
            y2 = 0+ 0+ 0+ -jj[0][0]*h[3] -jj[0][1]*h[4] -1*h[5]+ jj[1][1]*jj[0][0]*h[6]+ jj[1][1]*jj[0][1]*h[7]+ jj[1][1]*h[8]
        
            if np.linalg.norm([x2,y2])<0.0065:
                
                gmatch.append(jj)
        
        if len(gmatch)>numinliers:
            numinliers = len(gmatch)
            print(numinliers)
            homoandinliers = [[h],numinliers,gmatch]
    return homoandinliers

def matchkeypoints(des1,des2,kp1,kp2):
    matchkeypoints=[]
    for di,d in enumerate(des1):
        mini = 99999999
        mininb = 99999999
        for dni,dn in enumerate(des2):
            eudis = np.linalg.norm(d-dn)
            if eudis < mini:
                mininb = mini
                mini = eudis
                ind = dni
            elif eudis < mininb:
                mininb = eudis
        if mini/mininb < 0.8:
        
            matchkeypoints.append([kp1[di],kp2[ind]])
    return matchkeypoints

def main():

    # Converting imgs into grayscale and collecting in the list called grayimgs
    grayimgs = []
    
    folder = sys.argv[1]
    # folder = "hotel1"
    for filename in os.listdir(folder):
        if filename=="panorama.jpg":
            continue
        print(filename)
        I = cv2.imread(os.path.join(folder, filename))
        
        
        I = cv2.resize(I, (600,500), interpolation = cv2.INTER_AREA)  #Resize 600,500
        
        grayimgs.append(I)

    # number of images
    numimgs = len(grayimgs)

    # sift
    sift = cv2.xfeatures2d.SIFT_create()

    # mking list of KeyPoints and Descriptors for each img
    kpl=[]
    desl=[]
    for n in grayimgs:
        kp, des = sift.detectAndCompute(n,None)
        kpl.append(kp)
        desl.append(des)
        print(len(kp))

    # getting pixle coordinates of key points
    kppixcoor = []
    for ii in range(numimgs):
        kppixcoor.append(cv2.KeyPoint_convert(kpl[ii])) # list of list of list


    # matching points
    numcomb = math.factorial(numimgs)/(math.factorial(numimgs-2)*2)
    #print('matching points')
    matchpoints = {}
    for desli,deslv in enumerate(desl[:-1]):
        for deni,denv in enumerate(desl[desli+1:]):
            
            matchpoints[(desli, deni+desli+1)]=[]
            for di,d in enumerate(deslv):
                mini = 99999999
                mininb = 99999999
                for dni,dn in enumerate(denv):
                    eudis = np.linalg.norm(d-dn)
                    if eudis < mini:
                        mininb = mini
                        mini = eudis
                        ind = dni
                    elif eudis < mininb:
                        mininb = eudis
                if mini/mininb < 0.8:
                
                    matchpoints[(desli, deni+desli+1)].append([kppixcoor[desli][di],kppixcoor[deni+desli+1][ind]])
            #print(len(matchpoints[(desli, deni+desli+1)]))

    
    # finding images that overlap    
    print("matchpoints pairs",matchpoints.keys())
    matchingpairs = []
    for v in matchpoints.keys():
        if (0.95>len(matchpoints[v])/len(kpl[v[0]])>0.08) or (0.95>len(matchpoints[v])/len(kpl[v[1]])>0.08):#0.09
            matchingpairs.append(v)

    # print("matchingpairs", matchingpairs)

    matchpointsallorders = {}
    for kk in matchpoints.keys():
        matchpointsallorders[(kk[1],kk[0])] = list(np.fliplr(matchpoints[kk]))
        matchpointsallorders[kk] = matchpoints[kk]

    # finding end images
    freq = {}
    for i in range(numimgs):
        freq[i] = 0
        for j in matchingpairs:
            if i in j:
                freq[i] += 1
    sorted_freq = sorted(freq.items(), key=operator.itemgetter(1))
    print("sorted_freq", sorted_freq)
    freqlis=[]
    for i in sorted_freq:
        freqlis.append(i[1])
    minfreq = min(freqlis)
    endpoints = []
    for i in sorted_freq:
        if i[1] <= minfreq:
            endpoints.append(i[0])
    print("endpoints", endpoints)
    
    
    #  finding image order
    chain = {}
    L = endpoints.copy()
    
    epdone = []
    for p in endpoints:
        mpcopy = matchingpairs.copy()
        if p not in epdone:
            current = p
            chain[p] = [p]
            L.remove(p)
            print(current in L)
            while current not in L: 
                for i in mpcopy:
                    print(i)
                    if current in i:
                        if i[0] == current:
                            current=i[1]
                            print(current)
                            chain[p].append(current)
                            mpcopy.remove(i)
                            break
                        else:
                            current=i[0]
                            print(current)
                            chain[p].append(current)
                            mpcopy.remove(i)
                            break
            epdone.append(p)
            epdone.append(current)
    

    leftrightH = {}
    for g in chain.keys():
        leftrightH[g] = findhomography(matchpointsallorders[(chain[g][0], chain[g][1])])
        tl = np.dot(np.array(leftrightH[g][0]).reshape(3,3) ,[[0],[0],[1]])
        if tl[0]/tl[2]>0 or tl[1]/tl[2]>0:
            chain[g].reverse()
    
    finalorder = []
    for g in chain.keys():
        finalorder = chain[g]
    for g in chain.keys():
        print(g, chain[g])
        if len(chain[g])>len(finalorder):
            finalorder = chain[g].copy()
    
    print(finalorder)
    reversefinalorder = finalorder[::-1]
    print(reversefinalorder)
    
    # calculating homography for each adjacent pair in order of the images
    h=[]
    for i,v in enumerate(reversefinalorder[:-1]):
        h.append(findhomography(matchpointsallorders[(reversefinalorder[i], reversefinalorder[i+1])]))


    # final stitching
    panorama = grayimgs[reversefinalorder[0]]
    # wid = 0
    # hei = grayimgs[reversefinalorder[0]].shape[0]
    wid = grayimgs[reversefinalorder[0]].shape[1] #
    hei = grayimgs[reversefinalorder[0]].shape[0] #
    for imi,im in enumerate(reversefinalorder[:-1]):
        wid += panorama.shape[1] + grayimgs[reversefinalorder[imi+1]].shape[1]
        hei += panorama.shape[0] + grayimgs[reversefinalorder[imi+1]].shape[0] #
        # wid += grayimgs[reversefinalorder[imi+1]].shape[1]
        # hei += panorama.shape[0] + grayimgs[reversefinalorder[imi+1]].shape[0]
        panorama = cv2.warpPerspective(panorama, np.array(h[imi][0]).reshape(3,3), (wid, hei))
        panorama[0:grayimgs[reversefinalorder[imi+1]].shape[0],0:grayimgs[reversefinalorder[imi+1]].shape[1]] = grayimgs[reversefinalorder[imi+1]]
        plt.figure()
        plt.imshow(panorama)
    # wid += panorama.shape[1] + grayimgs[reversefinalorder[imi+1]].shape[1]
    # hei += panorama.shape[0] + grayimgs[reversefinalorder[imi+1]].shape[0]
    panorama = cv2.warpPerspective(panorama, np.array([[1, 0, 0],[0, 1, hei/2],[0, 0, 1]]), (wid, hei))
    plt.figure()
    plt.imshow(panorama)
    
    
    img = np.asarray(panorama, dtype=np.uint8)
    cv2.imwrite(os.path.join(folder , 'panorama.jpg'), img)
    plt.show()
    # cv2.imwrite('panorama.jpg', img)

    
    
    
    # matchpoints.clear()
    # matchpointsallorders.clear()
    # panorama = grayimgs[reversefinalorder[0]]
    # kpl.clear()
    # desl.clear()
    # wid = 0
    # hei = 0
    # for imi, im in enumerate(reversefinalorder[:-1]):
    #     kppanorama, despanorama = sift.detectAndCompute(panorama,None)
    #     nextim = grayimgs[reversefinalorder[imi+1]]
    #     kpnextim, desnextim = sift.detectAndCompute(nextim,None)
    #     kppcpanorama = cv2.KeyPoint_convert(kppanorama)
    #     kppcnextim = cv2.KeyPoint_convert(kpnextim)
    #     matchingpoints = matchkeypoints(despanorama,desnextim,kppcpanorama,kppcnextim)
    #     h = findhomography(matchingpoints)
    #     wid += panorama.shape[1] + nextim.shape[1]
    #     hei += panorama.shape[0] + nextim.shape[0]
    #     # plt.figure()
    #     # plt.imshow(panorama)
    #     panorama = cv2.warpPerspective(panorama, np.array(h[0]).reshape(3,3), (wid, hei))
    #     panorama[0:nextim.shape[0],0:nextim.shape[1]] = nextim
    #     print("BOBO")
    #     # panorama = panorama
    #     # plt.figure()
    #     # plt.imshow(panorama)

    # plt.figure()
    # plt.imshow(panorama)
    # panorama = cv2.warpPerspective(panorama, np.array([[1, 0, 0],[0, 1, hei/2],[0, 0, 1]]), (wid, hei))
    # plt.figure()
    # plt.imshow(panorama)
    # plt.show()




main()





