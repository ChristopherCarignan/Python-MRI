#!python3
#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import path
from PIL import ImageTk
import PIL.Image
import cv2
import PIL.ImageDraw
import sys, getopt, argparse
from tkinter import *
from scipy.ndimage.morphology import binary_fill_holes as imfill
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pdb

def main(argv):
    '''
    Created by Christopher Carignan 2019
    Institute für Phonetik und Sprachverarbeitung (Ludwig-Maximilians-Universität München)
    c.carignan@phonetik.uni-muenchen.de

    Development funded by: 
        -   European Research Council Advanced Grant 295573 
            "Human interaction and the evolution of spoken accent (J. Harrington)
        -   Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) grant HA 3512/15-1 
            "Nasal coarticulation and sound change: a real-time MRI study" (J. Harrington & J. Frahm)
        
    Description:
        User provides a matrix of MRI video frames, then selects a region of interest (ROI).
        The ROI can include any number of points/vertices, with any shape.
        When the ROI is closed by clicking on the initial point, a principal components analysis (PCA) model is
            performed, using the pixels within the ROI as dimensions and the MRI video frames as observations.
        Only the first PC (PC1) is retained, which captures the most amount of variance and (hopefully)
            the primary articulatory degree of freedom within the ROI.
        The output provides a plot with:
            1) the PC1 coefficients/loadings overlaid on a video frame, in order to interpret the PC1 space
            2) a time-varying signal composed of PC1 scores (one score for each MRI video frame)
    
    Arguments:
        - -i: File name of MRI (required)
        - -o: File name of PC1 plot to be saved (optional)
    '''
    videofile = ''
    pcaplot = ''
    try:
        # Check for input (-i) and output (-o) arguments
        opts, args = getopt.getopt(argv,'hi:o:',['ifile=','ofile='])
    except getopt.GetoptError:
        print('MRI_PCA.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == ('-h','--help'):
            print('MRI_PCA.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ('-i', '--ifile'):
            videofile = arg
        elif opt in ('-o', '--ofile'):
            pcaplot = arg

    # If MATLAB file, load and extract MRI images
    if videofile.endswith('.mat'):
        matfile = sio.loadmat(videofile)
        video = matfile['data']
        maxval = np.max(video)
        for frame in range(video.shape[2]):
            video[:,:,frame] = video[:,:,frame][::-1] # flip the frames
            video[:,:,frame] = np.multiply(video[:,:,frame], 255/maxval) # scale to 8-bit
        img = PIL.Image.fromarray(video[:,:,0]) # get first frame in video file, convert it to image

    # If numpy matrix, load it
    elif videofile.endswith('.npy'): 
        video = np.load(videofile)
        maxval = np.max(video)
        for frame in range(video.shape[2]):
            video[:,:,frame] = video[:,:,frame][::-1] # flip the frames
            video[:,:,frame] = np.multiply(video[:,:,frame], 255/maxval) # scale to 8-bit
        img = PIL.Image.fromarray(video[:,:,0]) # get first frame in video file, convert it to image
        
    # If neither MATLAB nor numpy matrix... I don't know how to help you, my friend :(
    else:
        print('Unknown MRI video file type')
        sys.exit(2)

    # Setting up a tkinter canvas
    root = Tk()
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    canvas = Canvas(frame, bd=0)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    frame.pack(fill=BOTH,expand=1)

    # Adding the image to the canvas
    imgdata = np.asarray(img, dtype="uint8")
    img = ImageTk.PhotoImage(img) # convert for tkinter canvas
    canvas.create_image(0,0,image=img,anchor='nw') # add to canvas

    # Preallocate canvas attributes
    canvas.mycoords = np.array([[0,0]])
    canvas.clicks = 0
    canvas.myline = None

    # Function to create polygon mask, using click points at vertices
    def points2mask(canvas,imgdata):
        mask = np.zeros([imgdata.shape[0], imgdata.shape[1]], dtype="double")
        cv2.fillConvexPoly(mask, canvas.mycoords, 1)
        return mask

    # Function to apply ROI mask to all video frames
    # and perform a principal components analysis on pixels within ROI
    def doPCA(canvas,video,mask):
        myheight = max(canvas.mycoords[:,1]) - min(canvas.mycoords[:,1])
        mywidth = max(canvas.mycoords[:,0]) - min(canvas.mycoords[:,0])
        mylength = myheight * mywidth
        pcmat = np.zeros([video.shape[2],mylength])

        for frame in range(video.shape[2]):
            im = PIL.Image.fromarray(video[:,:,frame]) # convert data array to image
            imbw = im.convert('L') # convert image to black & white
            imbw = np.asarray(imbw) # convert back to data array
            maskedimg = np.multiply(imbw,mask) # apply the region of interest mask

            # Crop the image, flatten to 1-dimensional vector, add to array for principle components analysis
            crop = maskedimg[min(canvas.mycoords[:,1]):max(canvas.mycoords[:,1]),min(canvas.mycoords[:,0]):max(canvas.mycoords[:,0])]
            crop = np.hstack(crop)
            pcmat[frame,:] = crop

        # Perform the principle components analysis
        x = StandardScaler().fit_transform(pcmat) # standardize the pixel dimensions
        pca = PCA(n_components=1) # do the PCA
        PCs = pca.fit_transform(x) # transform the results based on the standardized dimensions

        # Get the PC1 scores
        scores = pd.DataFrame(data = PCs)
        
        # Get the PC1 coefficients/loadings
        loadings = pca.components_ * np.sqrt(pca.explained_variance_)
        loadings = np.reshape(loadings, (myheight, mywidth))
        loadings = ( (loadings - loadings.min()) * (1/(loadings.max() - loadings.min())) * 255).astype('uint8')

        # Add the PC1 loadings to a base image, so that they can be interpreted
        coeffimg = video[:,:,0]
        coeffimg[min(canvas.mycoords[:,1]):max(canvas.mycoords[:,1]),min(canvas.mycoords[:,0]):max(canvas.mycoords[:,0])] = loadings
        
        return scores, coeffimg

    def plotPCA(scores,coeffs,pcaplot):
        # Create figure with two subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.imshow(coeffs) # plot the PC1 coefficient/loading image in first subplot
        ax2.plot(scores, c='red', linewidth=4.0) # plot the PC1 scores in second subplot

        ax1.yaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        ax2.set_ylabel("PC1 score")
        ax2.set_xlabel("Frame number")

        # If user provides an output file name, save the PCA plot
        if pcaplot != '':
            # Save it!
            plt.savefig(pcaplot, bbox_inches='tight', pad_inches=0)
            # Plot it!
            plt.show()
        else:
            # Plot it!
            plt.show()


    # Function to be called when mouse is clicked
    def updatecoords(event):
        # Save coordinate info to canvas attributes
        if canvas.clicks == 0:
            canvas.mycoords[canvas.clicks,:] = [event.x,event.y] # update coordinate list
            canvas.box = canvas.create_rectangle(event.x-4, event.y-4, event.x+4, event.y+4, fill='white') # draw box
        else:
            # Close canvas if click is the same as the first click (i.e., the region of interest is complete)
            if canvas.coords(canvas.box)[0] <= event.x <= canvas.coords(canvas.box)[2] and \
               canvas.coords(canvas.box)[1] <= event.y <= canvas.coords(canvas.box)[3]:

                polymask = points2mask(canvas,imgdata) # create a mask with the clicked points
                pcscores, pccoeffs = doPCA(canvas,video,polymask) # do the PC analysis
                root.destroy() # close the canvas

                plotPCA(pcscores,pccoeffs,pcaplot) # plot the results

            # If the region of interest is not complete, update the canvas attributes
            else:
                canvas.mycoords = np.vstack((canvas.mycoords,[event.x,event.y])) # update coordinate list
                canvas.create_line(canvas.mycoords[canvas.clicks-1,0], canvas.mycoords[canvas.clicks-1,1],
                    canvas.mycoords[canvas.clicks,0], canvas.mycoords[canvas.clicks,1],fill='red') # make old line red
                canvas.create_rectangle(event.x-4, event.y-4, event.x+4, event.y+4, fill='white') # draw box

        canvas.clicks += 1 # update number of clicks
        
    # Function to be called when mouse is moved
    def drawline(event):
        x, y = event.x, event.y # get mouse coordinates
        if canvas.clicks > 0: # if mouse has been clicked at least once, draw a line
            if canvas.myline is not None:
                canvas.delete(canvas.myline) # update deletion of line from previous mouse movement

            # Color first box red if mouse is hovering over it
            if canvas.coords(canvas.box)[0] <= event.x <= canvas.coords(canvas.box)[2] and \
                canvas.coords(canvas.box)[1] <= event.y <= canvas.coords(canvas.box)[3]:

                canvas.itemconfig(canvas.box,fill='red')
            else:
            # Color it white if not...
                canvas.itemconfig(canvas.box,fill='white')
            
            # Draw a white line extending from previous click
            canvas.myline = canvas.create_line(canvas.mycoords[canvas.clicks-1,0],
                canvas.mycoords[canvas.clicks-1,1], x, y, fill='white')

    # Mouseclick event
    canvas.bind("<Button 1>",updatecoords)

    # Mousemove event
    canvas.bind("<Motion>", drawline)

    root.mainloop()

if __name__ == "__main__":
    main(sys.argv[1:])