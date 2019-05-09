#!python3
#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageTk
import PIL.Image
import sys, getopt, argparse, cv2
from tkinter import *

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
        User provides a matrix of MRI video frames, then selects a line for analysis.
        The output provides a time-varying plot with intensities of pixel sites that fall along the line selection.
        Although the intention is for estimating changing larynx height, the function can be used for any articulator.
    
    Arguments:
        - -i: File name of MRI (required)
        - -o: File name of time-varying plot (optional)
    '''
    videofile = ''
    larynxplot = ''
    try:
        # Check for input (-i) and output (-o) arguments
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('MRI_larynx_height.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('MRI_larynx_height.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            videofile = arg
        elif opt in ("-o", "--ofile"):
            larynxplot = arg

    # If MATLAB file, load and extract MRI images
    if videofile.endswith('.mat'):
        matfile = sio.loadmat(videofile)
        video = matfile['data']
        imgmat = video[:,:,0] # Get first frame in video file
        mpimg.imsave('MRI_img.png', imgmat) # Save frame
    # If numpy matrix, load it
    elif videofile.endswith('.npy'): 
        video = np.load(videofile)
        imgmat = video[:,:,0]
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
    img = PIL.Image.open('MRI_img.png') # Open image
    img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM) # Flip image vertically
    img = ImageTk.PhotoImage(img) # Convert for tkinter canvas
    canvas.create_image(0,0,image=img,anchor="nw") # Add to canvas

    # Preallocate canvas attributes
    canvas.coords = np.array([[0,0],[0,0]])
    canvas.clicks = 0
    canvas.myline = None

    # Function to be called when mouse is clicked
    def printcoords(event):
        canvas.coords[canvas.clicks,:] = [event.x,event.y] # Save coordinates to canvas attribute
        canvas.clicks += 1 # Update number of clicks
        if canvas.clicks == 2: # Close canvas on last (i.e., second) click 
            root.destroy()
    
    # Function to be called when mouse is moved
    def drawline(event):
        x, y = event.x, event.y # Get mouse coordinates
        if canvas.clicks == 1: # If mouse has been clicked once, draw a line
            if canvas.myline is not None:
                canvas.delete(canvas.myline) # Update deletion of line from previous mouse movement
            # Draw a white line extending from first click
            canvas.myline = canvas.create_line(canvas.coords[0,0], canvas.coords[0,1], x, y, fill="white")

    # Mouseclick event
    canvas.bind("<Button 1>",printcoords)

    # Mousemove event
    canvas.bind("<Motion>", drawline)

    root.mainloop()

    def createLineIterator(P1, P2, img):
        """
        Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
        """
        # Define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # Difference and absolute difference between points
        # (used to calculate slope and relative location between points)
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # Predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X: # It's a vertical line segment
            itbuffer[:,0] = P1X
            if negY:
                itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
        elif P1Y == P2Y: # It's a horizontal line segment
            itbuffer[:,1] = P1Y
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
        else: # It's a diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32)/dY.astype(np.float32)
                if negY:
                    itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
                else:
                    itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
                itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32)/dX.astype(np.float32)
                if negX:
                    itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
                else:
                    itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
                itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:,0]
        colY = itbuffer[:,1]
        itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

        # Get intensities from image array
        itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

        return itbuffer

    # Get the intensity values of the pixels along the selected line
    itbuffer = createLineIterator(canvas.coords[0,:], canvas.coords[1,:], imgmat)

    larynx = np.zeros((itbuffer.shape[0],video.shape[2])) # Preallocate numpy array

    # Arrange coordinates so that the y-axis always has the same orientation
    if canvas.coords[1,1] < canvas.coords[0,1]:
        canvas.coords = np.roll(canvas.coords,-1,axis=0)

    # For each frame in the video file, get the intensity values of the pixels along the selected line
    # Plot the results over time (x-axis)
    for frame in range(0,int(video.shape[2])):
        I = video[:,:,frame] # Get MRI frame
        itbuffer = createLineIterator(canvas.coords[0,:], canvas.coords[1,:], I[::-1]) # Use vertically flipped frame
        larynx[:,frame] = itbuffer[:,2]

    # Plot the time-varying larynx height
    imgplot = plt.imshow(larynx)
    plt.axis('off')
    plt.show(imgplot)

    # If user provides an output file name, save the larynx height plot
    if larynxplot != '':
        mpimg.imsave(larynxplot, larynx)

if __name__ == "__main__":
    main(sys.argv[1:])