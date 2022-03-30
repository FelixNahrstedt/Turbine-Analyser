from os import dup
import numpy as np
import cv2 as cv
from PIL import ImageGrab, Image
from scipy import signal
import time
import moviepy.editor as mp
import numpy as np
import argparse
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk



# -----------------------------------------------------------
# all functions for "normal" change detection in Images
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

def motion_detector():
  
  frame_count = 0
  previous_frame = None
  
  while True:
    frame_count += 1

    # 1. Load image; convert to RGB
    img_brg = np.array(ImageGrab.grab())
    img_rgb = cv.cvtColor(src=img_brg, code=cv.COLOR_BGR2RGB)

    if ((frame_count % 2) == 0):

      # 2. Prepare image; grayscale and blur
      prepared_frame = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
      prepared_frame = cv.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)

def dense_optical_flow(image0, image1):
  # --- Compute the optical flow
  v, u = optical_flow_ilk(np.array(image0), np.array(image1), radius=1, prefilter=True)
  # --- Compute flow magnitude
  norm = np.sqrt(u ** 2 + v ** 2)

  # --- Display
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

  # --- Sequence image sample

  ax0.imshow(image0, cmap='gray')
  ax0.set_title("Sequence image sample")
  ax0.set_axis_off()

  # --- Quiver plot arguments

  nvec = 20  # Number of vectors to be displayed along each image dimension
  nl, nc = image0.shape
  step = max(nl//nvec, nc//nvec)

  y, x = np.mgrid[:nl:step, :nc:step]
  u_ = u[::step, ::step]
  v_ = v[::step, ::step]

  ax1.imshow(norm)
  ax1.quiver(x, y, u_, v_, color='r', units='dots',
            angles='xy', scale_units='xy', lw=3)
  ax1.set_title("Optical flow magnitude and vector field")
  ax1.set_axis_off()
  fig.tight_layout()

  plt.show()

def lucas_kanade(path_to_gif,path_images):
  cap = cv.VideoCapture(cv.samples.findFile(path_to_gif))
  feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
  # Parameters for lucas kanade optical flow
  lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
  # Create some random colors
  color = np.random.randint(0, 255, (100, 3))

  # Take first frame and find corners in it
  ret, old_frame = cap.read()
  old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
  p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  # Create a mask image for drawing purposes
  mask = np.zeros_like(old_frame)

  while(1):        
      ret, frame = cap.read()
      if not ret:
          cv.imwrite(f'{path_images}-bgr_flow.png', img)
          print('No frames grabbed!')
          break
      frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      # calculate optical flow
      p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params, minEigThreshold=1e-2)
      # Select good points
      if p1 is not None:
          good_new = p1[st==1]
          good_old = p0[st==1]
      # draw the tracks
      for i, (new, old) in enumerate(zip(good_new, good_old)):
          a, b = new.ravel()
          c, d = old.ravel()
          mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(),2)
          frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), 1)
      img = cv.add(frame, mask)
      cv.imshow('frame', img)
      k = cv.waitKey(200) & 0xff
      if k == 27:
          break
      # Now update the previous frame and previous points
      old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1, 1, 2)
  cv.destroyAllWindows()