import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 as cv
from skimage import filters 
from skimage import exposure
import utils.data_preperation.data_information as info
import utils.data_preperation.data_information

# -----------------------------------------------------------
# Plot Usefult Information
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

class recognizability:
    recognizable = []
    not_recognizable = []

    def __init__(self, csv_recognizable_path, csv_not_recognizable_path):
        self.recognizable, self.not_recognizable = self.__normalize_data(csv_recognizable_path, csv_not_recognizable_path)

    def cluster(self):
        df = np.append(self.recognizable[:,3:5],self.not_recognizable[:,3:5], axis=0)
        #Initialize the class object
        kmeans = KMeans(n_clusters= 4,init='k-means++', 
                max_iter=1000, n_init=1, verbose=0, random_state=1)

        #predict the labels of clusters.
        label = kmeans.fit_predict(df)
        
        u_labels = np.unique(label)
                
        #Getting the Centroids
        centroids = kmeans.cluster_centers_
        u_labels = np.unique(label)
        
        #plotting the results:
        labels = ["too bright","problem mix","recognizable","terrain differences"]
        for i in u_labels:
            plt.scatter(df[label == i , 0] , df[label == i , 1] , label = labels[i],alpha = 0.6, s=10)
        print(len(centroids))
        plt.title("Visibility Clusters in Turbine Shadow Detection")
        plt.xlabel('standard deviation')
        plt.ylabel('mean')
        plt.scatter(centroids[:,0] , centroids[:,1] ,c='b', marker='^', s=20)
        plt.legend()
        plt.show()

    def __normalize_data(self,csv_recognizable_path,csv_not_recognizable_path):
        
        # Using numpy we can use the function loadtxt to load your CSV file.
        # We ignore the first line with the column names and use ',' as a delimiter.
        recognizable = np.loadtxt(csv_recognizable_path, delimiter=',', skiprows=1)
        not_recognizable = np.loadtxt(csv_not_recognizable_path, delimiter=',', skiprows=1)

        # Normalze Values
        v = recognizable[:,3]
        v2 = not_recognizable[:,3]
        recognizable[:,3] = (v - v.min()) / (v.max() - v.min())
        not_recognizable[:,3] = (v2 - v2.min()) / (v2.max() - v2.min())
        return recognizable, not_recognizable

    def recognizability(self):
        #normal PLot
        plt.scatter(self.recognizable[:,3], self.recognizable[:,4], s=10, c='b', marker="s", label='recognizable')
        plt.scatter(self.not_recognizable[:,3], self.not_recognizable[:,4], s=10, c='r', marker="o", label='not_recognizable')
        plt.title("Distribution of recognizability in Turbine Shadow Detection")
        plt.xlabel('standard deviation')
        plt.ylabel('mean')
        
        plt.legend(loc='upper left');   

        # Show the legend
        plt.show()
    

# -----------------------------------------------------------
# Image Enhancement methods
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

class img_plots:
    path = ""
    id = 0
    bands = ""
    date = ""

    def __init__(self,jpg_filePath, id, bands, date):
        self.path = jpg_filePath
        self.id = id 
        self.bands = bands
        self.date = date

    def plot_img_distribution(self):
        imgArr = []
        equArr = []
        localEnhance = []
        modernArt = [] 
        non_linear_gauss = []
        maxBrightness = 0
        for band in self.bands:
            brighnessVal = 0
            path = f'{self.path}/{self.id}-{self.date}-{band}.jpg'
            arr = cv.imread(path ,0)
            path = f'{self.path}/{self.id}-{self.date}-{band}.jpg'
            imgArr.append( cv.imread(path ,0))
            #blur and Hist
            equ = cv.equalizeHist(arr)
            equArr.append(equ   	)
            thresh = cv.threshold(equ, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
            cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                if cv.contourArea(c) < 300:
                    cv.drawContours(thresh, [c], -1, (0,0,0), -1)

            result = 255 - thresh
            modernArt.append(result)
            #local contrast enhancement
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            smooth = cv.GaussianBlur(arr, (95,95), 0)
            # divide gray by morphology image
            division = cv.divide(arr, smooth, scale=255)
            # sharpen using unsharp masking
            result = filters.unsharp_mask(division, radius=1.5, amount=1.5, multichannel=False, preserve_range=False)
            result2 = cv.filter2D(result,-1,kernel)
            localEnhance.append((255*result2).clip(0,255).astype(np.uint8))
            enhanced_gauss, cdf_gauss = self.gauss_enhancement(equ, 0.15)
            non_linear_gauss.append(enhanced_gauss)
            #cv.imwrite(f'{self.path}/{self.id}-{self.date}-{band}-HIST.jpg',equ)
            cv.imwrite(f'{self.path}/{self.id}-{self.date}-{band}.jpg',arr)
            thisBright = utils.data_information.calcBrightness(arr)
            if thisBright>brighnessVal:
                maxBrightness = self.bands.index(band)
                brighnessVal = thisBright

        matchArr = []
        ref = imgArr[maxBrightness]
        
        for i in range(len(imgArr)):
            multi = True if imgArr[i].shape[-1] > 1 else False
            matched = exposure.match_histograms(imgArr[i], ref, multichannel=multi)
            matchArr.append(matched)


        return equArr, matchArr

    def get_cdf_hist(self,image_input):
        """
        Method to compute histogram and cumulative distribution function
        :param image_input: input image
        :return: cdf
        """
        hist, bins = np.histogram(image_input.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        return cdf_normalized

    def gauss_enhancement(self,image, gain):
        """
        Non-linear transformation function to enhance brightness and contrast
        :param image: input image
        :param gain: contrast enhancement factor
        :return: enhanced image
        """
        normalized_image = image / np.max(image)
        enhanced_image = 1 - np.exp(-normalized_image**2/gain)
        enhanced_image = enhanced_image*255
        cdf = self.get_cdf_hist(enhanced_image)
        return enhanced_image, cdf

    def adjust_gamma(self,image, gamma=0.5):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv.LUT(image, table)

    def motion_detector_Pixel_Differences(self, add=""):
        frame_count = 0
        previous_frame = None
        img_arr = []
        for band in self.bands:
            frame_count += 1

            arr = cv.imread(f'{self.path}/{self.id}-{self.date}-{band}{add}.jpg' ,0)
            prepared_frame = cv.GaussianBlur(src=arr, ksize=(5,5), sigmaX=0)   
            if (previous_frame is None):
                # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue     
            # calculate difference and update previous frame
            diff_frame = cv.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv.threshold(src=diff_frame, thresh=35, maxval=255, type=cv.THRESH_BINARY)[1]
            img_arr.append(thresh_frame)
        
        return img_arr

    def histogram_matching(self, arr):
        maxBrightness = 0
        for img in arr:
            for band in self.bands:
                brighnessVal = 0
                thisBright = info.calcBrightness(img)
                if thisBright>brighnessVal:
                    maxBrightness = self.bands.index(band)
                    brighnessVal = thisBright

        matchArr = []
        ref = arr[maxBrightness]
        
        for i in range(len(arr)):
            matched = exposure.match_histograms(np.asarray(arr[i]), ref)
            matchArr.append(matched.astype(np.uint8))
        
        return matchArr

    def histogram_Transform(self, arr):
        maxBrightness = 0
        brighnessVal = 0

        for img in range(len(arr)):
                thisBright = utils.data_information.calcBrightness(arr[img])
                if thisBright>brighnessVal:
                    maxBrightness = img
                    brighnessVal = thisBright
                    print(f"chose: {brighnessVal}")

        matchArr = []
        ref = arr[maxBrightness]
        
        for i in range(len(arr)):
            matched = exposure.match_histograms(ref, np.asarray(arr[i]))
            matchArr.append(matched.astype(np.uint8))
        
        return matchArr