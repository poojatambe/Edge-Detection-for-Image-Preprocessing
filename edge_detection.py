import cv2
import numpy as np


class EdgeDetect:
    def __init__(self, image):
        self.img = image

    def sobel_edge(self, thresh, kernel_size, opencv):
        # img = cv2.imread(self.path)
        gray_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if opencv == False:
            if kernel_size == 3:
                gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gradient_x = cv2.filter2D(gray_im, cv2.CV_64F, gx)
                gradient_y = cv2.filter2D(gray_im, cv2.CV_64F, gy)
                magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                threshold = thresh
                edge = magnitude > threshold
                sobel_edge = edge.astype(np.uint8) * 255
                return sobel_edge, gradient_x, gradient_y
            else: 
                gx = np.array([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], 
                               [-3, -5, 0, 5, 3], [-2, -3, 0, 3, 2], 
                               [-1, -2, 0, 2, 1]])
                gy = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 3, 2], 
                               [0, 0, 0, 0, 0], [-2, -3, -5, -3, -2], 
                               [-1, -2, -3, -2, -1]])
                gradient_x = cv2.filter2D(gray_im, cv2.CV_64F, gx)
                gradient_y = cv2.filter2D(gray_im, cv2.CV_64F, gy)
                magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                threshold = thresh
                edge = magnitude > threshold
                sobel_edge = edge.astype(np.uint8) * 255
                return sobel_edge, gradient_x, gradient_y
        else: 
            # cv2 sobel implmentation
            sobelx = cv2.Sobel(src=gray_im, ddepth=cv2.CV_64F,
                               dx=1, dy=0, ksize=kernel_size)
            sobely = cv2.Sobel(src=gray_im, ddepth=cv2.CV_64F,
                               dx=0, dy=1, ksize=kernel_size)
            sobelxy = cv2.Sobel(src=gray_im, ddepth=cv2.CV_64F,
                                dx=1, dy=1, ksize=kernel_size)
            sobelx = cv2.convertScaleAbs(sobelx)
            sobely = cv2.convertScaleAbs(sobely)
            return sobelxy, sobelx, sobely

    def robert_edge(self, thresh):
        """
        Robert edge detection.
        """
        # img = cv2.imread(self.path)
        gray_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        gx = np.array([[0, 1], [-1, 0]])
        gy = np.array([[1, 0], [0, -1]])

        gradient_x = cv2.filter2D(gray_im, cv2.CV_64F, gx)
        gradient_y = cv2.filter2D(gray_im, cv2.CV_64F, gy)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        threshold = thresh
        edge = magnitude > threshold
        robert_edge = edge.astype(np.uint8) * 255
        return robert_edge, gradient_x, gradient_y     

    def prewitt_edge(self, thresh):
        """
        Prewitt edge detection
        """
        # img = cv2.imread(self.path)
        gray_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        gradient_x = cv2.filter2D(gray_im, cv2.CV_64F, gx)
        gradient_y = cv2.filter2D(gray_im, cv2.CV_64F, gy)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        threshold = thresh
        edge = magnitude > threshold
        prewitt_edge = edge.astype(np.uint8) * 255
        return prewitt_edge, gradient_x, gradient_y 

    def laplacian_edge(self, kernel_size):
        """
        Laplacian edge detector
        """
        # img = cv2.imread(self.path)
        gray_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # remove noise
        n_img = cv2.GaussianBlur(gray_im, (3, 3), 0)
        log_img = cv2.Laplacian(n_img, ddepth=cv2.CV_16S, ksize=kernel_size)
        log_img = cv2.convertScaleAbs(log_img)
        return log_img

    def canny_edge(self, t1, t2, aperture_size, l2norm):
        """
        Canny edge detection
        """
        # img = cv2.imread(self.path)
        gray_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # remove noise
        n_img = cv2.GaussianBlur(gray_im, (3, 3), 0)

        canny_img = cv2.Canny(n_img, t1, t2, 
                              apertureSize=aperture_size, L2gradient=l2norm)
        return canny_img

    def holistically_nested_edge(self, factor, swap):
        protoPath = r"D:/Winjit_training/ML/deploy.prototxt.txt"
        modelPath = r"D:/Winjit_training/ML/hed_pretrained_bsds.caffemodel"
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath) 
        (H, W) = self.img.shape[:2]

        mean_pixel_values = np.average(self.img, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(self.img,
                                     scalefactor=factor,
                                     size=(W, H), 
                                     mean=(mean_pixel_values[0], 
                                           mean_pixel_values[1], 
                                           mean_pixel_values[2]), 
                                    swapRB=swap, 
                                    crop=False
                                    )

        #View image after preprocessing (blob)
        blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)

        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        net.setInput(blob)
        hed = net.forward()
        hed = hed[0, 0, :, :]   # Drop the other axes 
        hed = (255 * hed).astype("uint8")  # rescale to 0-255 
        return blob_for_plot, hed
