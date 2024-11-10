from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main(): 
     original = cv2.imread("C:/College Materials/4th YEAR/SEM 7/Minor Projecct/project/SRGAN_CustomDataset-main/result/corss_final/og/4.jpg") 
     compressed = cv2.imread("C:/College Materials/4th YEAR/SEM 7/Minor Projecct/project/SRGAN_CustomDataset-main/result/corss_final/super_resolution/4.png", 1) 
     value = PSNR(original, compressed) 
     print(f"PSNR value is {value} dB") 
       
if __name__ == "__main__": 
    main() 