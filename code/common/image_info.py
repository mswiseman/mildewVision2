import cv2

img = cv2.imread("/Users/michelewiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/data/labeled/3176_infected.png")
height, width, channels = img.shape
print("height:", height)
print("width:", width)
print("channels:", channels)