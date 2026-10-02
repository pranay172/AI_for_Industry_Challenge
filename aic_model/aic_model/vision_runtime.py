"""Initialize the compatible vision library before TorchVision's bundled JPEG.

The pinned conda OpenCV/TIFF needs libjpeg's 12-bit API. Loading TorchVision's
older bundled libjpeg.so.8 first shadows that API despite LD_LIBRARY_PATH.
Keep this import order shared by runtime, training and evaluation entry points.
"""
import cv2  # noqa: F401 -- deliberate native-library initialization
from torchvision import transforms
