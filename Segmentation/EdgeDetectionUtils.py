def applyEdgeDetection(img, modelPath):
    import cv2 as cv
    edge_detection = cv.ximgproc.createStructuredEdgeDetection(modelPath)
    import numpy as np
    edges = edge_detection.detectEdges(np.float32(img) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv.ximgproc.createEdgeBoxes()
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)
    return boxes, scores