import cv2 as cv2
# read image
def readImage(imagePath):
    img = cv2.imread(imagePath)
    img2rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img2rgb

def applySelectiveSearch(img, isCombined):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.clearStrategies()
    color_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    texture_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    fill_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    size_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    if isCombined:
        strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(color_strategy,
                                                                                                    texture_strategy,
                                                                                                    fill_strategy,
                                                                                                    size_strategy)
    else:
        strategy=color_strategy
    ss.addStrategy(strategy)

    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    bboxes = ss.process()
    return bboxes

def drawBoxesInImage(img, n, bboxes, color='b'):
    img_with_bboxes = img.copy()
    thickness = 2
    for bbox in bboxes[:n]:
        x1, y1, x2, y2 = bbox
        img_with_bboxes = cv2.rectangle(img_with_bboxes, (x1, y1), (x2, y2), color, thickness)
    return img_with_bboxes

def drawBoxesInImagePredBox(img, n, bboxes, color='b'):
    img_with_bboxes = img.copy()
    thickness = 2
    for bbox in bboxes[:n]:
        x1, y1, x2, y2 = bbox
        img_with_bboxes = cv2.rectangle(img_with_bboxes, (x1, y1), (x2+x1, y2+y1), color, thickness)
    return img_with_bboxes

def parseXMLAndFindObjects(xmlPath):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    objects = []
    listOfValues = []
    for cobject in root.findall("./object/bndbox/"):
        listOfValues.append(cobject.text)

    for i in range(len(listOfValues)):
        if (i % 4 == 0):
            if i != 0:
                objects.append(tuple((x1, x2, y1, y2)))
            x1 = int(listOfValues[i])
        if (i % 4 == 1):
            x2 = int(listOfValues[i])
        if (i % 4 == 2):
            y1 = int(listOfValues[i])
        if (i % 4 == 3):
            y2 = int(listOfValues[i])
    objects.append(tuple((x1, x2, y1, y2)))

    return objects

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxAx1, boxAy1, boxAx2, boxAy2 =boxA
    boxBx1, boxBy1, boxBx2, boxBy2 =boxB
    xA = min(boxAx1, boxBx1)
    yA = min(boxAy1, boxBy1)
    xB = max(boxAx2, boxBx2)
    yB = max(boxAy2, boxBy2)

    # compute the area of intersection rectangle
    if not ((boxAx2 < boxBx1 and boxAy1 < boxBy2) or (boxAx2 < boxBx1 and boxAy1 < boxBy2)):
        interArea= 0  # no overlap
    else:
        interArea = (xA-xB)*(yA-yB)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxAx2 - boxAx1) * (boxAy2 - boxAy1)
    boxBArea = (boxBx2 - boxBx1) * (boxBy2 - boxBy1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def filterBboxesWithIOU(threshold, bboxes, GTimageObjects):
    selectedBox=[]
    for bbox in bboxes:
        for gtbbox in GTimageObjects:
            iou=calculate_iou(gtbbox, bbox)
            #print(iou)
            if iou>threshold:
                selectedBox.append(bbox)
    return selectedBox

''' Get area of a bounding box in xyxy format. '''
def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

''' Get overlap area of two bounding boxes, returns 0 if no overlap. '''
def get_overlap_area(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[2], bbox2[2])
    y1 = max(bbox1[1], bbox2[1])
    y2 = min(bbox1[3], bbox2[3])
    if not (x1 < x2 and y1 < y2):
        return 0 # no overlap
    return (x2 - x1) * (y2 - y1)

''' Evaluate predicted bounding boxes using IOU.
Args:
    pred_bboxes: bounding boxes to be evaluated.
    gt_bboxes: ground truth bounding boxes.
Returns:
    precision: precision of the predicted bboxes.
    recall: recall value of the predicted bboxes.
    correct_bboxes: a list of correct bboxes (with iou > 0.5).
'''
def eval_iou(pred_bboxes, gt_bboxes):
    tp = 0
    correct_bboxes = []
    for gt_bbox in gt_bboxes:
        for pred_bbox in pred_bboxes:
            x, y, w, h=pred_bbox
            corr_pred_bbox=x,y,x+w,y+h
            area_pred_bbox = get_area(list(corr_pred_bbox))
            area_gt_bbox = get_area(list(gt_bbox))
            intersect_area = get_overlap_area(list(corr_pred_bbox), list(gt_bbox))
            union_area = area_pred_bbox + area_gt_bbox - intersect_area
            iou = float(intersect_area) / union_area
            if iou > 0.5:
                tp += 1
                correct_bboxes.append(pred_bbox)
                break
    precision = tp / len(pred_bboxes)
    recall = tp / len(gt_bboxes)
    return precision, recall, correct_bboxes

def loadYMLModel(modelPath):
    import yaml

    model = None
    with open(modelPath, 'r') as stream:
        try:
            model = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return model



