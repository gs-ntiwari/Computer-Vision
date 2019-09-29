import cv2 as cv2
# read image
def readImage(imagePath):
    img = cv2.imread(imagePath)
    img2rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img2rgb


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
        if bbox is None:
            continue
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

''' Get area of a bounding box in xyxy format. '''
def calculateArea(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

''' Get overlap area of two bounding boxes, returns 0 if no overlap. '''
def calculateOverlapArea(bbox1, bbox2):
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
def calculateIOU(pred_bboxes, gt_bboxes, threshold=0.5):
    tp = 0
    correct_bboxes = []
    maxOverlap_bboxes=[]
    mabo=0
    for gt_bbox in gt_bboxes:
        maxOverlap=0
        ctp = 0
        maxOverlapBox=None
        for pred_bbox in pred_bboxes:
            x, y, w, h=pred_bbox
            corr_pred_bbox=x,y,x+w,y+h
            area_pred_bbox = calculateArea(list(corr_pred_bbox))
            area_gt_bbox = calculateArea(list(gt_bbox))
            intersect_area = calculateOverlapArea(list(corr_pred_bbox), list(gt_bbox))
            union_area = area_pred_bbox + area_gt_bbox - intersect_area
            iou = float(intersect_area) / union_area
            if(maxOverlap<iou):
                maxOverlapBox=pred_bbox
                maxOverlap=iou
            if iou > threshold:
                ctp= 1
                correct_bboxes.append(pred_bbox)
        tp+=ctp
        maxOverlap_bboxes.append(maxOverlapBox)
        mabo+=maxOverlap
    mabo/=len(gt_bboxes)
    precision = tp / len(pred_bboxes)
    recall = tp / len(gt_bboxes)
    return mabo, precision*100, recall*100, correct_bboxes, maxOverlap_bboxes

def loadYMLModel(modelPath):
    import yaml

    model = None
    with open(modelPath, 'r') as stream:
        try:
            model = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return model



