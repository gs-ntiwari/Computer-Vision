
def applySelectiveSearch(img, isCombined):
    import cv2 as cv2
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