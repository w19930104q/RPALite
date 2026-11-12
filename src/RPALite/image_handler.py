import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
import math
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Default parameters for multi-scale template matching. They can be overridden
# per ImageHandler instance (constructor) or per find_image_matches() call.
DEFAULT_MATCH_THRESHOLD = 0.9
DEFAULT_MATCH_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_NMS_THRESHOLD = 0.3

# A result entry of find_image_matches:
#   (top_left_x, top_left_y, width, height, match_score)
ImageMatch = Tuple[int, int, int, int, float]


class ImageHandler:
    def __init__(self, debug_mode: bool = False, ocr_engine: str = "paddleocr", languages: List[str] = ['en'], debug_image_show_seconds=5,
                 match_threshold: float = DEFAULT_MATCH_THRESHOLD,
                 match_scales: Optional[Tuple[Union[float, Tuple[float, float]], ...]] = None,
                 nms_threshold: float = DEFAULT_NMS_THRESHOLD):
        '''
        Initialize the ImageHandler class. This class use EasyOCR for text OCR. About language codes please check https://www.jaided.ai/easyocr/

        Parameters
        ----------
        match_threshold: float
            Correlation threshold (TM_CCOEFF_NORMED) in the range [0, 1]. Template
            matches below this score are discarded.
        match_scales: tuple
            Configurable search scales. Each entry is either a float for isotropic
            scaling or an (x_scale, y_scale) tuple for independent dual-axis scaling.
            A scale of 1.0 means the template is searched at its original size.
        nms_threshold: float
            Intersection-over-union threshold used by non-maximum suppression.
            Overlapping candidates whose IoU is greater than this value are merged
            into the single highest-scoring candidate.
        '''
        self.debug_mode = debug_mode
        self.languages = languages
        self.debug_image_show_milliseconds = debug_image_show_seconds * 1000
        # OCR handler is initialized lazily (see _get_ocr_handler) so that image
        # matching works in environments where the OCR backend is not installed.
        self.ocr_engine = ocr_engine
        self._ocr_handler = None
        self.match_threshold = self._validate_threshold(match_threshold, DEFAULT_MATCH_THRESHOLD)
        self.match_scales = self._normalize_scales(match_scales if match_scales is not None else DEFAULT_MATCH_SCALES)
        self.nms_threshold = self._validate_threshold(nms_threshold, DEFAULT_NMS_THRESHOLD)

    @property
    def ocr_handler(self):
        '''Lazily initialize and return the OCR handler. Keeps the historical
        ``handler.ocr_handler`` attribute while avoiding loading heavy OCR
        dependencies until an OCR method is actually called.'''
        return self._get_ocr_handler()

    @staticmethod
    def _validate_threshold(value, default: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return default
        return value

    @staticmethod
    def _normalize_scales(scales) -> Tuple[Tuple[float, float], ...]:
        '''Normalize the configurable scales into a tuple of (scale_x, scale_y)
        pairs. Invalid entries are skipped; an empty/invalid input falls back to
        the single identity scale (1.0, 1.0).'''
        normalized = []
        if scales is None:
            scales = DEFAULT_MATCH_SCALES
        if not isinstance(scales, (list, tuple)):
            scales = (scales,)
        for scale in scales:
            try:
                if isinstance(scale, (list, tuple)):
                    if len(scale) < 2:
                        continue
                    sx, sy = float(scale[0]), float(scale[1])
                else:
                    sx = sy = float(scale)
            except (TypeError, ValueError):
                continue
            if math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0:
                pair = (sx, sy)
                if pair not in normalized:
                    normalized.append(pair)
        if not normalized:
            normalized = [(1.0, 1.0)]
        return tuple(normalized)

    def _get_ocr_handler(self):
        if self._ocr_handler is not None:
            return self._ocr_handler
        # Initialize OCR handler based on selected engine
        if self.ocr_engine.lower() == "paddleocr":
            from .paddleocr_handler import PaddleOCRHandler
            self._ocr_handler = PaddleOCRHandler(self.languages, self.debug_mode)
        else:
            from .easyocr_handler import EasyOCRHandler
            self._ocr_handler = EasyOCRHandler(self.languages, self.debug_mode)
        return self._ocr_handler

    @staticmethod
    def _to_gray(image) -> Optional[np.ndarray]:
        '''Convert a PIL image or numpy array to a single-channel uint8 image.
        Returns None for empty or otherwise invalid inputs instead of raising.'''
        if image is None:
            return None
        try:
            if isinstance(image, np.ndarray):
                array = image
            else:
                # PIL image (or anything exposing the PIL array protocol)
                array = np.array(image)
        except Exception as e:
            logger.error(f"Error converting image to array: {e}")
            return None
        if array is None or array.size == 0:
            return None
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            if array.shape[2] == 4:
                return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
            if array.shape[2] == 3:
                return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            if array.shape[2] == 1:
                return array[:, :, 0]
        return None

    @staticmethod
    def _iou(box_a, box_b) -> float:
        '''Intersection over union of two (x, y, width, height) boxes.'''
        ax1, ay1, aw, ah = box_a[:4]
        bx1, by1, bw, bh = box_b[:4]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        union_area = aw * ah + bw * bh - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _non_max_suppression(self, candidates: List[ImageMatch], nms_threshold: float) -> List[ImageMatch]:
        '''Greedily suppress overlapping candidates. Candidates are
        (x, y, width, height, score); the highest-scoring candidate of each
        overlapping group survives, so every visible control yields a single
        unique hit rather than a cluster of overlapping coordinates.'''
        if not candidates:
            return []
        # Highest score first, then top-left position for deterministic ordering.
        ordered = sorted(candidates, key=lambda c: (-float(c[4]), c[1], c[0]))
        kept: List[ImageMatch] = []
        for candidate in ordered:
            if all(self._iou(candidate, existing) <= nms_threshold for existing in kept):
                kept.append(candidate)
        # Keep the score-descending order for callers.
        return kept

    def find_image_matches(self, image, parentImage,
                           threshold: Optional[float] = None,
                           scales=None,
                           nms_threshold: Optional[float] = None) -> List[ImageMatch]:
        '''
        Multi-scale, thresholded and non-maximum-suppressed template matching.

        For every visible occurrence of ``image`` inside ``parentImage`` it returns
        exactly one entry: ``(top_left_x, top_left_y, width, height, match_score)``.
        Coordinates are expressed in the parent image's pixel space (physical
        screenshot pixels); callers convert them back to logical screen
        coordinates themselves.

        Parameters
        ----------
        image: PIL.Image or numpy.ndarray
            The template to search for.
        parentImage: PIL.Image or numpy.ndarray
            The image to search in.
        threshold: float, optional
            Correlation threshold in [0, 1]. Overrides the instance default.
        scales: tuple/list, optional
            Scales to search at. Each entry is a float or an (x_scale, y_scale)
            pair. Overrides the instance default.
        nms_threshold: float, optional
            IoU threshold for non-maximum suppression. Overrides the instance
            default.

        Returns
        -------
        list
            Unique matches sorted by descending score, each a 5-tuple
            ``(x, y, width, height, score)``. Returns an empty list for empty
            images, templates larger than the parent image, or any other edge
            case; it never raises.
        '''
        try:
            threshold = self.match_threshold if threshold is None else self._validate_threshold(threshold, self.match_threshold)
            nms_threshold = self.nms_threshold if nms_threshold is None else self._validate_threshold(nms_threshold, self.nms_threshold)
            search_scales = self.match_scales if scales is None else self._normalize_scales(scales)

            gray_parent = self._to_gray(parentImage)
            gray_template = self._to_gray(image)
            if gray_parent is None or gray_template is None:
                return []

            parent_h, parent_w = gray_parent.shape[:2]
            base_h, base_w = gray_template.shape[:2]
            if parent_h <= 0 or parent_w <= 0 or base_h <= 0 or base_w <= 0:
                return []

            # TM_CCOEFF_NORMED is undefined for zero-variance (constant)
            # templates; some OpenCV builds return NaN, others a misleading
            # perfect 1.0 everywhere. Treat such templates as unmatchable.
            if float(gray_template.std()) == 0.0:
                logger.warning("Template image has no variance (constant color); skipping template matching")
                return []

            if self.debug_mode:
                try:
                    cv2.imshow('shapes', gray_parent)
                    cv2.waitKey(self.debug_image_show_milliseconds)
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            candidates: List[ImageMatch] = []
            for scale_x, scale_y in search_scales:
                scaled_w = max(1, int(round(base_w * scale_x)))
                scaled_h = max(1, int(round(base_h * scale_y)))
                # A template that is not strictly smaller than the parent image
                # cannot be matched; skip this scale rather than raising.
                if scaled_w > parent_w or scaled_h > parent_h:
                    continue

                if scaled_w == base_w and scaled_h == base_h:
                    resized_template = gray_template
                else:
                    resized_template = cv2.resize(gray_template, (scaled_w, scaled_h),
                                                  interpolation=cv2.INTER_AREA)
                if resized_template is None or resized_template.size == 0:
                    continue

                result = cv2.matchTemplate(gray_parent, resized_template, cv2.TM_CCOEFF_NORMED)
                if result is None or result.size == 0:
                    continue
                # TM_CCOEFF_NORMED can produce NaN on degenerate (constant)
                # regions; replace NaNs so threshold comparisons stay safe.
                result = np.nan_to_num(result, nan=-1.0, posinf=1.0, neginf=-1.0)

                ys, xs = np.where(result >= threshold)
                scale_candidates = [
                    (int(x), int(y), scaled_w, scaled_h, float(result[y, x]))
                    for x, y in zip(xs, ys)
                ]
                # Suppress overlapping raw peaks produced at the same scale first.
                candidates.extend(self._non_max_suppression(scale_candidates, nms_threshold))

            # A single real control may be detected at several adjacent scales;
            # suppress cross-scale duplicates so each visible control is unique.
            matches = self._non_max_suppression(candidates, nms_threshold)
            return matches
        except Exception as e:
            logger.error(f"Error in find_image_matches: {e}")
            return []
        finally:
            if self.debug_mode:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def find_image_match_centers(self, image, parentImage,
                                 threshold: Optional[float] = None,
                                 scales=None,
                                 nms_threshold: Optional[float] = None) -> List[Tuple[int, int, float]]:
        '''
        Return the unique center point and hit score of every visible control.

        Returns
        -------
        list
            ``(center_x, center_y, match_score)`` tuples sorted by descending
            score. Returns an empty list when nothing matches or on any edge
            case; it never raises.
        '''
        centers = []
        for x, y, w, h, score in self.find_image_matches(image, parentImage, threshold, scales, nms_threshold):
            centers.append((x + int(round(w / 2.0)), y + int(round(h / 2.0)), score))
        return centers

    def check_point_inide_rect(self, point, rect):  # Should probably be "inside_rect"
        '''Check if a point is in a rect. The point's coordinates are (x, y). The rect's coordinates are (x, y, width, height).'''
        if point is None or rect is None:
            return False
        # If the point is inside the rectangle
        return rect[0] <= point[0] and rect[1] <= point[1] and rect[0] + rect[2] >= point[0] and rect[1] + rect[3] >= point[1]
    
    def check_point_inide_rects(self, point, rects):
        '''Check if a point is in the rects list. The point's coordinates are (x, y). The rects's coordinates are (x, y, width, height).'''
        if point is None or rects is None or len(rects) == 0:
            return False
        
        for rect in rects:
            if self.check_point_inide_rect(point, rect):
                return True
        return False
        
    
    def find_image_location(self, image, parentImage):
        '''
        Find the single best location of ``image`` inside ``parentImage`` using
        configurable multi-scale template matching with a correlation threshold
        and non-maximum suppression.

        Returns
        -------
        tuple
            ``(top_left_x, top_left_y, width, height)`` of the highest-scoring
            match, or None when there is no match / the input is invalid.
        '''
        if parentImage is None or image is None:
            return None
        matches = self.find_image_matches(image, parentImage)
        if not matches:
            return None
        x, y, w, h, score = matches[0]
        return (x, y, w, h)

    def find_all_image_locations(self, image, parentImage):
        '''
        Find every distinct visible occurrence of ``image`` inside
        ``parentImage``. Multi-scale matching is thresholded and non-maximum
        suppressed, so one real control produces exactly one rectangle instead
        of a cluster of overlapping coordinates.

        Returns
        -------
        list
            A list of ``(top_left_x, top_left_y, width, height)`` tuples sorted
            by descending match score. Returns an empty list (never None) when
            nothing matches or on any edge case.
        '''
        if parentImage is None or image is None:
            return []
        return [(x, y, w, h) for x, y, w, h, score in self.find_image_matches(image, parentImage)]

    def find_text_in_array(self, text, arrays, windows):
        '''
        Returns the location information, format is (top_x, top_y, width, height) of the text. This function will first iterate over the arrays, find matched text and check it is in the window. If the matched text is in the window, the function will return the text location. If no matched text found returns None.
        '''
        for r in arrays:
            if text in r[1] and self.check_point_inide_rects(r[0][0], windows):
                return r[0][0]        
        return None
    

    def check_text_and_filter_in_window(self, image, arrays, position, text, filter_args_in_parent, rect=None):
        if filter_args_in_parent is None:
            return True
        windows = self.find_rects_outside_position(image, (position[0][0], position[0][1], position[1][0] - position[0][0], position[2][1] - position[0][1]))
        if windows is None or len(windows) == 0:
            return False
        
        for filter_text in filter_args_in_parent:
            location = self.find_text_in_array(filter_text, arrays, windows)
            if location is not None:
                return True
        return False
    



    def read_text(self, image):
        try:
            img_array = np.array(image)
            
            cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)        
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            low_contrast_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY_INV, 11, 2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            enhanced_borders = cv2.morphologyEx(low_contrast_mask, cv2.MORPH_CLOSE, kernel)
            
            enhanced_borders = cv2.cvtColor(enhanced_borders, cv2.COLOR_GRAY2BGR)
            enhanced_borders = cv2.GaussianBlur(enhanced_borders, (3, 3), 0)
            img_array = cv2.addWeighted(cv_image, 0.9, enhanced_borders, 0.1, 0)

            arr = self._get_ocr_handler().find_texts_in_image(image)
            return arr
        except Exception as e:
            logger.error(f"Error in read_text: {e}")
            return None

    def find_texts_in_rects(self, image, text, filter_args_in_parent=None, rects=None):
        '''
        Returns the location information list, format is ((top_x, top_y, width, height), text) of the text in the image.
        '''
        if rects == None:
            return self.find_texts_in_image(image, text, filter_args_in_parent)
        
        if type(rects) == list and len(rects) > 0:    
            arr = self.read_text(image)
            if arr is None:
                return None
            for rect in rects:
                loc = self.find_texts_in_array_and_rect(text, arr, image, filter_args_in_parent, rect)
                if loc is not None:
                    return loc  
        else:
            return self.find_texts_in_image(image, text, filter_args_in_parent, rects)
  
      
        
    def find_texts_in_array_and_rect(self, text, text_arr, image, filter_args_in_parent=None, rect = None):
        '''
        Returns the location information, format is (top_x, top_y, width, height, ratio) of the text in the image.
        '''
        if text_arr is None:
            return None
            
        results = []
        (position, target_text, best_ratio) = None, None, 0
        for r in text_arr:  
            ratio = SequenceMatcher(None, r[1], text).ratio()
            if  text in r[1]: 
                if ((rect is not None and self.check_point_inide_rect(r[0][0], rect)) or rect is None) and self.check_text_and_filter_in_window(image, text_arr, r[0], r[1], filter_args_in_parent, rect): 
                    position = r[0]
                    target_text = r[1]
                    location = position[0][0], position[0][1], position[1][0]-position[0][0], position[3][1]-position[0][1]

                    results.append( (location, target_text, 1))

            elif ratio > best_ratio and ratio > 0.75 and ((rect is not None and self.check_point_inide_rect(r[0][0], rect)) or rect is None) and self.check_text_and_filter_in_window(image, text_arr, r[0], r[1], filter_args_in_parent, rect):
                best_ratio = ratio
                target_text = r[1]
                position = r[0]
                
                location = position[0][0], position[0][1], position[1][0]-position[0][0], position[3][1]-position[0][1]

                results.append((location, target_text, ratio))

        if len(results) == 0:
            return None
        results = sorted(results, key=lambda l: l[2], reverse=True)
        return results


    def find_texts_in_image(self, image, text, filter_args_in_parent=None, rect=None):
        '''
        Returns the location information, format is (top_x, top_y, width, height) of the text in the image.
        If the text is not found, returns None.
        '''
        
        arr = self.read_text(image)
        
        return self.find_texts_in_array_and_rect(text, arr, image, filter_args_in_parent, rect)


    def validate_inside(self, outside, inside):
        # Check if the rectangle is inside the target rectangle. Paramete rect and target's formats are (x, y, width, height)
        return outside[0] <= inside[0] and outside[1] <= inside[1] and outside[0] + outside[2] >= inside[0] + inside[2] and outside[1] + outside[3] >= inside[1] + inside[3]

    def find_rects_outside_position(self, image, target):
        if target is None:
            return None
            
        try:
            img = np.array(image)
            # converting image into grayscale image 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
          
            # setting threshold of gray image 
            edged = cv2.Canny(gray, 50, 200, apertureSize = 5) 

            # using a findContours() function 
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            
            targets = []
            approx_list = []
           
            # list for storing names of shapes 
            for contour in contours: 
                # Approximate contour to a polygon
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

                # Calculate aspect ratio and bounding box
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)

                    if(h<10 or w<10):
                        #ignore too small shapes
                        continue
            
                    if(self.validate_inside((x, y, w, h), target)):
                        approx_list.append(approx)
                        targets.append((x, y, w, h))

            if(targets is None or len(targets)==0):
                return None
                
            if self.debug_mode:
                img_copy = img.copy()
                cv2.drawContours(img_copy, approx_list, -1, (255, 0, 0), 1)
                cv2.imshow('target', img_copy) 
                cv2.waitKey(self.debug_image_show_milliseconds)
                cv2.destroyAllWindows()
                
            return targets
        except Exception as e:
            logger.error(f"Error in find_rects_outside_position: {e}")
            return None
        finally:
            if self.debug_mode:
                cv2.destroyAllWindows()
    

    def find_window_outside_position(self, image, target):
        if target is None:
            return None
            
        try:
            img = np.array(image)
            # converting image into grayscale image 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
          
            # setting threshold of gray image 
            edged = cv2.Canny(gray, 50, 200, apertureSize = 5) 

            # using a findContours() function 
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            
            i = 0
            dist = 1000000
            a = 0
            target_information = None

            # list for storing names of shapes 
            for contour in contours: 
                # Approximate contour to a polygon
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

                # Calculate aspect ratio and bounding box
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)

                    if(h<10 or w<10):
                        #ignore too small shapes
                        continue

                    dist1 = abs(cv2.pointPolygonTest(contour,(float(target[0]), float(target[1])),True))
            
                    if(self.validate_inside((x, y, w, h), target) and dist1 < dist):
                        dist = dist1
                        target_information = approx, (x, y, w, h)
            
            if(target_information is None):
                return None
                
            if self.debug_mode:
                img_copy = img.copy()
                cv2.drawContours(img_copy, [target_information[0]], -1, (255, 0, 0), 1)
                cv2.imshow('target', img_copy) 
                cv2.waitKey(self.debug_image_show_milliseconds)
                cv2.destroyAllWindows()
       
            return target_information[1]
        except Exception as e:
            logger.error(f"Error in find_window_outside_position: {e}")
            return None
        finally:
            if self.debug_mode:
                cv2.destroyAllWindows()
    
    def find_texts_inside_rect(self, image, target_text, rect):
        if rect is None:
            return None
            
        try:
            text_arr = self.read_text(image)
            if text_arr is None:
                return None
                
            results = []
            (position, target_text) = None, None
            for r in text_arr:  
            
                if (rect is not None and self.check_point_inide_rect(r[0][0], rect) and self.check_point_inide_rect(r[0][2], rect)):
                    position = r[0]
                    target_text = r[1]
                    location = position[0][0], position[0][1], position[1][0]-position[0][0], position[3][1]-position[0][1]

                    results.append((location, target_text))

            if len(results) == 0:
                return None
            return results
        except Exception as e:
            logger.error(f"Error in find_texts_inside_rect: {e}")
            return None
    

    def find_window_near_position(self, image, target):
        return self.find_control_near_position(image, target)


    
    def find_control_near_position(self, image, target, left_or_top_label=False, text_label_match_ration=1.0):
        if target is None:
            return None
            
        try:
            img = np.array(image)
            # converting image into grayscale image 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
          
            # setting threshold of gray image 
            edged = cv2.Canny(gray, 50, 150, apertureSize = 5) 
            # using a findContours() function 
            contours, _ = cv2.findContours(
                edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            
            i = 0
            dist = 1000000
            a = 0
            target_information = None
            distance_threshold = 20
            
            # Use HoughLinesP to detect lines in the image, adjust parameters to detect isolated horizontal lines
            # Hough transform for line detection
            lines = cv2.HoughLinesP(edged, 1, np.pi/180, threshold=100, 
                                    minLineLength=50, maxLineGap=1)
      

            angle_threshold  = 10
            horizontal_lines = []
            other_lines = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        continue  # Ignore zero-length line segments
                    
                    # Calculate line angle (convert to 0-180 degrees)
                    angle = math.degrees(math.atan2(dy, dx)) % 180
                    
                    # Classify as horizontal line segments or other line segments
                    if (angle <= angle_threshold) or (angle >= 180 - angle_threshold):
                        if dx > 100:
                            horizontal_lines.append((x1, y1, x2, y2))
                    else:
                        other_lines.append((x1, y1, x2, y2))
        
            lines = horizontal_lines
            
           
            # Merge lines and contours into a single array and sort
            combined_elements = []
            if lines is not None:
                combined_elements.extend(('line', line) for line in lines)
            combined_elements.extend(('contour', c) for c in contours)
            
            # Unified sorting logic: Sort by y-coordinate in ascending order, for same y, rectangles come first, then by x-coordinate
            combined_elements.sort(key=lambda elem: (
                (elem[1][1] if elem[0] == 'line' else cv2.boundingRect(elem[1])[1]),  # y-coordinate
                1 if elem[0] == 'line' else 0,  # for same y-coordinate, rectangles come first
                elem[1][0] if elem[0] == 'line' else cv2.boundingRect(elem[1])[0]   # x-coordinate
            ))

            # Additional processing logic
            temp_contours = []
            current_y = None
            elements_to_remove = set()
            
            for i, elem in enumerate(combined_elements):
                elem_type, data = elem
                
                # Get y-coordinate of current element
                if elem_type == 'line':
                    y = data[1]  # Take y1 coordinate of the line
                else:
                    y = cv2.boundingRect(data)[1]
                    
                # If it's a contour
                if elem_type == 'contour':
                    if current_y is None or y != current_y:
                        # Start of a new row
                        current_y = y
                        temp_contours = [data]
                    else:
                        # Same row, add to temporary contours
                        temp_contours.append(data)
                        
                # If it's a line
                elif elem_type == 'line':
                    if y != current_y:
                        # Not in current row, skip
                        elements_to_remove.add(i)
                        continue
                        
                    # Check if it's contained by contours
                    x1, y1, x2, y2 = data
                    is_contained = False
                    for contour in temp_contours:
                        # Get contour boundaries
                        x, y, w, h = cv2.boundingRect(contour)
                        # Check if the line is inside the contour
                        if x <= x1 and x + w >= x2 and y <= y1 and y + h >= y2:
                            is_contained = True
                            break
                            
                    if is_contained:
                        elements_to_remove.add(i)
                        
            # Remove elements to exclude
            combined_elements = [elem for i, elem in enumerate(combined_elements) if i not in elements_to_remove]

            # Create image copy for display
            if self.debug_mode:
                display_image = img.copy()
                
                # Draw all detected lines
               
                for elem in combined_elements:
                    elem_type, data = elem
                    
                    # Get unified bounding box information
                    if elem_type == 'line':
                        # Generate rectangle area for line segments (simulate contour)
                        x1, y1, x2, y2 = data
                        # Draw line in red with width 1
                        cv2.line(display_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                cv2.imshow('Lines', display_image) 
                cv2.waitKey(self.debug_image_show_milliseconds)
                cv2.destroyAllWindows()


            for elem in combined_elements:
                elem_type, data = elem
                
                # Get unified bounding box information for elements
                if elem_type == 'line':
                    # Generate rectangle area for line segment (simulate contour)
                    x1, y1, x2, y2 = data
                    x, y, w, h = x1, y1 - 15, x2 - x1, 15
                    # Create simulated approximated contour (rectangle)
                    approx = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
                else:  # contour
                    # Original contour processing logic
                    contour = data
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
                
                # Unified contour logic
                if len(approx) >= 4 and len(approx) <= 8:
                    x, y, w, h = cv2.boundingRect(approx)
                else:
                    continue

                if h < 10 or w < 10:
                    continue

                # Unified area relationship judgment
                target_area = target[2] * target[3]
                rect_area = w * h
                
                if target_area > rect_area:
                    rect_in_target = (x >= target[0] and y >= target[1] and 
                                    x + w <= target[0] + target[2] and 
                                    y + h <= target[1] + target[3])
                    if rect_in_target:
                        continue
                
                # Unified position relationship check
                target_in_rect = (target[0] >= x and target[1] >= y and
                                target[0] + target[2] <= x + w and
                                target[1] + target[3] <= y + h)
                
                if target_in_rect and abs(w - target[2]) < 40:
                    continue

                # Unified distance calculation (using contour method)
                dist_to_left_bottom = abs(cv2.pointPolygonTest(approx, (float(target[0]), float(target[1]+target[3])), True))
                dist_to_right_bottom = abs(cv2.pointPolygonTest(approx, (float(target[0]+target[2]), float(target[1]+target[3])), True))
                dist1 = min(dist_to_left_bottom, dist_to_right_bottom)

                # Unified weight adjustment logic
                multiple = 1

                if (abs(x - target[0]) < target[2] or abs(y - target[1]) < 50) and h/target[3] < 3:
                    multiple *= 0.5

                target_multiple = 1
                # Unified nested relationship check
                if target_information is not None:
                    current_control = target_information[1]
                    target_multiple = target_information[2]
                    if (x >= current_control[0] and y >= current_control[1] and 
                        x + w <= current_control[0] + current_control[2] and 
                        y + h <= current_control[1] + current_control[3]):
                        multiple_for_current_control = (w*h)/(current_control[2]*current_control[3])

                        if multiple_for_current_control < 0.3:
                            multiple_for_current_control = 0.5
                        elif multiple_for_current_control <= 0.85:
                            multiple_for_current_control = 0.85
                        else:
                            multiple_for_current_control = 1

                        multiple *= target_multiple * multiple_for_current_control
                                
                if (dist1 * multiple < dist * target_multiple) and ((not left_or_top_label) or (left_or_top_label and ((target[0] + target[2])/2 < x  or (target[1] + target[3])/2 < y ))):
                    dist = dist1 
                    target_information = (approx, (x, y, w, h), 0.5 if multiple <= 0.5 else 1)
                 
            # displaying the image after drawing contours 
            if(target_information is None):
                return None
                
            if self.debug_mode:
                img_copy = img.copy()
                cv2.drawContours(img_copy, [target_information[0]], -1, (0, 255, 0), 1)
                cv2.imshow('Targets', img_copy) 
                cv2.waitKey(self.debug_image_show_milliseconds)
                cv2.destroyAllWindows()
       
            return target_information[1]
        except Exception as e:
            logger.error(f"Error in find_control_near_position: {e}")
            return None
        finally:
            if self.debug_mode:
                cv2.destroyAllWindows()

