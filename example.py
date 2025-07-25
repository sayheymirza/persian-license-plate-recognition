from license_plate_extractor import extract_digits
from image_classifier import ImageClassifier
import cv2


def detect(image=''):
    try:
        weights_path = 'persian_digit_classifier.pt'
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Alef', 'BE', 'ch', 'd', 'ein', 'f', 'g', 'ghaf', 'ghein', 'h2',
                       'hj', 'j', 'k', 'kh', 'l', 'm', 'n', 'p', 'r', 's', 'sad', 'sh', 't', 'ta', 'th', 'Vav', 'y', 'z', 'za', 'zad', 'zal', 'zh']

        digits, plate = extract_digits(
            image,
            min_area=0.002,
            max_area=0.05,
            debug=False,
            show=False,
        )

        classifier = ImageClassifier(weights_path, class_names)

        # Predict the class
        predicted_digits = []
        for i in range(len(digits)):
            predicted_class = classifier.predict(1.0 - digits[i])
            predicted_digits.append(predicted_class)

        return predicted_digits, plate
    except Exception as e:
        print(e)
        return None, None


def show(original_path, plate_img, effect = 'blur'):
    # Read the original image
    original = cv2.imread(original_path)
    
    # Convert plate image to grayscale if it's not already
    if len(plate_img.shape) == 3:
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        plate_gray = plate_img
    
    # Use template matching to find the plate in the original image
    result = cv2.matchTemplate(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), plate_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Get the dimensions of the plate image
    h, w = plate_img.shape[:2]
    
    # Create white mask for the plate area
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Fill the plate area with white color
    if effect == 'white':
        original[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = [255, 255, 255]
    elif effect == 'blur':
        # Get the region of interest (ROI)
        roi = original[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        # Apply Gaussian blur with odd kernel size
        blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
        # Place the blurred region back
        original[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi

    # Display the result
    cv2.imshow('Result', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


files = ['examples/1.jpg', 'examples/2.jpeg', 'examples/3.webp', 'examples/4.png']

for file in files:
    result = detect(file)

    if result[0] is not None and result[1] is not None:
        show(file, result[1])
        print("Predicted License Plate Number:", " ".join(result[0]))
