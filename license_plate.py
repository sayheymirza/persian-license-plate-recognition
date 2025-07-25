from license_plate_extractor import extract_digits
from image_classifier import ImageClassifier

weights_path = 'persian_digit_classifier.pt'
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Alef', 'BE', 'ch', 'd', 'ein', 'f', 'g', 'ghaf', 'ghein', 'h2',
               'hj', 'j', 'k', 'kh', 'l', 'm', 'n', 'p', 'r', 's', 'sad', 'sh', 't', 'ta', 'th', 'Vav', 'y', 'z', 'za', 'zad', 'zal', 'zh']


def detect(image=''):
    try:

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
