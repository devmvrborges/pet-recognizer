import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

#my model
model = load_model('Becca_Stella_Data')

#my pets names
class_labels = ['stella', 'becca']

def decode_custom_predictions(predictions):
    decoded_predictions = [class_labels[int(round(pred[0]))] for pred in predictions]
    return decoded_predictions

def identify_my_pet(frame):
    img = cv2.resize(frame, (299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)

    decoded_predictions = decode_custom_predictions(predictions)

    return decoded_predictions, predictions

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        predicted_breed, predictions = identify_my_pet(frame)

        true_labels = [1 if label in ['becca', 'stella'] else 0 for label in predicted_breed]
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(np.equal(true_labels, predicted_labels)) * 100

        if any(label in ['becca', 'stella'] for label in predicted_breed):
            roi_color = frame
            cv2.putText(frame, f"Pet name: {', '.join(predicted_breed)}   Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "not recognized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('my pet recognizer', frame)

        #for quit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
