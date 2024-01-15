import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Carregar o modelo treinado
model = load_model('Becca_Stella_Data')

# Rótulos de classe
class_labels = ['stella', 'becca']

def decode_custom_predictions(predictions):
    decoded_predictions = [class_labels[int(round(pred[0]))] for pred in predictions]
    return decoded_predictions

def identify_pet(frame):
    img = cv2.resize(frame, (299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)

    decoded_predictions = decode_custom_predictions(predictions)

    predicted_classes = np.argmax(predictions, axis=1)

    return decoded_predictions, predicted_classes

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        predicted_breed, predicted_classes = identify_pet(frame)

        # Calcular o score de acerto
        true_class = [class_labels.index(label) for label in predicted_breed]
        accuracy = np.mean(np.equal(true_class, predicted_classes)) * 100

        # Adicionar retângulo ao redor do cachorro
        if any(label in ['becca', 'stella'] for label in predicted_breed):
            # Obter a região de interesse (ROI)
            roi_color = frame
            cv2.putText(frame, f"Pet finder: {', '.join(predicted_breed)} Ac: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "not recognized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # windown with results
        cv2.imshow('my pet recognizer', frame)

        # stop program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
