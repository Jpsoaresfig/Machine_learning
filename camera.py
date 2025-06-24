import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import time

model = load_model('rock_paper_scissors_model.h5')
classes = ['Rock', 'Paper', 'Scissors']

engine = pyttsx3.init()

def falar(texto):
    engine.say(texto)
    engine.runAndWait()

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

ultima_classe = None
ultimo_falado = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_prep = preprocess(img_rgb)

    preds = model.predict(img_prep, verbose=0)
    class_id = np.argmax(preds)
    prob = preds[0][class_id]

    if prob > 0.7:
        agora = time.time()
        if ultima_classe != class_id or (agora - ultimo_falado) > 2:
            falar(classes[class_id])
            ultima_classe = class_id
            ultimo_falado = agora

    cv2.putText(frame, f'{classes[class_id]} ({prob:.2f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Reconhecimento Rock Paper Scissors', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
