import cv2

detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# camera = cv2.VideoCapture(1)
cap = cv2.VideoCapture("C:/Temp/teste6.mp4")
classificadorOlho = cv2.CascadeClassifier("haarcascade-eye.xml")

'''while (True):
    canectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        imagemface = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 200), 2)
        id, confianca = reconhecedor.predict(imagemface)
        cv2.putText(imagem, str(id), (x, y + (a + 30)), font, 2, (255, 255, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()'''

while (cap.isOpened()):
    ret, frame = cap.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(cinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        # reconhecimento face
        imagemface = cv2.resize(cinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 200), 2)
        id, confianca = reconhecedor.predict(imagemface)
        cv2.putText(frame, str(id), (x, y + (a + 30)), font, 2, (255, 255, 255))

        # reconhecimento olhos
        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

    cv2.imshow("Face", frame)
    if cv2.waitKey(32) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
