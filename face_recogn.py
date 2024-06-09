import cv2
import face_recognition
import os
import numpy as np

# Cargar las imágenes de referencia y obtener sus codificaciones
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces/"

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Cargar cada imagen
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_loc = face_recognition.face_locations(image)[0]
        if face_loc:
            face_encoding = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No se encontró ninguna cara en la imagen {filename}")

# Inicializar la captura de video (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Encontrar todas las caras y sus codificaciones en el frame de video
    face_locations = face_recognition.face_locations(frame)
    
    if face_locations != []:
        for face_location in face_locations:
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_frame_encodings)
            name = "Desconocido"
            print(matches)

            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
            
            # Determinar el mensaje de acceso
            access_message = "Acceso permitido" if name != "Desconocido" else "Acceso denegado"

            cv2.putText(frame, name, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, access_message, (face_location[1], face_location[2] + 20), 2, 0.7, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
