from deepface import DeepFace

# Identificar a la persona en el dataset
identify = DeepFace.find(
  img_path = "/known_faces/Cristina.png",
  db_path = "/images/"
  )

print(identify) #No reconoce ninguna imagen, lo que es correcto porque en images no hay una foto de la usuaria

# Verificar que la cara pertenece a la misma persona
verify = DeepFace.verify(
    img1_path="/known_faces/Agus.jpeg",
    img2_path="/images/Agus1.jpeg"
    )

print(verify)