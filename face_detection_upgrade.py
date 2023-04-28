import cv2

# Mengaktifkan webcam
cap = cv2.VideoCapture(0)

# Menggunakan classifier wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set ukuran window yang dihasilkan
cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Detection", 640, 480)

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()

    # Mengubah frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocessing gambar
    gray = cv2.equalizeHist(gray)

    # Deteksi wajah pada frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Menandai wajah dengan persegi pada frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Menampilkan frame hasil deteksi wajah
    cv2.imshow('Face Detection', frame)

    # Tombol q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan webcam
cap.release()

# Menutup semua windows
cv2.destroyAllWindows()