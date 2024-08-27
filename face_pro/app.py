
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
app=Flask(__name__)
camera = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
bose_img = face_recognition.load_image_file("images/bose.png")
bose_encoding = face_recognition.face_encodings(bose_img)[0]

einstein_img = face_recognition.load_image_file("images/einstein.png")
einstein_encoding = face_recognition.face_encodings(einstein_img)[0]

newton_img = face_recognition.load_image_file("images/newton.png")
newton_encoding = face_recognition.face_encodings(newton_img)[0]

ramanujan_img = face_recognition.load_image_file("images/ramanujan.png")
ramanujan_encoding = face_recognition.face_encodings(ramanujan_img)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    bose_encoding,
    einstein_encoding,
    newton_encoding,
    ramanujan_encoding
]
known_face_names = [
    "SN Bose",
    "Albert Einstein",
    "Sir Issac Newton",
    "Ramanujan"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
           
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

           
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)            
            face_names = []
            for face_encoding in face_encodings:
               
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
               
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

               
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)