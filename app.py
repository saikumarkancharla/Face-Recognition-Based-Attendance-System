import cv2
import os
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-key'

nimgs = 10
DATA_DIR = 'Attendance'
FACES_DIR = os.path.join('static', 'faces')
MODEL_PATH = os.path.join('static', 'face_recognition_model.pkl')


def today():
    return date.today()


def attendance_filename(date_obj=None):
    date_obj = date_obj or today()
    return os.path.join(DATA_DIR, f'Attendance-{date_obj.strftime("%m_%d_%y")}.csv')


def attendance_display_date(date_obj=None):
    date_obj = date_obj or today()
    return date_obj.strftime("%d-%B-%Y")


def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    if not os.path.isfile(attendance_filename()):
        with open(attendance_filename(), 'w') as f:
            f.write('Name,Roll,Time\n')


ensure_directories()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def totalreg():
    return len([name for name in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, name))])


def extract_faces(img):
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))


def identify_face(facearray):
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = [name for name in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, name))]
    for user in userlist:
        folder = os.path.join(FACES_DIR, user)
        for imgname in os.listdir(folder):
            filepath = os.path.join(folder, imgname)
            img = cv2.imread(filepath)
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    if faces:
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, MODEL_PATH)
    elif os.path.isfile(MODEL_PATH):
        os.remove(MODEL_PATH)


def extract_attendance():
    path = attendance_filename()
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            f.write('Name,Roll,Time\n')
        return [], [], [], 0
    df = pd.read_csv(path)
    if df.empty:
        return [], [], [], 0
    return list(df['Name']), list(df['Roll']), list(df['Time']), len(df)


def add_attendance(name):
    username, userid = name.split('_', 1)
    current_time = datetime.now().strftime("%H:%M:%S")
    path = attendance_filename()
    df = pd.read_csv(path)
    if int(userid) not in df['Roll'].astype(int).tolist():
        with open(path, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def getallusers():
    userlist = [name for name in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, name))]
    userlist.sort()
    names = []
    rolls = []
    for entry in userlist:
        parts = entry.split('_', 1)
        if len(parts) == 2:
            names.append(parts[0])
            rolls.append(parts[1])
        else:
            names.append(entry)
            rolls.append('')
    return userlist, names, rolls, len(userlist)


def deletefolder(duser):
    folder = os.path.join(FACES_DIR, duser)
    if not os.path.isdir(folder):
        return
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
    os.rmdir(folder)




def model_exists():
    return os.path.isfile(MODEL_PATH)


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=attendance_display_date(),
        model_exists=model_exists(),
    )


@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template(
        'listusers.html',
        userlist=userlist,
        names=names,
        rolls=rolls,
        l=l,
        totalreg=totalreg(),
        datetoday2=attendance_display_date(),
        model_exists=model_exists(),
    )


@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    if not duser:
        flash('User selection is missing.', 'warning')
        return redirect(url_for('listusers'))

    deletefolder(duser)
    if totalreg() == 0 and os.path.isfile(MODEL_PATH):
        os.remove(MODEL_PATH)

    train_model()
    flash(f'User "{duser}" deleted successfully.', 'success')
    return redirect(url_for('listusers'))


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()
    if not model_exists():
        flash('No trained model found. Add a new user to enable attendance capture.', 'warning')
        return render_template(
            'home.html',
            names=names,
            rolls=rolls,
            times=times,
            l=l,
            totalreg=totalreg(),
            datetoday2=attendance_display_date(),
            model_exists=model_exists(),
        )

    cap = cv2.VideoCapture(0)
    frame_count = 0
    while frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(
                frame,
                identified_person,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    flash('Attendance capture completed.', 'success')
    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=attendance_display_date(),
        model_exists=model_exists(),
    )


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername'].strip()
    newuserid = request.form['newuserid'].strip()
    if not newusername or not newuserid:
        flash('Please provide both name and ID to register a new user.', 'warning')
        return redirect(url_for('home'))

    userimagefolder = os.path.join(FACES_DIR, f'{newusername}_{newuserid}')
    os.makedirs(userimagefolder, exist_ok=True)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(
                frame,
                f'Images Captured: {i}/{nimgs}',
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
                cv2.LINE_AA,
            )
            if j % 5 == 0 and i < nimgs:
                filename = os.path.join(userimagefolder, f'{newusername}_{i}.jpg')
                cv2.imwrite(filename, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j >= nimgs * 5 or i >= nimgs:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()
    flash(f'User "{newusername}_{newuserid}" added and model retrained successfully.', 'success')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
