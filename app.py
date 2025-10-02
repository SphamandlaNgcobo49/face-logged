from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, Response
import sqlite3, smtplib, ssl, secrets, string, os, json, pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import time
import base64
import cv2
import numpy as np
import threading
import random

app = Flask(__name__)
app.secret_key = "supersecretkey123"
app.config['UPLOAD_FOLDER'] = 'static/face_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------
# Hardcoded Admin
# ---------------------------
ADMIN_USER = {"username": "admin", "password": "password123"}

# Global variables for face registration and continuous attendance
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
attendance_sessions = {}

# ---------------------------
# Database Setup & Migration
# ---------------------------
def update_database_schema():
    """Update existing database with new columns if they don't exist"""
    conn = sqlite3.connect("face_logged.db", check_same_thread=False)
    c = conn.cursor()
    
    # Check if new columns exist and add them if they don't
    try:
        # Check for Face_Registered column
        c.execute("SELECT Face_Registered FROM Students LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        c.execute("ALTER TABLE Students ADD COLUMN Face_Registered BOOLEAN DEFAULT FALSE")
        print("✅ Added Face_Registered column to Students table")
    
    try:
        # Check for Face_Image_Path column
        c.execute("SELECT Face_Image_Path FROM Students LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        c.execute("ALTER TABLE Students ADD COLUMN Face_Image_Path TEXT")
        print("✅ Added Face_Image_Path column to Students table")

    try:
        c.execute("SELECT Session_Status FROM LectureSessions LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE LectureSessions ADD COLUMN Session_Status TEXT DEFAULT 'scheduled'")
        print("✅ Added Session_Status column to LectureSessions table")
    
    # Check for Modules table columns
    try:
        c.execute("SELECT Attendance_Threshold FROM Modules LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE Modules ADD COLUMN Attendance_Threshold INTEGER DEFAULT 75")
        print("✅ Added Attendance_Threshold column to Modules table")
    
    try:
        c.execute("SELECT Number_of_Classes FROM Modules LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE Modules ADD COLUMN Number_of_Classes INTEGER DEFAULT 0")
        print("✅ Added Number_of_Classes column to Modules table")
    
    # Check if FaceEncodings table exists
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='FaceEncodings'
    """)
    if not c.fetchone():
        c.execute("""
            CREATE TABLE FaceEncodings (
                Encoding_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Student_ID INTEGER UNIQUE,
                Encoding_Data BLOB,
                Date_Created TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (Student_ID) REFERENCES Students(Student_ID)
            )
        """)
        print("✅ Created FaceEncodings table")
    
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect("face_logged.db", check_same_thread=False)
    c = conn.cursor()
    
    # Lecturers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS Lecturers (
            Lecturer_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            First_Name TEXT NOT NULL,
            Last_Name TEXT NOT NULL,
            Email TEXT NOT NULL UNIQUE,
            Phone_Number TEXT,
            Department TEXT,
            Username TEXT UNIQUE,
            Password TEXT,
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Modules table
    c.execute("""
        CREATE TABLE IF NOT EXISTS Modules (
            Module_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Module_Code TEXT NOT NULL UNIQUE,
            Module_Name TEXT NOT NULL,
            Description TEXT,
            Credits INTEGER,
            Semester TEXT,
            Department TEXT,
            Lecturer_ID INTEGER,
            Number_of_Classes INTEGER DEFAULT 0,
            Attendance_Threshold INTEGER DEFAULT 75,
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (Lecturer_ID) REFERENCES Lecturers(Lecturer_ID)
        )
    """)
    
    # Students table (with new columns)
    c.execute("""
        CREATE TABLE IF NOT EXISTS Students (
            Student_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            First_Name TEXT NOT NULL,
            Last_Name TEXT NOT NULL,
            Student_Number TEXT NOT NULL UNIQUE,
            Email TEXT NOT NULL UNIQUE,
            Password TEXT NOT NULL,
            Face_Registered BOOLEAN DEFAULT FALSE,
            Face_Image_Path TEXT,
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # StudentModules table
    c.execute("""
        CREATE TABLE IF NOT EXISTS StudentModules (
            Student_ID INTEGER,
            Module_ID INTEGER,
            FOREIGN KEY (Student_ID) REFERENCES Students(Student_ID),
            FOREIGN KEY (Module_ID) REFERENCES Modules(Module_ID),
            PRIMARY KEY (Student_ID, Module_ID)
        )
    """)
    
    # Lecture Sessions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS LectureSessions (
            Session_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Module_ID INTEGER,
            Session_Date TEXT NOT NULL,
            Start_Time TEXT,
            End_Time TEXT,
            Topic TEXT,
            Session_Status TEXT DEFAULT 'scheduled',
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (Module_ID) REFERENCES Modules(Module_ID)
        )
    """)
    
    # Attendance table
    c.execute("""
        CREATE TABLE IF NOT EXISTS Attendance (
            Attendance_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Student_ID INTEGER,
            Module_ID INTEGER,
            Session_ID INTEGER,
            Attendance_Date TEXT NOT NULL,
            Attendance_Time TEXT NOT NULL,
            Status TEXT DEFAULT 'Present',
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (Student_ID) REFERENCES Students(Student_ID),
            FOREIGN KEY (Module_ID) REFERENCES Modules(Module_ID),
            FOREIGN KEY (Session_ID) REFERENCES LectureSessions(Session_ID),
            UNIQUE(Student_ID, Session_ID)
        )
    """)
    
    # Lecture Ratings table
    c.execute("""
        CREATE TABLE IF NOT EXISTS LectureRatings (
            Rating_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Student_ID INTEGER,
            Module_ID INTEGER,
            Session_ID INTEGER,
            Rating INTEGER CHECK (Rating >= 1 AND Rating <= 5),
            Feedback TEXT,
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (Student_ID) REFERENCES Students(Student_ID),
            FOREIGN KEY (Module_ID) REFERENCES Modules(Module_ID),
            FOREIGN KEY (Session_ID) REFERENCES LectureSessions(Session_ID),
            UNIQUE(Student_ID, Session_ID)
        )
    """)
    
    # Face Encodings table (for advanced recognition)
    c.execute("""
        CREATE TABLE IF NOT EXISTS FaceEncodings (
            Encoding_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Student_ID INTEGER UNIQUE,
            Encoding_Data BLOB,
            Date_Created TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (Student_ID) REFERENCES Students(Student_ID)
        )
    """)
    
    conn.commit()
    conn.close()
    
    # Update existing database schema
    update_database_schema()

# Initialize database
init_db()

# ---------------------------
# Improved Email Config
# ---------------------------
SENDER_EMAIL = "mpilozikhali72@gmail.com"
SENDER_PASSWORD = "etvn fogt ndvv lcse"

def send_email(receiver_email, subject, message_text):
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = SENDER_EMAIL
        message["To"] = receiver_email

        # Create both plain and HTML versions
        part1 = MIMEText(message_text, "plain")
        message.attach(part1)

        # Try multiple email servers
        servers = [
            ("smtp.gmail.com", 587),  # TLS
            ("smtp.gmail.com", 465),  # SSL
        ]
        
        email_sent = False
        for server, port in servers:
            try:
                if port == 587:  # TLS
                    server = smtplib.SMTP(server, port)
                    server.starttls(context=ssl.create_default_context())
                else:  # SSL
                    server = smtplib.SMTP_SSL(server, port, context=ssl.create_default_context())
                
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, receiver_email, message.as_string())
                server.quit()
                email_sent = True
                break
            except Exception as e:
                print(f"Failed to send via {server}:{port} - {e}")
                continue
        
        if email_sent:
            print(f"✅ Email sent to: {receiver_email}")
            return True
        else:
            print(f"❌ All email servers failed for: {receiver_email}")
            return False
            
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

def send_login_email(receiver_email, username, password):
    subject = "Your Face Logged Account Details"
    message = f"""
Hello,

Your account has been created.

Username: {username}
Password: {password}

Please log in and change your password after first login.

Regards,
Face Logged Admin
"""
    return send_email(receiver_email, subject, message)

# ---------------------------
# Enhanced Face Recognition Functions
# ---------------------------
class AttendanceSession:
    def __init__(self, module_id, lecturer_id):
        self.module_id = module_id
        self.lecturer_id = lecturer_id
        self.is_active = False
        self.detected_students = set()
        self.session_id = None
        self.start_time = None
        
    def start_session(self):
        self.is_active = True
        self.start_time = datetime.now()
        self.detected_students.clear()
        
        # Create a new lecture session in database
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        c.execute("""
            INSERT INTO LectureSessions (Module_ID, Session_Date, Start_Time, Topic, Session_Status)
            VALUES (?, ?, ?, ?, 'active')
        """, (self.module_id, self.start_time.strftime("%Y-%m-%d"), 
              self.start_time.strftime("%H:%M:%S"), "Auto-detected Session"))
        self.session_id = c.lastrowid
        conn.commit()
        conn.close()
        print(f"✅ Attendance session started: {self.session_id}")
        
    def stop_session(self):
        self.is_active = False
        # Update session status
        if self.session_id:
            conn = sqlite3.connect("face_logged.db")
            c = conn.cursor()
            c.execute("""
                UPDATE LectureSessions 
                SET Session_Status = 'completed', End_Time = ?
                WHERE Session_ID = ?
            """, (datetime.now().strftime("%H:%M:%S"), self.session_id))
            conn.commit()
            conn.close()
            print(f"✅ Attendance session completed: {self.session_id}")

def generate_frames_with_face_detection():
    """Generate frames with face detection for registration page"""
    camera = cv2.VideoCapture(0)
    
    # Set camera resolution for better quality
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            # If camera fails, show error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error - Please check connection", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with better parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add status information
            if len(faces) == 0:
                status = "No Face Detected - Please position face in frame"
                color = (0, 0, 255)  # Red
            elif len(faces) == 1:
                status = "Face Detected - Ready for Registration"
                color = (0, 255, 0)  # Green
            else:
                status = "Multiple Faces Detected - Ensure only one face"
                color = (0, 165, 255)  # Orange
            
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_with_continuous_recognition(module_id, lecturer_id):
    """Generate frames with continuous face recognition for attendance"""
    camera = cv2.VideoCapture(0)
    
    # Initialize session if not exists
    session_key = f"{module_id}_{lecturer_id}"
    if session_key not in attendance_sessions:
        attendance_sessions[session_key] = AttendanceSession(module_id, lecturer_id)
    
    session = attendance_sessions[session_key]
    
    # Load known student faces for this module
    known_students = load_student_faces_for_module(module_id)
    
    # Track last detection time for each student to avoid multiple detections in short time
    last_detection_time = {}
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Always show face detection, but only mark attendance when session is active
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Simulate face recognition (in real system, use face encodings)
            student_id = simulate_face_recognition(frame, known_students)
            
            # Draw rectangle and info
            color = (0, 255, 0) if student_id else (0, 0, 255)  # Green if recognized, red if unknown
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if student_id:
                student_name = get_student_name(student_id)
                status_text = f"Student: {student_name}"
                
                # Mark attendance if session is active and not recently detected
                if (session.is_active and student_id not in session.detected_students and 
                    (student_id not in last_detection_time or 
                     current_time - last_detection_time[student_id] > 10)):  # 10 seconds cooldown
                    
                    if mark_attendance_automatically(student_id, module_id, session.session_id):
                        session.detected_students.add(student_id)
                        last_detection_time[student_id] = current_time
                        print(f"✅ Attendance marked for student {student_id}")
                
            else:
                status_text = "Unknown Face"
            
            cv2.putText(frame, status_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add session status to frame
        status_text = "ACTIVE - Recording Attendance" if session.is_active else "READY - Press Start"
        status_color = (0, 255, 0) if session.is_active else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Detected: {len(session.detected_students)} students", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Session: {session.session_id if session.session_id else 'Not started'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def load_student_faces_for_module(module_id):
    """Load student faces registered for a specific module"""
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("""
        SELECT s.Student_ID, s.Face_Image_Path 
        FROM Students s
        JOIN StudentModules sm ON s.Student_ID = sm.Student_ID
        WHERE sm.Module_ID = ? AND s.Face_Registered = TRUE
    """, (module_id,))
    students = c.fetchall()
    conn.close()
    return students

def get_student_name(student_id):
    """Get student name by ID"""
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT First_Name, Last_Name FROM Students WHERE Student_ID = ?", (student_id,))
    student = c.fetchone()
    conn.close()
    if student:
        return f"{student[0]} {student[1]}"
    return f"Student {student_id}"

def simulate_face_recognition(frame, known_students):
    """Simulate face recognition (replace with actual face recognition)"""
    # In a real system, you would:
    # 1. Extract face encoding from frame
    # 2. Compare with encodings of known students
    # 3. Return student_id if match found
    
    # For demo, return a random student ID from known students with 80% probability
    if known_students and random.random() < 0.8:
        return random.choice(known_students)[0]
    return None

def mark_attendance_automatically(student_id, module_id, session_id):
    """Mark attendance for a recognized student"""
    try:
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        
        # Check if already marked for this session
        c.execute("SELECT * FROM Attendance WHERE Student_ID = ? AND Session_ID = ?", 
                 (student_id, session_id))
        if not c.fetchone():
            # Mark attendance
            today = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            
            c.execute("""
                INSERT INTO Attendance (Student_ID, Module_ID, Session_ID, Attendance_Date, Attendance_Time, Status)
                VALUES (?, ?, ?, ?, ?, 'Present')
            """, (student_id, module_id, session_id, today, current_time))
            
            conn.commit()
            conn.close()
            return True
    
    except Exception as e:
        print(f"Error marking attendance: {e}")
    
    return False

def capture_face_image(student_id):
    """Capture and save face image"""
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return False, "Camera not available"
        
        # Allow camera to warm up
        time.sleep(2)
        
        # Capture frame
        success, frame = camera.read()
        camera.release()
        
        if not success:
            return False, "Failed to capture image"
        
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, "No face detected. Please ensure your face is clearly visible."
        elif len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only one face is visible."
        
        # Save the image
        filename = f"student_{student_id}_{int(time.time())}.jpg"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        cv2.imwrite(save_path, frame)
        
        # Update database
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        c.execute("UPDATE Students SET Face_Registered = TRUE, Face_Image_Path = ? WHERE Student_ID = ?", 
                  (save_path, student_id))
        conn.commit()
        conn.close()
        
        return True, "Face registered successfully!"
        
    except Exception as e:
        return False, f"Error during face registration: {str(e)}"

def check_face_registered(student_id):
    """Check if student has already registered face"""
    try:
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        c.execute("SELECT Face_Registered FROM Students WHERE Student_ID = ?", (student_id,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else False
    except sqlite3.OperationalError:
        # If column doesn't exist yet, return False
        return False

# ---------------------------
# Utility Functions
# ---------------------------
def get_student_attendance_percentage(student_id, module_id):
    """Calculate attendance percentage for a student in a module"""
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("""
        SELECT COUNT(*) FROM LectureSessions 
        WHERE Module_ID = ? AND Session_Date <= date('now')
    """, (module_id,))
    total_sessions = c.fetchone()[0]
    
    if total_sessions == 0:
        return 100
    
    c.execute("""
        SELECT COUNT(*) FROM Attendance 
        WHERE Student_ID = ? AND Module_ID = ? AND Status = 'Present'
    """, (student_id, module_id))
    attended_sessions = c.fetchone()[0]
    
    percentage = (attended_sessions / total_sessions) * 100
    conn.close()
    
    return round(percentage, 2)

def validate_email(email):
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Basic phone number validation"""
    import re
    pattern = r'^\+?[\d\s\-\(\)]{10,}$'
    return re.match(pattern, phone) is not None

def get_modules_for_student(student_id):
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("""
        SELECT m.Module_ID, m.Module_Code, m.Module_Name
        FROM StudentModules sm
        JOIN Modules m ON sm.Module_ID = m.Module_ID
        WHERE sm.Student_ID = ?
    """, (student_id,))
    modules = c.fetchall()
    conn.close()
    return modules

def check_attendance_thresholds():
    """Check and send alerts for students below attendance threshold"""
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    # Get all students and their modules
    c.execute("""
        SELECT s.Student_ID, s.Email, s.First_Name, sm.Module_ID, m.Module_Name, m.Attendance_Threshold
        FROM Students s
        JOIN StudentModules sm ON s.Student_ID = sm.Student_ID
        JOIN Modules m ON sm.Module_ID = m.Module_ID
    """)
    
    students_modules = c.fetchall()
    
    for student_id, email, first_name, module_id, module_name, threshold in students_modules:
        attendance_percentage = get_student_attendance_percentage(student_id, module_id)
        
        if attendance_percentage < threshold:
            subject = f"Low Attendance Alert - {module_name}"
            message = f"""
Dear {first_name},

Your attendance for {module_name} is currently {attendance_percentage}%, which is below the required threshold of {threshold}%.

Please ensure you attend future lectures to maintain satisfactory attendance.

Regards,
University Administration
"""
            if send_email(email, subject, message):
                print(f"✅ Low attendance alert sent to {email}")
    
    conn.close()

def check_module_attendance_thresholds(module_id):
    """Check attendance thresholds for a specific module and send alerts"""
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    # Get module threshold
    c.execute("SELECT Attendance_Threshold, Module_Name FROM Modules WHERE Module_ID = ?", (module_id,))
    module_info = c.fetchone()
    threshold = module_info[0] if module_info else 75
    module_name = module_info[1] if module_info else "Unknown Module"
    
    # Get all students in this module
    c.execute("""
        SELECT s.Student_ID, s.Email, s.First_Name, s.Last_Name
        FROM Students s
        JOIN StudentModules sm ON s.Student_ID = sm.Student_ID
        WHERE sm.Module_ID = ?
    """, (module_id,))
    
    students = c.fetchall()
    
    alerts_sent = 0
    for student_id, email, first_name, last_name in students:
        attendance_percentage = get_student_attendance_percentage(student_id, module_id)
        
        if attendance_percentage < threshold:
            subject = f"Low Attendance Alert - {module_name}"
            message = f"""
Dear {first_name},

Your attendance for {module_name} is currently {attendance_percentage}%, 
which is below the required threshold of {threshold}%.

Please ensure you attend future lectures to maintain satisfactory attendance.

Regards,
University Administration
"""
            if send_email(email, subject, message):
                alerts_sent += 1
                print(f"⚠️ Low attendance alert sent to {email}")
    
    conn.close()
    return alerts_sent

# ---------------------------
# Routes - Authentication & Core
# ---------------------------
@app.route("/")
def home():
    if request.args.get("loaded") == "true":
        return render_template("index.html")
    return render_template("loader.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            return render_template("login.html", error="Please enter both username and password")

        # Admin login
        if username == ADMIN_USER["username"] and password == ADMIN_USER["password"]:
            session["role"] = "admin"
            session["logged_in"] = True
            session["user_id"] = "admin"
            return redirect(url_for("admin_dashboard"))

        # Lecturer login
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        c.execute("SELECT * FROM Lecturers WHERE Username = ? AND Password = ?", (username, password))
        lecturer = c.fetchone()
        conn.close()

        if lecturer:
            session["role"] = "lecturer"
            session["logged_in"] = True
            session["user_id"] = lecturer[0]
            session["lecturer_id"] = lecturer[0]
            return redirect(url_for("lecturer_dashboard"))

        # Student login
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        c.execute("SELECT * FROM Students WHERE (Student_Number = ? OR Email = ?) AND Password = ?", 
                 (username, username, password))
        student = c.fetchone()
        conn.close()

        if student:
            session["role"] = "student"
            session["logged_in"] = True
            session["user_id"] = student[0]
            session["student_id"] = student[0]
            return redirect(url_for("student_dashboard"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------
# Routes - Admin
# ---------------------------
@app.route("/admin-dashboard")
def admin_dashboard():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM Students")
    student_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM Lecturers")
    lecturer_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM Modules")
    module_count = c.fetchone()[0]
    
    c.execute("""
        SELECT 'Student' as type, First_Name, Last_Name, Date_Created 
        FROM Students 
        ORDER BY Date_Created DESC LIMIT 5
    """)
    recent_students = c.fetchall()
    
    c.execute("""
        SELECT 'Lecturer' as type, First_Name, Last_Name, Date_Created 
        FROM Lecturers 
        ORDER BY Date_Created DESC LIMIT 5
    """)
    recent_lecturers = c.fetchall()
    
    conn.close()
    
    return render_template("admin_dashboard.html", 
                         student_count=student_count,
                         lecturer_count=lecturer_count,
                         module_count=module_count,
                         recent_students=recent_students,
                         recent_lecturers=recent_lecturers)

@app.route("/lecturers")
def lecturers():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT * FROM Lecturers")
    lecturers_list = c.fetchall()
    conn.close()
    return render_template("lecturers.html", lecturers=lecturers_list)

@app.route("/add-lecturer", methods=["GET", "POST"])
def add_lecturer():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        department = request.form.get("department")
        username = request.form.get("username")

        if not all([first_name, last_name, email, department, username]):
            flash("Please fill in all required fields", "danger")
            return render_template("add_lecturer.html")
        
        if not validate_email(email):
            flash("Please enter a valid email address", "danger")
            return render_template("add_lecturer.html")

        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(10))

        try:
            conn = sqlite3.connect("face_logged.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO Lecturers (First_Name, Last_Name, Email, Phone_Number, Department, Username, Password)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (first_name, last_name, email, phone, department, username, password))
            conn.commit()
            conn.close()

            if send_login_email(email, username, password):
                flash("Lecturer added successfully and login details sent via email!", "success")
            else:
                flash("Lecturer added, but email could not be sent.", "warning")
            return redirect(url_for("lecturers"))
        
        except sqlite3.IntegrityError:
            flash("Username or email already exists", "danger")
            return render_template("add_lecturer.html")

    return render_template("add_lecturer.html")

@app.route("/edit-lecturer/<int:lecturer_id>", methods=["GET", "POST"])
def edit_lecturer(lecturer_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        department = request.form.get("department")
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            if password:
                c.execute("""
                    UPDATE Lecturers 
                    SET First_Name=?, Last_Name=?, Email=?, Phone_Number=?, Department=?, Username=?, Password=?
                    WHERE Lecturer_ID=?
                """, (first_name, last_name, email, phone, department, username, password, lecturer_id))
            else:
                c.execute("""
                    UPDATE Lecturers 
                    SET First_Name=?, Last_Name=?, Email=?, Phone_Number=?, Department=?, Username=?
                    WHERE Lecturer_ID=?
                """, (first_name, last_name, email, phone, department, username, lecturer_id))
            
            conn.commit()
            flash("Lecturer updated successfully!", "success")
            return redirect(url_for("lecturers"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists", "danger")
    
    c.execute("SELECT * FROM Lecturers WHERE Lecturer_ID = ?", (lecturer_id,))
    lecturer = c.fetchone()
    conn.close()
    
    return render_template("edit_lecturer.html", lecturer=lecturer)

@app.route("/delete-lecturer/<int:lecturer_id>", methods=["POST"])
def delete_lecturer(lecturer_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("DELETE FROM Lecturers WHERE Lecturer_ID = ?", (lecturer_id,))
    conn.commit()
    conn.close()
    flash("Lecturer deleted successfully!", "success")
    return redirect(url_for("lecturers"))

@app.route("/add-module", methods=["GET", "POST"])
def add_module():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT * FROM Lecturers")
    lecturers = c.fetchall()
    conn.close()

    if request.method == "POST":
        module_code = request.form.get("module_code")
        module_name = request.form.get("module_name")
        description = request.form.get("description")
        credits = request.form.get("credits")
        semester = request.form.get("semester")
        department = request.form.get("department")
        lecturer_id = request.form.get("lecturer_id")
        number_of_classes = request.form.get("number_of_classes")
        attendance_threshold = request.form.get("attendance_threshold", 75)

        if not all([module_code, module_name, credits, semester, department, lecturer_id, number_of_classes]):
            flash("Please fill in all required fields", "danger")
            return render_template("add_module.html", lecturers=lecturers)

        try:
            conn = sqlite3.connect("face_logged.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO Modules
                (Module_Code, Module_Name, Description, Credits, Semester, Department, Lecturer_ID, Number_of_Classes, Attendance_Threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (module_code, module_name, description, credits, semester, department, lecturer_id, number_of_classes, attendance_threshold))
            conn.commit()
            conn.close()
            flash("Module added successfully!", "success")
            return redirect(url_for("modules"))
        
        except sqlite3.IntegrityError:
            flash("Module code already exists", "danger")
            return render_template("add_module.html", lecturers=lecturers)

    return render_template("add_module.html", lecturers=lecturers)

@app.route("/edit-module/<int:module_id>", methods=["GET", "POST"])
def edit_module(module_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("SELECT * FROM Lecturers")
    lecturers = c.fetchall()
    
    if request.method == "POST":
        module_code = request.form.get("module_code")
        module_name = request.form.get("module_name")
        description = request.form.get("description")
        credits = request.form.get("credits")
        semester = request.form.get("semester")
        department = request.form.get("department")
        lecturer_id = request.form.get("lecturer_id")
        number_of_classes = request.form.get("number_of_classes")
        attendance_threshold = request.form.get("attendance_threshold", 75)

        try:
            c.execute("""
                UPDATE Modules 
                SET Module_Code=?, Module_Name=?, Description=?, Credits=?, 
                    Semester=?, Department=?, Lecturer_ID=?, Number_of_Classes=?, Attendance_Threshold=?
                WHERE Module_ID=?
            """, (module_code, module_name, description, credits, semester, 
                  department, lecturer_id, number_of_classes, attendance_threshold, module_id))
            conn.commit()
            flash("Module updated successfully!", "success")
            return redirect(url_for("modules"))
        except sqlite3.IntegrityError:
            flash("Module code already exists", "danger")
    
    c.execute("SELECT * FROM Modules WHERE Module_ID = ?", (module_id,))
    module = c.fetchone()
    conn.close()
    
    return render_template("edit_module.html", module=module, lecturers=lecturers)

@app.route("/delete-module/<int:module_id>", methods=["POST"])
def delete_module(module_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    # Delete module and related records
    c.execute("DELETE FROM StudentModules WHERE Module_ID = ?", (module_id,))
    c.execute("DELETE FROM Attendance WHERE Module_ID = ?", (module_id,))
    c.execute("DELETE FROM LectureSessions WHERE Module_ID = ?", (module_id,))
    c.execute("DELETE FROM LectureRatings WHERE Module_ID = ?", (module_id,))
    c.execute("DELETE FROM Modules WHERE Module_ID = ?", (module_id,))
    
    conn.commit()
    conn.close()
    flash("Module deleted successfully!", "success")
    return redirect(url_for("modules"))

@app.route("/modules")
def modules():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("""
        SELECT m.Module_ID, m.Module_Code, m.Module_Name, m.Description, m.Credits,
               m.Semester, m.Department, l.First_Name, l.Last_Name, m.Number_of_Classes, m.Attendance_Threshold
        FROM Modules m
        LEFT JOIN Lecturers l ON m.Lecturer_ID = l.Lecturer_ID
    """)
    modules_list = c.fetchall()
    conn.close()
    return render_template("modules.html", modules=modules_list)

@app.route("/students")
def students():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT * FROM Students")
    students_list = c.fetchall()
    conn.close()

    return render_template("students.html", students=students_list, get_modules_for_student=get_modules_for_student)

@app.route("/add-student", methods=["GET", "POST"])
def add_student():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT Module_ID, Module_Code, Module_Name FROM Modules")
    modules = c.fetchall()
    conn.close()

    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        student_number = request.form.get("student_number")
        email = request.form.get("email")
        password = request.form.get("password")
        selected_modules = request.form.getlist("modules")

        if not all([first_name, last_name, student_number, email, password]):
            flash("Please fill in all required fields", "danger")
            return render_template("add_student.html", modules=modules)
        
        if not validate_email(email):
            flash("Please enter a valid email address", "danger")
            return render_template("add_student.html", modules=modules)

        try:
            conn = sqlite3.connect("face_logged.db")
            c = conn.cursor()

            c.execute("SELECT * FROM Students WHERE Email = ? OR Student_Number = ?", (email, student_number))
            existing = c.fetchone()

            if existing:
                conn.close()
                flash("A student with this email or student number already exists!", "danger")
                return redirect(url_for("add_student"))

            c.execute("""
                INSERT INTO Students (First_Name, Last_Name, Student_Number, Email, Password)
                VALUES (?, ?, ?, ?, ?)
            """, (first_name, last_name, student_number, email, password))
            student_id = c.lastrowid

            for module_id in selected_modules:
                c.execute("""
                    INSERT INTO StudentModules (Student_ID, Module_ID)
                    VALUES (?, ?)
                """, (student_id, module_id))

            conn.commit()
            conn.close()

            if send_login_email(email, student_number, password):
                flash("Student added successfully and login details sent via email!", "success")
            else:
                flash("Student added, but email could not be sent.", "warning")

            return redirect(url_for("students"))
        
        except sqlite3.IntegrityError:
            flash("Student number or email already exists", "danger")
            return render_template("add_student.html", modules=modules)

    return render_template("add_student.html", modules=modules)

@app.route("/edit-student/<int:student_id>", methods=["GET", "POST"])
def edit_student(student_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        student_number = request.form.get("student_number")
        email = request.form.get("email")
        password = request.form.get("password")
        selected_modules = request.form.getlist("modules")

        try:
            # Update student basic info
            if password:
                c.execute("""
                    UPDATE Students 
                    SET First_Name=?, Last_Name=?, Student_Number=?, Email=?, Password=?
                    WHERE Student_ID=?
                """, (first_name, last_name, student_number, email, password, student_id))
            else:
                c.execute("""
                    UPDATE Students 
                    SET First_Name=?, Last_Name=?, Student_Number=?, Email=?
                    WHERE Student_ID=?
                """, (first_name, last_name, student_number, email, student_id))
            
            # Update modules
            c.execute("DELETE FROM StudentModules WHERE Student_ID = ?", (student_id,))
            for module_id in selected_modules:
                c.execute("INSERT INTO StudentModules (Student_ID, Module_ID) VALUES (?, ?)", 
                         (student_id, module_id))
            
            conn.commit()
            flash("Student updated successfully!", "success")
            return redirect(url_for("students"))
        except sqlite3.IntegrityError:
            flash("Student number or email already exists", "danger")
    
    c.execute("SELECT * FROM Students WHERE Student_ID = ?", (student_id,))
    student = c.fetchone()
    
    c.execute("SELECT Module_ID, Module_Code, Module_Name FROM Modules")
    all_modules = c.fetchall()
    
    c.execute("SELECT Module_ID FROM StudentModules WHERE Student_ID = ?", (student_id,))
    student_modules = [row[0] for row in c.fetchall()]
    
    conn.close()
    
    return render_template("edit_student.html", 
                         student=student, 
                         modules=all_modules, 
                         student_modules=student_modules)

@app.route("/delete-student/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    # Delete student's face image if exists
    c.execute("SELECT Face_Image_Path FROM Students WHERE Student_ID = ?", (student_id,))
    result = c.fetchone()
    if result and result[0]:
        try:
            if os.path.exists(result[0]):
                os.remove(result[0])
        except:
            pass
    
    # Delete student and related records
    c.execute("DELETE FROM StudentModules WHERE Student_ID = ?", (student_id,))
    c.execute("DELETE FROM Attendance WHERE Student_ID = ?", (student_id,))
    c.execute("DELETE FROM LectureRatings WHERE Student_ID = ?", (student_id,))
    c.execute("DELETE FROM Students WHERE Student_ID = ?", (student_id,))
    
    conn.commit()
    conn.close()
    flash("Student deleted successfully!", "success")
    return redirect(url_for("students"))

@app.route("/admin-manage-faces")
def admin_manage_faces():
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("""
        SELECT s.Student_ID, s.Student_Number, s.First_Name, s.Last_Name, 
               s.Face_Registered, s.Face_Image_Path, s.Date_Created
        FROM Students s
        ORDER BY s.Student_ID
    """)
    students = c.fetchall()
    
    conn.close()
    
    return render_template("admin_manage_faces.html", students=students)

@app.route("/admin-reset-face/<int:student_id>", methods=["POST"])
def admin_reset_face(student_id):
    if not session.get("logged_in") or session.get("role") != "admin":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    # Get current face image path
    c.execute("SELECT Face_Image_Path FROM Students WHERE Student_ID = ?", (student_id,))
    result = c.fetchone()
    
    if result and result[0]:
        # Delete the face image file
        try:
            if os.path.exists(result[0]):
                os.remove(result[0])
        except:
            pass
    
    # Reset face registration in database
    c.execute("UPDATE Students SET Face_Registered = FALSE, Face_Image_Path = NULL WHERE Student_ID = ?", (student_id,))
    conn.commit()
    conn.close()
    
    flash("Face registration has been reset for the student", "success")
    return redirect(url_for("admin_manage_faces"))

# ---------------------------
# Routes - Student with Face Registration
# ---------------------------
@app.route("/student-dashboard")
def student_dashboard():
    if not session.get("logged_in") or session.get("role") != "student":
        return redirect(url_for("login"))
    
    student_id = session.get("student_id")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("SELECT * FROM Students WHERE Student_ID = ?", (student_id,))
    student = c.fetchone()
    
    if not student:
        flash("Student not found", "error")
        return redirect(url_for("login"))
    
    # Safely get modules with fallback for missing columns
    try:
        c.execute("""
            SELECT m.Module_ID, m.Module_Code, m.Module_Name, 
                   COALESCE(m.Attendance_Threshold, 75) as Attendance_Threshold
            FROM StudentModules sm
            JOIN Modules m ON sm.Module_ID = m.Module_ID
            WHERE sm.Student_ID = ?
        """, (student_id,))
        modules = c.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        # If there's still an error, use a basic query
        c.execute("""
            SELECT m.Module_ID, m.Module_Code, m.Module_Name, 75 as Attendance_Threshold
            FROM StudentModules sm
            JOIN Modules m ON sm.Module_ID = m.Module_ID
            WHERE sm.Student_ID = ?
        """, (student_id,))
        modules = c.fetchall()
    
    module_attendance = []
    for module in modules:
        module_id = module[0]
        attendance_percentage = get_student_attendance_percentage(student_id, module_id)
        module_attendance.append({
            'module_id': module_id,
            'module_code': module[1],
            'module_name': module[2],
            'threshold': module[3],  # This now uses COALESCE fallback
            'attendance': attendance_percentage,
            'status': 'Good' if attendance_percentage >= module[3] else 'Low'
        })
    
    # Get recent attendance
    c.execute("""
        SELECT a.Attendance_Date, m.Module_Code, m.Module_Name, a.Status
        FROM Attendance a
        JOIN Modules m ON a.Module_ID = m.Module_ID
        WHERE a.Student_ID = ?
        ORDER BY a.Attendance_Date DESC, a.Attendance_Time DESC
        LIMIT 10
    """, (student_id,))
    recent_attendance = c.fetchall()
    
    # Check if face is registered
    face_registered = check_face_registered(student_id)
    
    conn.close()
    
    return render_template("student_dashboard.html", 
                         student=student,
                         module_attendance=module_attendance,
                         recent_attendance=recent_attendance,
                         face_registered=face_registered)

@app.route("/student-attendance")
def student_attendance():
    if not session.get("logged_in") or session.get("role") != "student":
        return redirect(url_for("login"))
    
    student_id = session.get("student_id")
    module_filter = request.args.get("module", "all")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("""
        SELECT m.Module_ID, m.Module_Code, m.Module_Name
        FROM StudentModules sm
        JOIN Modules m ON sm.Module_ID = m.Module_ID
        WHERE sm.Student_ID = ?
    """, (student_id,))
    modules = c.fetchall()
    
    query = """
        SELECT a.Attendance_Date, a.Attendance_Time, m.Module_Code, m.Module_Name, a.Status, ls.Topic
        FROM Attendance a
        JOIN Modules m ON a.Module_ID = m.Module_ID
        LEFT JOIN LectureSessions ls ON a.Session_ID = ls.Session_ID
        WHERE a.Student_ID = ?
    """
    params = [student_id]
    
    if module_filter != "all":
        query += " AND a.Module_ID = ?"
        params.append(module_filter)
    
    query += " ORDER BY a.Attendance_Date DESC, a.Attendance_Time DESC"
    
    c.execute(query, params)
    attendance_records = c.fetchall()
    
    conn.close()
    
    return render_template("student_attendance.html",
                         attendance_records=attendance_records,
                         modules=modules,
                         current_filter=module_filter)

@app.route("/register-face")
def register_face():
    if not session.get("logged_in") or session.get("role") != "student":
        return redirect(url_for("login"))
    
    student_id = session.get("student_id")
    
    # Check if face is already registered
    if check_face_registered(student_id):
        flash("Your face is already registered. Contact admin if you need to update it.", "warning")
        return redirect(url_for("student_dashboard"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT First_Name, Last_Name, Student_Number FROM Students WHERE Student_ID = ?", (student_id,))
    student = c.fetchone()
    conn.close()
    
    return render_template("register_face.html", student=student)

@app.route("/video_feed")
def video_feed():
    """Video streaming route for face detection"""
    return Response(generate_frames_with_face_detection(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture-face", methods=["POST"])
def capture_face():
    if not session.get("logged_in") or session.get("role") != "student":
        return jsonify({"success": False, "message": "Unauthorized"})
    
    student_id = session.get("student_id")
    
    # Check if face is already registered
    if check_face_registered(student_id):
        return jsonify({"success": False, "message": "Face already registered"})
    
    success, message = capture_face_image(student_id)
    
    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "message": message})

@app.route("/face-registration-success")
def face_registration_success():
    if not session.get("logged_in") or session.get("role") != "student":
        return redirect(url_for("login"))
    
    student_id = session.get("student_id")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT First_Name, Last_Name FROM Students WHERE Student_ID = ?", (student_id,))
    student = c.fetchone()
    conn.close()
    
    return render_template("face_registration_success.html", student=student)

@app.route("/rate-lecture", methods=["GET", "POST"])
def rate_lecture():
    if not session.get("logged_in") or session.get("role") != "student":
        return redirect(url_for("login"))
    
    student_id = session.get("student_id")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    if request.method == "POST":
        session_id = request.form.get("session_id")
        rating = request.form.get("rating")
        feedback = request.form.get("feedback", "")

        if not session_id or not rating:
            flash("Please provide both session and rating", "danger")
            return redirect(url_for("rate_lecture"))
        
        # Validate rating
        try:
            rating = int(rating)
            if rating < 1 or rating > 5:
                flash("Rating must be between 1 and 5", "danger")
                return redirect(url_for("rate_lecture"))
        except ValueError:
            flash("Invalid rating value", "danger")
            return redirect(url_for("rate_lecture"))

        # Check if already rated
        c.execute("SELECT * FROM LectureRatings WHERE Student_ID = ? AND Session_ID = ?", 
                 (student_id, session_id))
        existing_rating = c.fetchone()
        
        if existing_rating:
            flash("You have already rated this lecture session", "warning")
        else:
            # Get module_id from session
            c.execute("SELECT Module_ID FROM LectureSessions WHERE Session_ID = ?", (session_id,))
            session_data = c.fetchone()
            
            if session_data:
                module_id = session_data[0]
                c.execute("""
                    INSERT INTO LectureRatings (Student_ID, Module_ID, Session_ID, Rating, Feedback)
                    VALUES (?, ?, ?, ?, ?)
                """, (student_id, module_id, session_id, rating, feedback))
                conn.commit()
                flash("Thank you for your feedback!", "success")
            else:
                flash("Invalid session selected", "danger")
        
        return redirect(url_for("rate_lecture"))
    
    # GET request - show sessions available for rating
    c.execute("""
        SELECT DISTINCT ls.Session_ID, m.Module_Code, m.Module_Name, 
               ls.Session_Date, ls.Topic, ls.Start_Time
        FROM Attendance a
        JOIN LectureSessions ls ON a.Session_ID = ls.Session_ID
        JOIN Modules m ON ls.Module_ID = m.Module_ID
        WHERE a.Student_ID = ? AND a.Status = 'Present'
        AND ls.Session_Date <= date('now')
        AND ls.Session_ID NOT IN (
            SELECT Session_ID FROM LectureRatings WHERE Student_ID = ?
        )
        ORDER BY ls.Session_Date DESC, ls.Start_Time DESC
    """, (student_id, student_id))
    
    sessions_to_rate = c.fetchall()
    conn.close()
    
    return render_template("rate_lecture.html", sessions_to_rate=sessions_to_rate)

# ---------------------------
# Routes - Lecturer with Enhanced Face Recognition Attendance
# ---------------------------
@app.route("/lecturer-dashboard")
def lecturer_dashboard():
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return redirect(url_for("login"))

    lecturer_id = session.get("lecturer_id")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("SELECT * FROM Lecturers WHERE Lecturer_ID = ?", (lecturer_id,))
    lecturer = c.fetchone()
    
    c.execute("""
        SELECT Module_ID, Module_Code, Module_Name, Number_of_Classes
        FROM Modules WHERE Lecturer_ID = ?
    """, (lecturer_id,))
    modules = c.fetchall()
    
    c.execute("""
        SELECT ls.Session_ID, m.Module_Code, ls.Session_Date, ls.Topic,
               COUNT(a.Student_ID) as attendance_count
        FROM LectureSessions ls
        JOIN Modules m ON ls.Module_ID = m.Module_ID
        LEFT JOIN Attendance a ON ls.Session_ID = a.Session_ID AND a.Status = 'Present'
        WHERE m.Lecturer_ID = ?
        GROUP BY ls.Session_ID
        ORDER BY ls.Session_Date DESC
        LIMIT 5
    """, (lecturer_id,))
    recent_sessions = c.fetchall()
    
    conn.close()
    
    return render_template("lecturer_dashboard.html", 
                         lecturer=lecturer,
                         modules=modules,
                         recent_sessions=recent_sessions)

@app.route("/take-attendance/<int:module_id>")
def take_attendance(module_id):
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT Lecturer_ID FROM Modules WHERE Module_ID = ?", (module_id,))
    module = c.fetchone()
    
    if not module or module[0] != session.get("lecturer_id"):
        flash("You don't have permission to access this module", "danger")
        return redirect(url_for("lecturer_dashboard"))
    
    c.execute("SELECT Module_Code, Module_Name FROM Modules WHERE Module_ID = ?", (module_id,))
    module_info = c.fetchone()
    conn.close()
    
    return render_template("take_attendance.html", 
                         module_id=module_id,
                         module_code=module_info[0],
                         module_name=module_info[1],
                         now=datetime.now())

@app.route("/take-attendance-enhanced/<int:module_id>")
def take_attendance_enhanced(module_id):
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT Lecturer_ID FROM Modules WHERE Module_ID = ?", (module_id,))
    module = c.fetchone()
    
    if not module or module[0] != session.get("lecturer_id"):
        flash("You don't have permission to access this module", "danger")
        return redirect(url_for("lecturer_dashboard"))
    
    c.execute("SELECT Module_Code, Module_Name FROM Modules WHERE Module_ID = ?", (module_id,))
    module_info = c.fetchone()
    
    # Get students enrolled in this module
    c.execute("""
        SELECT s.Student_ID, s.First_Name, s.Last_Name, s.Face_Registered
        FROM Students s
        JOIN StudentModules sm ON s.Student_ID = sm.Student_ID
        WHERE sm.Module_ID = ?
    """, (module_id,))
    enrolled_students = c.fetchall()
    
    conn.close()
    
    return render_template("take_attendance_enhanced.html", 
                         module_id=module_id,
                         module_code=module_info[0],
                         module_name=module_info[1],
                         enrolled_students=enrolled_students,
                         now=datetime.now())

@app.route("/video_feed_continuous/<int:module_id>")
def video_feed_continuous(module_id):
    """Continuous face recognition video feed"""
    lecturer_id = session.get("lecturer_id")
    return Response(generate_frames_with_continuous_recognition(module_id, lecturer_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start-attendance/<int:module_id>", methods=["POST"])
def start_attendance(module_id):
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return jsonify({"success": False, "message": "Unauthorized"})
    
    lecturer_id = session.get("lecturer_id")
    session_key = f"{module_id}_{lecturer_id}"
    
    if session_key in attendance_sessions:
        attendance_sessions[session_key].start_session()
        return jsonify({"success": True, "message": "Attendance session started!"})
    
    return jsonify({"success": False, "message": "Session not found"})

@app.route("/stop-attendance/<int:module_id>", methods=["POST"])
def stop_attendance(module_id):
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return jsonify({"success": False, "message": "Unauthorized"})
    
    lecturer_id = session.get("lecturer_id")
    session_key = f"{module_id}_{lecturer_id}"
    
    if session_key in attendance_sessions:
        session = attendance_sessions[session_key]
        session.stop_session()
        
        # Check attendance thresholds and send alerts
        alerts_sent = check_module_attendance_thresholds(module_id)
        
        return jsonify({
            "success": True, 
            "message": f"Attendance session completed! Detected {len(session.detected_students)} students. {alerts_sent} low attendance alerts sent."
        })
    
    return jsonify({"success": False, "message": "Session not found"})

@app.route("/mark-attendance-face", methods=["POST"])
def mark_attendance_face():
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return jsonify({"success": False, "message": "Unauthorized"})
    
    data = request.get_json()
    module_id = data.get('module_id')
    
    # Simulate face recognition and attendance marking
    success, message = recognize_face_for_attendance()
    
    if success:
        # For demo purposes, we'll mark attendance for a random student
        # In real system, you would identify the specific student
        conn = sqlite3.connect("face_logged.db")
        c = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get or create session
        c.execute("SELECT Session_ID FROM LectureSessions WHERE Module_ID = ? AND Session_Date = ?", 
                  (module_id, today))
        session_record = c.fetchone()
        
        if not session_record:
            c.execute("INSERT INTO LectureSessions (Module_ID, Session_Date, Start_Time, Topic) VALUES (?, ?, ?, ?)",
                     (module_id, today, datetime.now().strftime("%H:%M:%S"), "Face Recognition Session"))
            session_id = c.lastrowid
        else:
            session_id = session_record[0]
        
        # Get a student for demo (first student in the module)
        c.execute("""
            SELECT s.Student_ID FROM Students s
            JOIN StudentModules sm ON s.Student_ID = sm.Student_ID
            WHERE sm.Module_ID = ? AND s.Face_Registered = TRUE
            LIMIT 1
        """, (module_id,))
        student = c.fetchone()
        
        if student:
            student_id = student[0]
            
            # Check if already marked
            c.execute("SELECT * FROM Attendance WHERE Student_ID = ? AND Session_ID = ?", 
                     (student_id, session_id))
            if not c.fetchone():
                c.execute("""
                    INSERT INTO Attendance (Student_ID, Module_ID, Session_ID, Attendance_Date, Attendance_Time, Status)
                    VALUES (?, ?, ?, ?, ?, 'Present')
                """, (student_id, module_id, session_id, today, datetime.now().strftime("%H:%M:%S")))
                
                # Get student name for message
                c.execute("SELECT First_Name, Last_Name FROM Students WHERE Student_ID = ?", (student_id,))
                student_info = c.fetchone()
                
                conn.commit()
                conn.close()
                
                return jsonify({
                    "success": True, 
                    "message": f"✅ Attendance recorded successfully for {student_info[0]} {student_info[1]}!",
                    "student_name": f"{student_info[0]} {student_info[1]}"
                })
        
        conn.close()
        return jsonify({"success": False, "message": "No registered students found for this module"})
    
    else:
        return jsonify({"success": False, "message": message})

def recognize_face_for_attendance():
    """Recognize face for attendance marking"""
    camera = cv2.VideoCapture(0)
    
    # For demo purposes, we'll simulate recognition
    # In a real system, you would compare with stored face encodings
    
    success, frame = camera.read()
    camera.release()
    
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Simulate recognition - in real system, compare with database
            return True, "Face recognized successfully!"
        else:
            return False, "No face detected"
    else:
        return False, "Failed to capture image"

@app.route("/view-attendance/<int:module_id>")
def view_attendance(module_id):
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    c.execute("SELECT Lecturer_ID FROM Modules WHERE Module_ID = ?", (module_id,))
    module = c.fetchone()
    
    if not module or module[0] != session.get("lecturer_id"):
        flash("You don't have permission to access this module", "danger")
        return redirect(url_for("lecturer_dashboard"))
    
    c.execute("SELECT Module_Code, Module_Name FROM Modules WHERE Module_ID = ?", (module_id,))
    module_info = c.fetchone()
    
    date_filter = request.args.get("date", "")
    student_filter = request.args.get("student", "")
    
    query = """
        SELECT s.Student_Number, s.First_Name, s.Last_Name, a.Attendance_Date, 
               a.Attendance_Time, a.Status, ls.Topic
        FROM Attendance a
        JOIN Students s ON a.Student_ID = s.Student_ID
        LEFT JOIN LectureSessions ls ON a.Session_ID = ls.Session_ID
        WHERE a.Module_ID = ?
    """
    params = [module_id]
    
    if date_filter:
        query += " AND a.Attendance_Date = ?"
        params.append(date_filter)
    
    if student_filter:
        query += " AND (s.First_Name LIKE ? OR s.Last_Name LIKE ? OR s.Student_Number LIKE ?)"
        params.extend([f"%{student_filter}%", f"%{student_filter}%", f"%{student_filter}%"])
    
    query += " ORDER BY a.Attendance_Date DESC, a.Attendance_Time DESC"
    
    c.execute(query, params)
    attendance_records = c.fetchall()
    
    c.execute("SELECT DISTINCT Attendance_Date FROM Attendance WHERE Module_ID = ? ORDER BY Attendance_Date DESC", (module_id,))
    dates = c.fetchall()
    
    conn.close()
    
    return render_template("view_attendance.html",
                         module_id=module_id,
                         module_code=module_info[0],
                         module_name=module_info[1],
                         attendance_records=attendance_records,
                         dates=dates,
                         current_date=date_filter,
                         current_student=student_filter)

@app.route("/lecture-ratings")
def lecture_ratings():
    if not session.get("logged_in") or session.get("role") != "lecturer":
        return redirect(url_for("login"))
    
    lecturer_id = session.get("lecturer_id")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("""
        SELECT m.Module_Code, m.Module_Name, ls.Session_Date, ls.Topic,
               AVG(lr.Rating) as avg_rating, COUNT(lr.Rating) as rating_count
        FROM Modules m
        JOIN LectureSessions ls ON m.Module_ID = ls.Module_ID
        LEFT JOIN LectureRatings lr ON ls.Session_ID = lr.Session_ID
        WHERE m.Lecturer_ID = ?
        GROUP BY ls.Session_ID
        ORDER BY ls.Session_Date DESC
    """, (lecturer_id,))
    
    rating_summary = c.fetchall()
    
    c.execute("""
        SELECT m.Module_Code, ls.Session_Date, lr.Rating, lr.Feedback, s.First_Name, s.Last_Name
        FROM LectureRatings lr
        JOIN LectureSessions ls ON lr.Session_ID = ls.Session_ID
        JOIN Modules m ON ls.Module_ID = m.Module_ID
        JOIN Students s ON lr.Student_ID = s.Student_ID
        WHERE m.Lecturer_ID = ?
        ORDER BY ls.Session_Date DESC, lr.Rating
    """, (lecturer_id,))
    
    detailed_ratings = c.fetchall()
    
    conn.close()
    
    return render_template("lecture_ratings.html",
                         rating_summary=rating_summary,
                         detailed_ratings=detailed_ratings)

# ---------------------------
# Export Routes
# ---------------------------
@app.route("/export-attendance/<int:module_id>")
def export_attendance(module_id):
    if not session.get("logged_in") or session.get("role") not in ["lecturer", "admin"]:
        return redirect(url_for("login"))
    
    format_type = request.args.get("format", "pdf")
    
    conn = sqlite3.connect("face_logged.db")
    c = conn.cursor()
    
    c.execute("SELECT Module_Code, Module_Name FROM Modules WHERE Module_ID = ?", (module_id,))
    module_info = c.fetchone()
    
    c.execute("""
        SELECT s.Student_Number, s.First_Name, s.Last_Name, a.Attendance_Date, 
               a.Attendance_Time, a.Status, ls.Topic
        FROM Attendance a
        JOIN Students s ON a.Student_ID = s.Student_ID
        LEFT JOIN LectureSessions ls ON a.Session_ID = ls.Session_ID
        WHERE a.Module_ID = ?
        ORDER BY a.Attendance_Date DESC, a.Attendance_Time DESC
    """, (module_id,))
    
    attendance_data = c.fetchall()
    conn.close()
    
    if format_type == "excel":
        df = pd.DataFrame(attendance_data, columns=[
            'Student Number', 'First Name', 'Last Name', 'Date', 
            'Time', 'Status', 'Topic'
        ])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
        
        output.seek(0)
        return send_file(output, 
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name=f"attendance_{module_info[0]}.xlsx")
    
    else:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        title = Paragraph(f"Attendance Report - {module_info[1]} ({module_info[0]})", styles['Title'])
        elements.append(title)
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Paragraph("<br/>", styles['Normal']))
        
        data = [['Student No', 'Name', 'Date', 'Time', 'Status', 'Topic']]
        for record in attendance_data:
            data.append([
                record[0],
                f"{record[1]} {record[2]}",
                record[3],
                record[4],
                record[5],
                record[6] or "N/A"
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(buffer, 
                        as_attachment=True,
                        download_name=f"attendance_{module_info[0]}.pdf",
                        mimetype='application/pdf')

# Background task for attendance alerts
def check_attendance_background():
    """Background task to check attendance thresholds"""
    while True:
        check_attendance_thresholds()
        # Run once per day
        time.sleep(24 * 60 * 60)

# Start background thread for attendance alerts
alert_thread = threading.Thread(target=check_attendance_background, daemon=True)
alert_thread.start()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)