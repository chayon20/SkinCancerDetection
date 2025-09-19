import os
import sqlite3
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from functools import wraps
from models.skin_model import predict_skin_cancer  # Your FP32 EfficientNet-B0 model

# --------------------------- Flask setup ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-strong-secret-key"
app.permanent_session_lifetime = timedelta(days=7)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------- Mail setup ---------------------------
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="",
    MAIL_PASSWORD="",
    MAIL_DEFAULT_SENDER=""
)
mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

# --------------------------- Database ---------------------------
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if not exists
conn = get_db_connection()
conn.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password TEXT,
    is_verified INTEGER DEFAULT 0
)
''')
conn.commit()
conn.close()

# --------------------------- Auth helpers ---------------------------
def login_user(user):
    session.permanent = True
    session["user"] = user['username']

def logout_user():
    session.pop('user', None)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Login required.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --------------------------- Email verification ---------------------------
def generate_confirmation_token(email):
    return s.dumps(email, salt='email-confirm')

def confirm_token(token, expiration=3600):
    try:
        email = s.loads(token, salt='email-confirm', max_age=expiration)
    except (SignatureExpired, BadSignature):
        return False
    return email

def send_verification_email(username, email):
    token = generate_confirmation_token(email)
    confirm_url = url_for('confirm_email', token=token, _external=True)
    html = f"""
    <p>Hi {username},</p>
    <p>Thanks for registering. Please click the link below to verify your email:</p>
    <a href="{confirm_url}">Verify Email</a>
    """
    msg = Message(subject="Email Verification", recipients=[email], html=html)
    mail.send(msg)

# --------------------------- Routes ---------------------------
@app.route('/')
def home():
    return render_template("home.html", user=session.get('user'))

# ---------- Register ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password'].strip()
        hashed_pw = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                         (username, email, hashed_pw))
            conn.commit()
            conn.close()
            send_verification_email(username, email)
            flash("Registration successful! Check your email to verify.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            conn.close()
            return redirect(url_for('register'))
    return render_template("register.html", user=session.get('user'))

# ---------- Email confirmation ----------
@app.route('/confirm/<token>')
def confirm_email(token):
    email = confirm_token(token)
    if not email:
        flash("Invalid or expired link.", "error")
        return redirect(url_for('login'))
    conn = get_db_connection()
    conn.execute("UPDATE users SET is_verified=1 WHERE email=?", (email,))
    conn.commit()
    conn.close()
    flash("Email verified! You can now login.", "success")
    return redirect(url_for('login'))

# ---------- Login ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            if not user['is_verified']:
                flash("Please verify your email before logging in.", "warning")
                return redirect(url_for('login'))
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials.", "error")
            return redirect(url_for('login'))
    return render_template("login.html", user=session.get('user'))

# ---------- Logout ----------
@app.route('/logout')
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for('home'))

# ---------- Profile ----------
@app.route('/profile')
@login_required
def profile():
    username = session['user']
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return render_template("profile.html", user=user)

# ---------- Edit Profile ----------
@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    username = session['user']
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if request.method == 'POST':
        new_email = request.form['email'].strip().lower()
        conn.execute("UPDATE users SET email=?, is_verified=0 WHERE username=?", (new_email, username))
        conn.commit()
        conn.close()
        send_verification_email(username, new_email)
        flash("Email updated. Verification sent.", "info")
        return redirect(url_for('profile'))
    conn.close()
    return render_template("edit_profile.html", user=user)

# ---------- Skin Cancer Detection ----------
@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction, confidence = predict_skin_cancer(filepath)
            return render_template("result.html", prediction=prediction, confidence=confidence, filename=filename)
    return render_template("detect.html")

# ---------- Serve uploaded files ----------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --------------------------- Run ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
