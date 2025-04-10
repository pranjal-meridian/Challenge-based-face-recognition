import datetime
import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from deepface import DeepFace
from insightface.app import FaceAnalysis
import mediapipe as mp
import cv2
import requests
from collections import defaultdict
from bson import Binary


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img1_path = "test.png"

# Connect to MongoDB
URI = "mongodb://localhost:27017/"
client = MongoClient('mongodb://localhost:27017/')
db = client['Liveliness']
User = db['Users']
Logs = db["Logs"]
Admins = db["Admins"]

face = FaceAnalysis(name="buffalo_l")
face.prepare(ctx_id=0, det_size=(640, 640))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# tasks = ["Look Front", "Look Left", "Look Right", "Look Up", "Look Down"]
# selected_task = ""


def get_active_users_by_day():
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Last 7 days (including today)
    last_7_days = [(today - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]

    # Get all registered emails
    registered_emails = set(user["email"] for user in User.find({}))

    # Query logs for the last 7 days with login_status true
    query = {
        "timestamp": {"$gte": today - datetime.timedelta(days=6), "$lt": today + datetime.timedelta(days=1)},
        "login_status": True
    }
    all_logs = list(Logs.find(query))

    # Filter logs to only those by registered users
    filtered_logs = [log for log in all_logs if log["email"] in registered_emails]

    # Count unique users per day
    daily_user_set = defaultdict(set)
    for log in filtered_logs:
        date_str = log["timestamp"].strftime('%Y-%m-%d')
        daily_user_set[date_str].add(log["email"])

    active_users_per_day = [len(daily_user_set[day]) for day in last_7_days]

    # Total unique active users across the 7-day span
    total_active_users = len(set(email for emails in daily_user_set.values() for email in emails))

    # Previous 7-day period
    prev_query = {
        "timestamp": {
            "$gte": today - datetime.timedelta(days=13),
            "$lt": today - datetime.timedelta(days=6)
        },
        "login_status": True
    }
    prev_logs = list(Logs.find(prev_query))
    prev_filtered_logs = [log for log in prev_logs if log["email"] in registered_emails]
    prev_active_users = len(set(log["email"] for log in prev_filtered_logs))

    # Calculate percentage change
    if prev_active_users == 0:
        percentage_change = 0
    else:
        percentage_change = round(((total_active_users - prev_active_users) / prev_active_users) * 100, 2)

    change_arrow = "up" if total_active_users > prev_active_users else "down"

    return {
        "active_users_per_day": active_users_per_day,
        "total_active_users": total_active_users,
        "percentage_change": percentage_change,
        "change_arrow": change_arrow,
        "days": last_7_days
    }


# Function to decode base64 image from frontend
def decode_image(img_base64):
    try:
        img_data = base64.b64decode(img_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error decoding image:", e)
        return None


# this code for postman api testing :-
# def decode_image(image_file):
#     try:
#         image_file.seek(0)  # Ensure file pointer is at the beginning
#         img_array = np.frombuffer(image_file.read(), np.uint8)  # Read bytes as NumPy array
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image
#         if img is None:
#             raise ValueError("cv2.imdecode failed to decode image")  # Explicit error message
#         return img
#     except Exception as e:
#         print("Error decoding image:", str(e))  # Print error for debugging
#         return None


# Function to get stored reference embeddings from MongoDB
def get_reference_embedding(email):
    user = User.find_one({"email": email})
    if user and "face_embedding" in user:
        return np.array(user["face_embedding"])
    return None


# Function to compute embeddings of captured image
def compute_embedding(img):
    faces = face.get(img)
    if len(faces) > 0:
        return faces[0].normed_embedding
    return None


# # Function to detect head position using MediaPipe
def detect_head_position(image, face_landmarks, img_w, img_h):
    face_3d, face_2d = [], []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
    return x, y, z


# Function to check whether user performed correct task or not
def validate_task(image):
    img_h, img_w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x, y, z = detect_head_position(image, face_landmarks, img_w, img_h)
            if y < -9:
                head_position = "Left"
            elif y > 9:
                head_position = "Right"
            elif x > 15:
                head_position = "Up"
            else:
                head_position = "Front"
            return head_position
    return "Unknown"


# Function to check for spoofing using DeepFace
def check_liveness(img):
    try:
        result = DeepFace.extract_faces(img_path=img, detector_backend="opencv", enforce_detection=False, align=False,
                                        anti_spoofing=True)
        return "Live" if result[0]["antispoof_score"] > 0.5 else "Spoof"
    except Exception as e:
        print("Liveness detection error:", e)
    return "Unknown"


def get_location(latitude, longitude):
    url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={latitude}&longitude={longitude}&localityLanguage=en"
    response = requests.get(url)
    data = response.json()

    if "locality" in data:
        return {
            "city": data.get('city', 'Unknown'),
            "state": data.get('principalSubdivision', 'Unknown'),
            "country": data.get('countryName', 'Unknown')
        }
    return {"city": "Unknown", "state": "Unknown", "country": "Unknown"}

@app.route("/log-verification", methods=["POST"])
def log_verification():
    data = request.json
    email = data.get("email")
    latitude = data.get("latitude")
    time_taken = data.get("time_taken")
    longitude = data.get("longitude")
    location_data = get_location(latitude, longitude)


    if not email:
        return jsonify({"status": "error", "message": "Missing email"}), 400

    result = Logs.find_one_and_update(
        {"email": email},
        {"$set": {"verification_status": True, "status": "Verified", "detail": "Logged in Successfully",
                  "location": location_data, "timestamp": datetime.datetime.now(), "time_taken": time_taken}},
        sort=[("timestamp", -1)]
    )
    if result:
        return jsonify({"status": "success", "message": "Verified successfully in db."})
    else:
        return jsonify({"status": "error", "message": "Log entry not found."})


# registartion route
@app.route("/register", methods=["POST"])
def register():
    front_image = request.form.get("frontImage")
    left_image = request.form.get("leftImage")
    right_image = request.form.get("rightImage")
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if latitude and longitude:
        location_data = get_location(latitude, longitude)

    if not all([front_image, left_image, right_image, name, email, password]):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400  # Bad request

    if User.find_one({"email": email}):
        return jsonify({"status": "error", "message": "User already exists"}), 400

    # Decode and store the front image as a binary blob
    front_image_data = base64.b64decode(front_image.split(",")[1])  
    front_image_blob = Binary(front_image_data)  

    images = [front_image, left_image, right_image] 
    embeddings = []

    Timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for i, img_data in enumerate(images):
        if i == 0:  
            continue

        img_data = img_data.split(",")[1]  
        filename = f'images/{Timestamp}_{i}.jpeg'
        with open(filename, "wb") as f:
            f.write(base64.b64decode(img_data))

        img = cv2.imread(filename)
        faces = face.get(img)

        if len(faces) == 0:
            return jsonify({"status": "error", "message": f"No face detected in image {i + 1}"}), 400

        embeddings.append(faces[0].embedding)

    avg_embedding = np.mean(embeddings, axis=0).tolist()

    User.insert_one({
        "name": name,
        "email": email,
        "password": password,
        "user_image": front_image_blob,
        "face_embedding": avg_embedding,
        "timestamp": datetime.datetime.now()
    })

    Logs.insert_one({
        "email": email,
        "name": name,
        "status": "In Process",
        "login_status": True,
        "verification_status": False,
        "detail": "Registered",
        "location": location_data,
        "timestamp": datetime.datetime.now()
    })

    return jsonify({"status": "success"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    name = User.find_one({"email":email})['name']
    password = data.get("password")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    location_data = get_location(latitude, longitude) if latitude and longitude else {}
    if not email or not password:
        return jsonify({"status": "error", "message": "Missing email or password"}), 400

    user = User.find_one({"email": email})

    if not user:
        Logs.insert_one({"email": email, "name": name, "status": "Rejected", "login_status": False, "verification_status": False,
                         "detail": "User not found", "location": location_data, "timestamp": datetime.datetime.now()})
        return jsonify({"status": "error", "message": "User not found"}), 404

    if user["password"] != password:
        Logs.insert_one({"email": email, "name": name, "status": "Rejected", "login_status": False, "verification_status": False,
                         "detail": "Invalid password", "location": location_data, "timestamp": datetime.datetime.now()})
        return jsonify({"status": "error", "message": "Invalid password"}), 401

    is_admin = Admins.find_one({"email": email}) is not None
    if is_admin:
        Logs.insert_one(
            {"email": email, "name": name, "status": "Admin Login", "login_status": True, "verification_status": False,
             "detail": "Admin Login", "location": location_data,
             "timestamp": datetime.datetime.now()})
        return jsonify(
            {"status": "success", "message": "Admin Login successful", "is_admin": is_admin})

    Logs.insert_one({"email": email, "name": name, "status": "In Process", "login_status": True, "verification_status": False,
                     "detail": "Passed login, awaiting verification", "location": location_data, "timestamp": datetime.datetime.now()})
    return jsonify({"status": "success", "message": "Login successful, proceed to verification", "is_admin": is_admin})


''' use random function in frontend to send randomly selected task to backend at /verify route. This one has scalability issues '''


# # route for sending random task to frontend
# @app.route("/task", methods=["GET"])
# def get_random_task():
#     selected_task = random.choice(tasks)
#     return jsonify({"task": selected_task})


@app.route('/active-users', methods=['GET'])
def active_users():
    data = get_active_users_by_day()
    return jsonify(data)


@app.route('/auth-rates', methods=['GET'])
def get_auth_rates():
    success_count = Logs.count_documents({"status": "Verified"})
    failure_count = Logs.count_documents({"status": "Rejected"})

    return {
        "success": success_count,
        "failure": failure_count
    }

# actual verification route
@app.route('/verify', methods=['POST'])
def verify():
    try:
        '''for postman api testing:-'''
        # selected_task = "Look Front"
        # email = request.form.get("email")
        # img_base64 = request.files.get("image")

        email = request.form.get("email")
        selected_task = request.form.get("task")
        img_base64 = request.form.get("image")
        img_base64 = img_base64.split(",")[1]

        print(email, selected_task)

        if not email or not img_base64:
            return jsonify({"error": "Missing email or image"}), 400

        # Decode and process the image
        img = decode_image(img_base64)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Get stored reference embedding
        reference_embedding = get_reference_embedding(email)
        if reference_embedding is None:
            return jsonify({"error": "Reference embedding not found for this user"}), 404

        # Compute embeddings for the captured image
        captured_embedding = compute_embedding(img)
        if captured_embedding is None:
            return jsonify({"error": "No face detected in the captured image"}), 400

        # Compute similarity
        def normalize(embedding):
            return embedding / np.linalg.norm(embedding)

        similarity = np.dot(normalize(captured_embedding), normalize(reference_embedding))
        print(similarity)
        threshold = 0.6  # Adjust based on performance

        # Check face match
        face_match = "Matched" if similarity >= threshold else "Not Matched"

        # Check liveness
        liveness_status = check_liveness(img)

        # Task validation
        task_validity = ""
        task_result = validate_task(img)
        print("Detected task", task_result)
        if task_result == selected_task:
            task_validity = "Correct"
        else:
            task_validity = "Incorrect"

        # Response
        return jsonify({
            "face_match": face_match,
            "similarity": round(float(similarity), 2),
            "liveness_status": liveness_status,
            "task_validity": task_validity
        })

    except Exception as e:
        # import traceback
        # print(traceback.format_exc())  # Print the full error stack trace
        return jsonify({"error": str(e)}), 500


@app.route("/get-logs", methods=["GET"])
def get_logs():
    logs = list(Logs.find().sort("timestamp", -1))  # Fetch logs in descending order
    for log in logs:
        log["_id"] = str(log["_id"])  # Convert ObjectId to string (for frontend)
    return jsonify(logs)


@app.route('/average-session-duration', methods=['GET'])
def get_avg_session_duration():
    try:
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - datetime.timedelta(days=6)  # Last 7 days including today

        # Generate all dates in range
        all_dates = {(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"): 0 for i in range(7)}

        # ----------------- Daily Average Session Duration -----------------
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                    "status": "$status"
                },
                "all_time_taken": {"$push": "$time_taken"}
            }},
            {"$group": {
                "_id": "$_id.date",
                "statuses": {"$push": {"status": "$_id.status", "time_taken": "$all_time_taken"}}
            }},
            {"$sort": {"_id": 1}}
        ]

        results = list(Logs.aggregate(pipeline))

        for entry in results:
            date = entry["_id"]
            session_times = []
            in_process_found = False

            for status_entry in entry["statuses"]:
                if status_entry["status"] == "In Process":
                    in_process_found = True
                session_times.extend(status_entry["time_taken"])

            if in_process_found and session_times:
                session_times[0] = 0

            avg_session_duration = sum(session_times) / len(session_times) if session_times else 0
            all_dates[date] = round(avg_session_duration, 2)

        last_7_days_data = [{"date": date, "average_session_duration": avg_duration} for date, avg_duration in all_dates.items()]

        # ----------------- Overall Avg (Last 7 Days Only) -----------------
        last_7_days_pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {"$group": {
                "_id": None,
                "all_time_taken": {"$push": "$time_taken"}
            }}
        ]
        last_7_days_result = list(Logs.aggregate(last_7_days_pipeline))

        if last_7_days_result and last_7_days_result[0]["all_time_taken"]:
            all_sessions = sum(last_7_days_result[0]["all_time_taken"])
            session_count = len(last_7_days_result[0]["all_time_taken"])
            overall_avg_session_duration = round(all_sessions / session_count, 2) if session_count > 0 else 0
        else:
            overall_avg_session_duration = 0

        # ----------------- Previous Week Avg -----------------
        previous_week_start = today - datetime.timedelta(days=13)
        previous_week_end = today - datetime.timedelta(days=7)

        prev_week_pipeline = [
            {"$match": {"timestamp": {"$gte": previous_week_start, "$lte": previous_week_end}}},
            {"$group": {
                "_id": None,
                "all_time_taken": {"$push": "$time_taken"}
            }}
        ]
        prev_week_results = list(Logs.aggregate(prev_week_pipeline))

        if prev_week_results and prev_week_results[0]["all_time_taken"]:
            prev_sessions = sum(prev_week_results[0]["all_time_taken"])
            prev_session_count = len(prev_week_results[0]["all_time_taken"])
            prev_week_avg = round(prev_sessions / prev_session_count, 2) if prev_session_count > 0 else 0
        else:
            prev_week_avg = 0

        # ----------------- Percentage Change & Trend -----------------
        if prev_week_avg > 0:
            percentage_change = round(((overall_avg_session_duration - prev_week_avg) / prev_week_avg) * 100, 2)
        else:
            percentage_change = 0

        trend = "up" if percentage_change > 0 else "down" if percentage_change < 0 else "neutral"

        return jsonify({
            "success": True,
            "last_7_days": last_7_days_data,
            "overall_avg_session_duration": overall_avg_session_duration,
            "percentage_change": percentage_change,
            "trend": trend
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/user-stats', methods=['GET'])
def get_user_stats():
    try:
        now = datetime.datetime.now()
        start_of_this_week = now - datetime.timedelta(days=now.weekday())  # Monday 00:00 of this week
        start_of_last_week = start_of_this_week - datetime.timedelta(days=7)
        end_of_last_week = start_of_this_week - datetime.timedelta(seconds=1)

        # Total users
        total_users = User.count_documents({})
        print(f"Total users: {total_users}")

        # New users this week
        new_users_this_week = User.count_documents({
            "timestamp": {"$gte": start_of_this_week}
        })
        print(f"New users this week: {new_users_this_week}")

        # New users last week
        new_users_last_week = User.count_documents({
            "timestamp": {"$gte": start_of_last_week, "$lte": end_of_last_week}
        })
        print(f"New users last week: {new_users_last_week}")

        # Total users till end of last week
        total_users_last_week = User.count_documents({
            "timestamp": {"$lte": end_of_last_week}
        })
        print(f"Total users till end of last week: {total_users_last_week}")

        # Percentage change in total users
        total_change = total_users - total_users_last_week
        percent_total_change = (total_change / total_users_last_week * 100) if total_users_last_week else 0
        total_change_str = f"{percent_total_change:+.2f}%"
        total_trend = "up" if percent_total_change > 0 else "down" if percent_total_change < 0 else "neutral"

        # Percentage change in new users
        new_user_change = new_users_this_week - new_users_last_week
        percent_new_user_change = (new_user_change / new_users_last_week * 100) if new_users_last_week else 0
        new_user_change_str = f"{percent_new_user_change:+.2f}%"
        new_user_trend = "up" if percent_new_user_change > 0 else "down" if percent_new_user_change < 0 else "neutral"

        return jsonify({
            "total_users": total_users,
            "total_users_change": total_change_str,
            "total_users_trend": total_trend,
            "new_users_this_week": new_users_this_week,
            "new_users_change": new_user_change_str,
            "new_users_trend": new_user_trend
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
