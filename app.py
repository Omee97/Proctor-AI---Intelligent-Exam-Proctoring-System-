from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
from datetime import datetime
import requests
import tempfile
from urllib.parse import urlparse
import math
import openai
import videodb

OPENAI_API_KEY = "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"
VIDEODB_AUTH_KEY = "sk-WMYgy951iiGt4VEX2pPv0bX1HLa0BaKS033gGx0Ix1I"

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory "database"
videos = {}

# Load OpenCV face detection cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

VIDEODB_AUTH_KEY = "sk-WMYgy951iiGt4VEX2pPv0bX1HLa0BaKS033gGx0Ix1I"
vdb_conn = videodb.connect(api_key=VIDEODB_AUTH_KEY)

def download_video_from_url(url):
    """Download video from URL and save to temporary file"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'video/mp4,video/*,*/*;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        print(f"Attempting to download video from: {url}")
        
        head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        print(f"HEAD response status: {head_response.status_code}")
        print(f"Content-Type: {head_response.headers.get('content-type', 'unknown')}")
        
        response = requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        print(f"GET response content-type: {content_type}")
        
        file_extension = '.mp4'
        if 'video/mp4' in content_type:
            file_extension = '.mp4'
        elif 'video/avi' in content_type:
            file_extension = '.avi'
        elif 'video/mov' in content_type:
            file_extension = '.mov'
        elif 'video/webm' in content_type:
            file_extension = '.webm'
        else:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            if path.endswith('.mp4'):
                file_extension = '.mp4'
            elif path.endswith('.avi'):
                file_extension = '.avi'
            elif path.endswith('.mov'):
                file_extension = '.mov'
            elif path.endswith('.webm'):
                file_extension = '.webm'
        
        print(f"Using file extension: {file_extension}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_path = temp_file.name
        temp_file.close()
        
        total_size = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"Downloaded {total_size} bytes to {temp_path}")
        
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"File successfully downloaded: {temp_path}")
            return temp_path
        else:
            print("Downloaded file is empty or doesn't exist")
            return None
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def detect_faces_in_frame(frame):
    """Detect faces in a single frame using multiple cascades"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        frontal_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Detect profile faces
        profile_faces = profile_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Combine and filter overlapping detections
        all_faces = list(frontal_faces) + list(profile_faces)
        filtered_faces = remove_overlapping_detections(all_faces)
        
        return len(filtered_faces), filtered_faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return 0, []

def remove_overlapping_detections(detections, overlap_threshold=0.3):
    """Remove overlapping face detections"""
    if len(detections) <= 1:
        return detections
    
    # Sort by area (largest first)
    detections = sorted(detections, key=lambda x: x[2] * x[3], reverse=True)
    
    filtered = []
    for detection in detections:
        x, y, w, h = detection
        is_overlapping = False
        
        for existing in filtered:
            ex, ey, ew, eh = existing
            
            # Calculate intersection
            x1 = max(x, ex)
            y1 = max(y, ey)
            x2 = min(x + w, ex + ew)
            y2 = min(y + h, ey + eh)
            
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = w * h
                area2 = ew * eh
                union = area1 + area2 - intersection
                
                if intersection / union > overlap_threshold:
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            filtered.append(detection)
    
    return filtered

def detect_eyes_in_frame(frame, faces):
    """Detect eyes within detected face regions"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total_eyes = 0
        
        for (x, y, w, h) in faces:
            # Define region of interest (upper half of face for eyes)
            roi_gray = gray[y:y+h//2, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(10, 10),
                maxSize=(w//3, h//4)
            )
            
            total_eyes += len(eyes)
        
        return total_eyes
    except Exception as e:
        print(f"Error in eye detection: {e}")
        return 0

def calculate_face_position(face, frame_shape):
    """Calculate face position relative to frame center"""
    x, y, w, h = face
    frame_height, frame_width = frame_shape[:2]
    
    face_center_x = x + w/2
    face_center_y = y + h/2
    
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    # Calculate normalized deviation from center
    deviation_x = abs(face_center_x - frame_center_x) / frame_center_x
    deviation_y = abs(face_center_y - frame_center_y) / frame_center_y
    
    return deviation_x, deviation_y

def detect_motion_between_frames(frame1, frame2):
    """Detect motion between consecutive frames"""
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion intensity
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        return motion_ratio
    except Exception as e:
        print(f"Error in motion detection: {e}")
        return 0.0

def analyze_video_for_cheating_real(video_path):
    """Real video analysis for cheating detection based on actual frame processing"""
    try:
        print(f"Starting real video analysis: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file")
            return {"suspicious_incidents": [], "summary": {"error": "Failed to open video"}}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if total_frames == 0 or fps == 0:
            print("Invalid video file - no frames or fps")
            cap.release()
            return {"suspicious_incidents": [], "summary": {"error": "Invalid video file"}}
        
        print(f"Video properties: {total_frames} frames, {fps} fps, {duration:.2f}s duration")
        
        suspicious_incidents = []
        frame_count = 0
        prev_frame = None
        
        # Analysis parameters
        sample_interval = max(1, int(fps))  # Analyze 1 frame per second
        no_face_start = None
        multiple_people_start = None
        looking_away_start = None
        head_turned_start = None
        high_motion_start = None
        
        # Thresholds
        FACE_DEVIATION_THRESHOLD = 0.4  # Head turned threshold
        MOTION_THRESHOLD = 0.15  # High motion threshold
        EYE_RATIO_THRESHOLD = 0.5  # Looking away threshold
        MIN_INCIDENT_DURATION = 2.0  # Minimum duration for incident (seconds)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Process every sample_interval frames
            if frame_count % sample_interval == 0:
                # Detect faces
                num_faces, faces = detect_faces_in_frame(frame)
                
                # 1. Check for no person detected
                if num_faces == 0:
                    if no_face_start is None:
                        no_face_start = current_time
                else:
                    if no_face_start is not None and current_time - no_face_start >= MIN_INCIDENT_DURATION:
                        suspicious_incidents.append({
                            "start": no_face_start,
                            "end": current_time,
                            "type": "no_person",
                            "confidence": 0.9,
                            "description": "No person detected in frame"
                        })
                    no_face_start = None
                
                # 2. Check for multiple people
                if num_faces > 1:
                    if multiple_people_start is None:
                        multiple_people_start = current_time
                else:
                    if multiple_people_start is not None and current_time - multiple_people_start >= MIN_INCIDENT_DURATION:
                        suspicious_incidents.append({
                            "start": multiple_people_start,
                            "end": current_time,
                            "type": "multiple_people",
                            "confidence": 0.85,
                            "description": f"Multiple people detected (max: {max([len(detect_faces_in_frame(frame)[1]) for _ in range(1)])})"
                        })
                    multiple_people_start = None
                
                # Process single person scenarios
                if num_faces == 1:
                    face = faces[0]
                    
                    # 3. Check for head turning (face position analysis)
                    deviation_x, deviation_y = calculate_face_position(face, frame.shape)
                    max_deviation = max(deviation_x, deviation_y)
                    
                    if max_deviation > FACE_DEVIATION_THRESHOLD:
                        if head_turned_start is None:
                            head_turned_start = current_time
                    else:
                        if head_turned_start is not None and current_time - head_turned_start >= MIN_INCIDENT_DURATION:
                            suspicious_incidents.append({
                                "start": head_turned_start,
                                "end": current_time,
                                "type": "head_turned",
                                "confidence": min(0.9, max_deviation),
                                "description": f"Head turned away from camera (deviation: {max_deviation:.2f})"
                            })
                        head_turned_start = None
                    
                    # 4. Check for looking away (eye detection)
                    num_eyes = detect_eyes_in_frame(frame, faces)
                    expected_eyes = num_faces * 2
                    eye_ratio = num_eyes / expected_eyes if expected_eyes > 0 else 0
                    
                    if eye_ratio < EYE_RATIO_THRESHOLD:
                        if looking_away_start is None:
                            looking_away_start = current_time
                    else:
                        if looking_away_start is not None and current_time - looking_away_start >= MIN_INCIDENT_DURATION:
                            suspicious_incidents.append({
                                "start": looking_away_start,
                                "end": current_time,
                                "type": "looking_away",
                                "confidence": 1.0 - eye_ratio,
                                "description": f"Person appears to be looking away (eye ratio: {eye_ratio:.2f})"
                            })
                        looking_away_start = None
                else:
                    # Reset single-person incident trackers
                    if head_turned_start is not None:
                        head_turned_start = None
                    if looking_away_start is not None:
                        looking_away_start = None
                
                # 5. Check for excessive motion
                if prev_frame is not None:
                    motion_ratio = detect_motion_between_frames(prev_frame, frame)
                    
                    if motion_ratio > MOTION_THRESHOLD:
                        if high_motion_start is None:
                            high_motion_start = current_time
                    else:
                        if high_motion_start is not None and current_time - high_motion_start >= MIN_INCIDENT_DURATION:
                            suspicious_incidents.append({
                                "start": high_motion_start,
                                "end": current_time,
                                "type": "excessive_motion",
                                "confidence": min(0.9, motion_ratio * 5),
                                "description": f"Excessive motion detected (ratio: {motion_ratio:.3f})"
                            })
                        high_motion_start = None
                
                prev_frame = frame.copy()
            
            frame_count += 1
        
        # Close any remaining incidents
        final_time = total_frames / fps
        
        if no_face_start is not None and final_time - no_face_start >= MIN_INCIDENT_DURATION:
            suspicious_incidents.append({
                "start": no_face_start,
                "end": final_time,
                "type": "no_person",
                "confidence": 0.9,
                "description": "No person detected in frame"
            })
        
        if multiple_people_start is not None and final_time - multiple_people_start >= MIN_INCIDENT_DURATION:
            suspicious_incidents.append({
                "start": multiple_people_start,
                "end": final_time,
                "type": "multiple_people",
                "confidence": 0.85,
                "description": "Multiple people detected"
            })
        
        if head_turned_start is not None and final_time - head_turned_start >= MIN_INCIDENT_DURATION:
            suspicious_incidents.append({
                "start": head_turned_start,
                "end": final_time,
                "type": "head_turned",
                "confidence": 0.8,
                "description": "Head turned away from camera"
            })
        
        if looking_away_start is not None and final_time - looking_away_start >= MIN_INCIDENT_DURATION:
            suspicious_incidents.append({
                "start": looking_away_start,
                "end": final_time,
                "type": "looking_away",
                "confidence": 0.7,
                "description": "Person appears to be looking away"
            })
        
        if high_motion_start is not None and final_time - high_motion_start >= MIN_INCIDENT_DURATION:
            suspicious_incidents.append({
                "start": high_motion_start,
                "end": final_time,
                "type": "excessive_motion",
                "confidence": 0.8,
                "description": "Excessive motion detected"
            })
        
        cap.release()
        
        # Merge nearby incidents of the same type
        suspicious_incidents = merge_nearby_incidents(suspicious_incidents, time_threshold=3.0)
        
        # Generate summary
        summary = {
            "total_duration": duration,
            "total_incidents": len(suspicious_incidents),
            "incident_types": {},
            "total_suspicious_time": 0
        }
        
        for incident in suspicious_incidents:
            incident_type = incident["type"]
            if incident_type not in summary["incident_types"]:
                summary["incident_types"][incident_type] = 0
            summary["incident_types"][incident_type] += 1
            summary["total_suspicious_time"] += incident["end"] - incident["start"]
        
        summary["suspicion_percentage"] = (summary["total_suspicious_time"] / duration * 100) if duration > 0 else 0
        
        print(f"Real analysis complete. Found {len(suspicious_incidents)} suspicious incidents")
        print(f"Suspicion percentage: {summary['suspicion_percentage']:.1f}%")
        
        return {
            "suspicious_incidents": suspicious_incidents,
            "summary": summary
        }
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return {
            "suspicious_incidents": [], 
            "summary": {"error": f"Analysis failed: {str(e)}"}
        }

def merge_nearby_incidents(incidents, time_threshold):
    """Merge incidents that are close in time and of the same type"""
    if not incidents:
        return incidents
    
    # Sort incidents by start time
    incidents.sort(key=lambda x: x["start"])
    
    merged = []
    current = incidents[0]
    
    for incident in incidents[1:]:
        if (incident["type"] == current["type"] and 
            incident["start"] - current["end"] <= time_threshold):
            # Merge incidents
            current["end"] = incident["end"]
            current["confidence"] = max(current["confidence"], incident["confidence"])
        else:
            merged.append(current)
            current = incident
    
    merged.append(current)
    return merged

# --- VideoDB Scene Indexing for Semantic Search ---
def create_scene_index_for_proctoring(video_object):
    """Create scene index with proctoring-specific prompts"""
    try:
        print(f"Creating scene index for video: {video_object.id}")
        
        # Custom prompt for proctoring behavior detection
        proctoring_prompt = """
        Analyze this video scene for exam proctoring violations. Look for and describe:
        1. Student looking away from screen or camera
        2. Multiple people in the frame
        3. Student talking or whispering
        4. Suspicious hand movements or reaching off-screen
        5. Student using phones, notes, or other materials
        6. Head turning or looking in different directions
        7. Unusual body language or nervous behavior
        8. Any objects or people entering the frame
        
        Describe each scene focusing on potential academic dishonesty behaviors.
        """
        
        # Create scene index with custom prompt
        index_id = video_object.index_scenes(
            prompt=proctoring_prompt,
            scene_threshold=0.3  # Adjust sensitivity
        )
        
        print(f"Scene index created with ID: {index_id}")
        return index_id
        
    except Exception as e:
        print(f"Scene indexing error: {e}")
        return None

def search_suspicious_scenes(video_object, query_type="cheating"):
    """Search for specific types of suspicious behavior"""
    try:
        search_queries = {
            "cheating": "student cheating, looking at notes, using phone, suspicious behavior",
            "looking_away": "student looking away from camera, head turned, not facing screen",
            "multiple_people": "multiple people in frame, someone else present, additional person",
            "talking": "student talking, whispering, moving lips, speaking",
            "materials": "books, notes, phone, papers, unauthorized materials",
            "movement": "suspicious hand movements, reaching off screen, fidgeting"
        }
        
        query = search_queries.get(query_type, search_queries["cheating"])
        
        # Search the indexed scenes
        search_results = video_object.search(query=query)
        
        suspicious_scenes = []
        for result in search_results:
            suspicious_scenes.append({
                "start": result.start,
                "end": result.end,
                "description": result.text,
                "confidence": result.score,
                "query_type": query_type,
                "stream_url": result.stream_url
            })
        
        return suspicious_scenes
        
    except Exception as e:
        print(f"Scene search error: {e}")
        return []

def comprehensive_scene_analysis(video_object):
    """Perform comprehensive scene analysis for all violation types"""
    try:
        all_suspicious_scenes = []
        
        # Search for different types of violations
        violation_types = ["cheating", "looking_away", "multiple_people", "talking", "materials", "movement"]
        
        for violation_type in violation_types:
            scenes = search_suspicious_scenes(video_object, violation_type)
            all_suspicious_scenes.extend(scenes)
        
        # Remove duplicates and merge overlapping scenes
        unique_scenes = merge_overlapping_scenes(all_suspicious_scenes)
        
        return unique_scenes
        
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        return []

def merge_overlapping_scenes(scenes, time_threshold=5.0):
    """Merge overlapping or nearby suspicious scenes"""
    if not scenes:
        return scenes
    
    # Sort by start time
    scenes.sort(key=lambda x: x["start"])
    
    merged = []
    current = scenes[0]
    
    for scene in scenes[1:]:
        # If scenes overlap or are very close
        if scene["start"] - current["end"] <= time_threshold:
            # Merge scenes
            current["end"] = max(current["end"], scene["end"])
            current["confidence"] = max(current["confidence"], scene["confidence"])
            # Combine descriptions
            if scene["description"] not in current["description"]:
                current["description"] += f" | {scene['description']}"
        else:
            merged.append(current)
            current = scene
    
    merged.append(current)
    return merged
# --- VideoDB SDK: Upload and store video with analysis ---
def upload_video_to_videodb(video_path, video_id, filename, analysis):
    try:
        print(f"Uploading video to VideoDB: {filename}")
        
        # Upload video file to VideoDB
        video = vdb_conn.upload(file_path=video_path)
        
        # Create scene index for semantic search
        scene_index_id = create_scene_index_for_proctoring(video)
        
        # Perform comprehensive scene analysis
        semantic_scenes = comprehensive_scene_analysis(video)
        
        # Combine traditional CV analysis with semantic scene analysis
        enhanced_analysis = analysis.copy()
        enhanced_analysis["semantic_scenes"] = semantic_scenes
        enhanced_analysis["scene_index_id"] = scene_index_id
        enhanced_analysis["total_semantic_incidents"] = len(semantic_scenes)
        
        # Add metadata to the video including enhanced analysis
        video.add_metadata({
            'local_video_id': video_id,
            'filename': filename,
            'analysis': enhanced_analysis,
            'scene_index_id': scene_index_id,
            'upload_time': datetime.now().isoformat(),
            'proctoring_system': 'enhanced_cv_semantic_analysis'
        })
        
        print(f"Video uploaded to VideoDB with ID: {video.id}")
        print(f"Scene index created: {scene_index_id}")
        print(f"Found {len(semantic_scenes)} semantic incidents")
        
        return video.id
        
    except Exception as e:
        print(f"VideoDB upload error: {e}")
        return None

# --- VideoDB SDK: Store analysis metadata ---
def store_video_metadata_in_videodb(video_id, filename, analysis):
    try:
        print(f"Storing analysis metadata for video: {filename}")
        
        # If we have a VideoDB video ID, update its metadata
        # This is a fallback for when we only want to store metadata
        return True
        
    except Exception as e:
        print(f"VideoDB metadata storage error: {e}")
        return False

# --- VideoDB SDK: Fetch all videos with analysis ---
def fetch_videos_from_videodb():
    try:
        print("Fetching videos from VideoDB...")
        
        # Get all videos from VideoDB
        videos_list = []
        
        # Use VideoDB's list method to get all videos
        for video in vdb_conn.get_videos():
            video_data = {
                'videodb_id': video.id,
                'id': video.get_metadata().get('local_video_id', video.id),
                'filename': video.get_metadata().get('filename', 'unknown'),
                'analysis': video.get_metadata().get('analysis', {'suspicious_incidents': []}),
                'upload_time': video.get_metadata().get('upload_time', ''),
                'videodb_url': video.stream_url if hasattr(video, 'stream_url') else None
            }
            videos_list.append(video_data)
        
        print(f"Retrieved {len(videos_list)} videos from VideoDB")
        return videos_list
        
    except Exception as e:
        print(f"VideoDB fetch error: {e}")
        # Fallback to local storage if VideoDB fails
        return [videos[vid] for vid in videos.keys()]

@app.route('/test-url', methods=['POST'])
def test_url():
    """Test endpoint to check if a URL is accessible"""
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.head(url, headers=headers, timeout=10)
        content_type = response.headers.get('content-type', '')
        
        return jsonify({
            'accessible': response.status_code == 200,
            'content_type': content_type,
            'content_length': response.headers.get('content-length', 'unknown')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        print(f"Upload request received. Content-Type: {request.content_type}")
        
        # Check if it's a JSON request (URL upload)
        if request.is_json:
            data = request.get_json()
            video_url = data.get('video_url')
            if not video_url:
                return jsonify({'error': 'No video URL provided'}), 400
            
            print(f"Processing URL upload: {video_url}")
            video_id = str(uuid.uuid4())
            
            # Download video from URL
            temp_path = download_video_from_url(video_url)
            if not temp_path:
                return jsonify({'error': 'Failed to download video from URL'}), 400
            
            # Get file extension from the downloaded file
            file_extension = os.path.splitext(temp_path)[1]
            filename = f"{video_id}_url_video{file_extension}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Move temporary file to uploads folder
            import shutil
            shutil.move(temp_path, filepath)
            print(f"Video saved to: {filepath}")
            
        else:
            # Handle file upload
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            print(f"Processing file upload: {file.filename}")
            video_id = str(uuid.uuid4())
            filename = f"{video_id}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            video_url = None
            print(f"Video saved to: {filepath}")
        
        # Analyze the video for cheating behavior using REAL analysis
        print("Starting REAL cheating detection analysis...")
        analysis = analyze_video_for_cheating_real(filepath)
        
        # Upload video to VideoDB with analysis metadata
        videodb_id = upload_video_to_videodb(filepath, video_id, filename, analysis)
        
        # Store in local memory as backup
        videos[video_id] = {
            "id": video_id,
            "filename": filename,
            "filepath": filepath,
            "video_url": video_url,
            "videodb_id": videodb_id,
            "analysis": analysis,
            "upload_time": datetime.now().isoformat()
        }
        
        print(f"Upload successful. Video ID: {video_id}")
        print(f"Analysis results: {len(analysis.get('suspicious_incidents', []))} incidents found")
        
        return jsonify({
            "id": video_id, 
            "filename": filename,
            "videodb_id": videodb_id,
            "analysis": analysis,
            "message": "Video uploaded to VideoDB and analyzed for cheating behavior using real computer vision"
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/videos', methods=['GET'])
def list_videos():
    # Try to get videos from VideoDB first, fallback to local storage
    try:
        videodb_videos = fetch_videos_from_videodb()
        if videodb_videos:
            return jsonify([{
                "id": v.get("id"),
                "videodb_id": v.get("videodb_id"), 
                "filename": v.get("filename"),
                "video_url": v.get("video_url"),
                "upload_time": v.get("upload_time", ""),
                "incident_count": len(v.get("analysis", {}).get("suspicious_incidents", [])),
                "duration": v.get("analysis", {}).get("summary", {}).get("total_duration", 0),
                "source": "videodb"
            } for v in videodb_videos])
    except Exception as e:
        print(f"Error fetching from VideoDB, using local storage: {e}")
    
    # Fallback to local storage
    return jsonify([{
        "id": v["id"], 
        "filename": v["filename"],
        "video_url": v.get("video_url"),
        "videodb_id": v.get("videodb_id"),
        "upload_time": v.get("upload_time", ""),
        "incident_count": len(v.get("analysis", {}).get("suspicious_incidents", [])),
        "duration": v.get("analysis", {}).get("summary", {}).get("total_duration", 0),
        "source": "local"
    } for v in videos.values()])

@app.route('/analysis/<video_id>', methods=['GET'])
def get_analysis(video_id):
    video = videos.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    return jsonify(video["analysis"])

# Serve uploaded video files
@app.route('/uploads/<filename>')
def serve_video(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/thumbnail/<video_id>/<int:timestamp>', methods=['GET'])
def get_thumbnail(video_id, timestamp):
    """Generate actual thumbnail from video at specified timestamp"""
    video = videos.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        cap = cv2.VideoCapture(video['filepath'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Resize frame for thumbnail
            height, width = frame.shape[:2]
            new_width = 320
            new_height = int((new_width * height) / width)
            thumbnail = cv2.resize(frame, (new_width, new_height))
            
            # Save thumbnail temporarily
            temp_path = os.path.join(UPLOAD_FOLDER, f"thumb_{video_id}_{timestamp}.jpg")
            cv2.imwrite(temp_path, thumbnail)
            
            return send_file(temp_path)
        else:
            return jsonify({'error': 'Could not extract frame'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Thumbnail generation failed: {str(e)}'}), 500

@app.route('/semantic-search/<video_id>', methods=['POST'])
def semantic_search_video(video_id):
    """Search for specific suspicious behaviors using semantic search"""
    try:
        data = request.get_json()
        query = data.get('query', 'suspicious behavior')
        
        video = videos.get(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        videodb_id = video.get('videodb_id')
        if not videodb_id:
            return jsonify({'error': 'Video not found in VideoDB'}), 404
        
        # Get VideoDB video object
        vdb_video = vdb_conn.get_video(videodb_id)
        
        # Perform semantic search
        search_results = vdb_video.search(query=query)
        
        results = []
        for result in search_results:
            results.append({
                "start": result.start,
                "end": result.end,
                "description": result.text,
                "confidence": result.score,
                "stream_url": result.stream_url,
                "thumbnail_url": result.thumbnail_url if hasattr(result, 'thumbnail_url') else None
            })
        
        return jsonify({
            "query": query,
            "total_results": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({'error': f'Semantic search failed: {str(e)}'}), 500

@app.route('/violation-search/<video_id>/<violation_type>', methods=['GET'])
def search_specific_violations(video_id, violation_type):
    """Search for specific types of violations"""
    try:
        video = videos.get(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        videodb_id = video.get('videodb_id')
        if not videodb_id:
            return jsonify({'error': 'Video not found in VideoDB'}), 404
        
        # Get VideoDB video object
        vdb_video = vdb_conn.get_video(videodb_id)
        
        # Search for specific violation type
        suspicious_scenes = search_suspicious_scenes(vdb_video, violation_type)
        
        return jsonify({
            "violation_type": violation_type,
            "total_incidents": len(suspicious_scenes),
            "incidents": suspicious_scenes
        })
        
    except Exception as e:
        return jsonify({'error': f'Violation search failed: {str(e)}'}), 500

@app.route('/videos_videodb', methods=['GET'])
def videos_from_videodb():
    try:
        videos_list = fetch_videos_from_videodb()
        return jsonify(videos_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)