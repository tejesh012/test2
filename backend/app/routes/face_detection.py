from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.user import DetectionLog
from app.ml.face_detection import face_detection_service

face_detection_bp = Blueprint('face_detection', __name__)


@face_detection_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_frame():
    """Analyze a video frame for face detection and emotion"""
    current_user_id = get_jwt_identity()
    data = request.get_json()

    if not data or not data.get('image'):
        return jsonify({'error': 'No image data provided'}), 400

    # Process the frame
    result = face_detection_service.process_frame(data['image'])

    if not result['success']:
        return jsonify({'error': result.get('error', 'Processing failed')}), 400

    # Generate assistance message based on detected emotion
    assistance_message = face_detection_service.get_assistance_message(result.get('analysis', []))

    # Log the detection if faces found
    if result['faces_detected'] > 0:
        detection_log = DetectionLog(
            user_id=current_user_id,
            detection_type='face',
            result_data={
                'faces_count': result['faces_detected'],
                'emotions': [a.get('emotion') for a in result.get('analysis', [])]
            },
            confidence=result['analysis'][0].get('emotion_confidence', 0) if result.get('analysis') else 0
        )
        db.session.add(detection_log)
        db.session.commit()

    return jsonify({
        'success': True,
        'faces_detected': result['faces_detected'],
        'faces': result['faces'],
        'analysis': result['analysis'],
        'assistance_message': assistance_message,
        'image_size': result['image_size']
    }), 200


@face_detection_bp.route('/analyze-guest', methods=['POST'])
def analyze_frame_guest():
    """Analyze a video frame without authentication (for demo purposes)"""
    data = request.get_json()

    if not data or not data.get('image'):
        return jsonify({'error': 'No image data provided'}), 400

    # Process the frame
    result = face_detection_service.process_frame(data['image'])

    if not result['success']:
        return jsonify({'error': result.get('error', 'Processing failed')}), 400

    # Generate assistance message
    assistance_message = face_detection_service.get_assistance_message(result.get('analysis', []))

    return jsonify({
        'success': True,
        'faces_detected': result['faces_detected'],
        'faces': result['faces'],
        'analysis': result['analysis'],
        'assistance_message': assistance_message,
        'image_size': result['image_size']
    }), 200


@face_detection_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_detection_stats():
    """Get detection statistics for the current user"""
    current_user_id = get_jwt_identity()

    total = DetectionLog.query.filter_by(user_id=current_user_id).count()
    face_count = DetectionLog.query.filter_by(user_id=current_user_id, detection_type='face').count()

    # Get emotion distribution
    recent_logs = DetectionLog.query.filter_by(
        user_id=current_user_id,
        detection_type='face'
    ).order_by(DetectionLog.created_at.desc()).limit(100).all()

    emotion_counts = {}
    for log in recent_logs:
        if log.result_data and 'emotions' in log.result_data:
            for emotion in log.result_data['emotions']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    return jsonify({
        'total_detections': total,
        'face_detections': face_count,
        'emotion_distribution': emotion_counts
    }), 200
