from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db, bcrypt
from app.models.user import User, DetectionLog

user_bp = Blueprint('user', __name__)


@user_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user': user.to_dict()}), 200


@user_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json()

    if 'first_name' in data:
        user.first_name = data['first_name']
    if 'last_name' in data:
        user.last_name = data['last_name']
    if 'avatar_url' in data:
        user.avatar_url = data['avatar_url']

    db.session.commit()

    return jsonify({
        'message': 'Profile updated successfully',
        'user': user.to_dict()
    }), 200


@user_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json()

    if not data.get('current_password') or not data.get('new_password'):
        return jsonify({'error': 'Current password and new password required'}), 400

    if not bcrypt.check_password_hash(user.password_hash, data['current_password']):
        return jsonify({'error': 'Current password is incorrect'}), 401

    if len(data['new_password']) < 8:
        return jsonify({'error': 'New password must be at least 8 characters'}), 400

    user.password_hash = bcrypt.generate_password_hash(data['new_password']).decode('utf-8')
    db.session.commit()

    return jsonify({'message': 'Password changed successfully'}), 200


@user_bp.route('/dashboard', methods=['GET'])
@jwt_required()
def get_dashboard_data():
    """Get user dashboard data"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Get detection statistics
    total_detections = DetectionLog.query.filter_by(user_id=current_user_id).count()
    face_detections = DetectionLog.query.filter_by(user_id=current_user_id, detection_type='face').count()
    emotion_detections = DetectionLog.query.filter_by(user_id=current_user_id, detection_type='emotion').count()

    # Get recent detections
    recent_detections = DetectionLog.query.filter_by(user_id=current_user_id)\
        .order_by(DetectionLog.created_at.desc())\
        .limit(10)\
        .all()

    return jsonify({
        'user': user.to_dict(),
        'stats': {
            'total_detections': total_detections,
            'face_detections': face_detections,
            'emotion_detections': emotion_detections,
        },
        'recent_detections': [d.to_dict() for d in recent_detections]
    }), 200


@user_bp.route('/detection-history', methods=['GET'])
@jwt_required()
def get_detection_history():
    """Get user's detection history"""
    current_user_id = get_jwt_identity()

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    detections = DetectionLog.query.filter_by(user_id=current_user_id)\
        .order_by(DetectionLog.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'detections': [d.to_dict() for d in detections.items],
        'total': detections.total,
        'pages': detections.pages,
        'current_page': page
    }), 200
