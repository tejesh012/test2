from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from app import db, bcrypt
from app.models.user import User, UserSession

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()

    # Validate required fields
    required = ['email', 'username', 'password']
    if not all(field in data for field in required):
        return jsonify({'error': 'Missing required fields'}), 400

    # Check if user already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409

    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already taken'}), 409

    # Validate password strength
    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    # Create new user
    password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')

    user = User(
        email=data['email'],
        username=data['username'],
        password_hash=password_hash,
        first_name=data.get('first_name', ''),
        last_name=data.get('last_name', ''),
    )

    db.session.add(user)
    db.session.commit()

    # Generate tokens
    access_token = create_access_token(identity=user.id)
    refresh_token = create_refresh_token(identity=user.id)

    # Create session
    session = UserSession(
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get('User-Agent', ''),
        ip_address=request.remote_addr,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    db.session.add(session)
    db.session.commit()

    return jsonify({
        'message': 'Registration successful',
        'user': user.to_dict(),
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user and return tokens"""
    data = request.get_json()

    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400

    user = User.query.filter_by(email=data['email']).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, data['password']):
        return jsonify({'error': 'Invalid email or password'}), 401

    if not user.is_active:
        return jsonify({'error': 'Account is deactivated'}), 403

    # Update last login
    user.last_login = datetime.utcnow()

    # Generate tokens
    access_token = create_access_token(identity=user.id)
    refresh_token = create_refresh_token(identity=user.id)

    # Create new session
    session = UserSession(
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get('User-Agent', ''),
        ip_address=request.remote_addr,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    db.session.add(session)
    db.session.commit()

    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 200


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user_id = get_jwt_identity()
    access_token = create_access_token(identity=current_user_id)

    return jsonify({
        'access_token': access_token
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user and invalidate session"""
    current_user_id = get_jwt_identity()

    # Invalidate all sessions for this user (or specific session)
    UserSession.query.filter_by(user_id=current_user_id, is_active=True).update({'is_active': False})
    db.session.commit()

    return jsonify({'message': 'Logged out successfully'}), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current authenticated user"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user': user.to_dict()}), 200
