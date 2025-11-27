import logging

from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import uuid
from datetime import datetime
import json

from pydantic import BaseModel

# Import existing SQL Copilot functions
from app import init
from utils.logger import get_logger

session_states = {}
logger = get_logger(__name__)
app = Flask(__name__)
app.secret_key = 'sql_copilot_secret_key_2024'
app.config['SESSION_TYPE'] = 'filesystem'

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

@app.route('/')
def index():
    """Render the main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['chat_history'] = []
    
    return render_template('index.html')

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle chat messages with streaming response"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['chat_history'] = []

    if session['session_id'] not in session_states:
        session_states[session['session_id']] = {}

    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': '消息不能为空'}), 400
    
    # Add user message to chat history
    user_message = {
        'type': 'user',
        'content': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append(user_message)
    session.modified = True

    # Check if we are waiting for input (resuming from interruption)
    resume_input = None
    if session_states[session['session_id']].get("waiting_for_input"):
        resume_input = message
        session_states[session['session_id']]["waiting_for_input"] = False
        query_to_process = None
    else:
        query_to_process = message

    from app import stream_chat

    def generate():
        try:
            for event in stream_chat(
                query=query_to_process,
                database_id="chenjie",
                session_id=session['session_id'],
                resume_input=resume_input
            ):
                if event['type'] == 'interrupt':
                    session_states[session['session_id']]["waiting_for_input"] = True

                # Format as SSE
                yield f"data: {json.dumps(event, cls=CustomJSONEncoder)}\n\n"
                
        except Exception as e:
            error_event = {
                "type": "error",
                "content": str(e)
            }
            logging.error(str(e), e)
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get the current chat history"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    return jsonify({
        'success': True,
        'history': session['chat_history']
    })

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_history():
    """Clear the chat history"""
    session['chat_history'] = []
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': '聊天记录已清空'
    })

@app.route('/api/init', methods=['POST'])
def initialize_database():
    """Initialize the database"""
    try:
        init("chenjie")
        return jsonify({
            'success': True,
            'message': '数据库初始化成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'数据库初始化失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5123)