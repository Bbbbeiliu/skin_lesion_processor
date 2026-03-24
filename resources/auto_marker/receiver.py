# receiver.py
import threading
import json
from flask import Flask, request, jsonify, current_app
import utils
import config

app = Flask(__name__)

def start_server(queue, log):
    """启动 Flask 服务器，将任务队列存入 app.config"""
    app.config['TASK_QUEUE'] = queue
    app.config['LOGGER'] = log
    app.run(host='0.0.0.0', port=config.SERVER_PORT, debug=False, threaded=True)

@app.route('/submit', methods=['POST'])
def submit_job():
    task_queue = current_app.config.get('TASK_QUEUE')
    logger = current_app.config.get('LOGGER')

    if task_queue is None:
        if logger:
            logger.error("任务队列未初始化")
        return jsonify({'error': 'Server not ready'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 获取参数（可选，保留用于未来扩展）
    params = request.form.get('params', '{}')
    try:
        params = json.loads(params)
    except:
        params = {}

    # 保存文件
    filename = f"{utils.generate_job_id()}_{file.filename}"
    filepath = utils.save_file(file.read(), filename)

    # 创建任务对象（不再包含 mode 和 template_path）
    job = {
        'id': filename,
        'filepath': filepath,
        'filename': file.filename,
        'params': params,
        'status': 'waiting',
        'message': ''
    }

    task_queue.append(job)
    if logger:
        logger.info(f"收到任务 {job['id']}，文件：{file.filename}")
    return jsonify({'status': 'accepted', 'job_id': job['id']})