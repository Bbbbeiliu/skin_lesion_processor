# worker.py
import subprocess
import time
import config
import os          # 添加这一行

def execute_job(job, gui_callback=None):
    """
    执行单个任务：调用 mark_automation.py
    job: 任务字典
    gui_callback: 可选回调函数，用于更新 GUI 状态
    """
    job['status'] = 'processing'
    job['message'] = '正在标刻...'
    if gui_callback:
        gui_callback(job)

    cmd = [
        config.PYTHON32_EXE,
        config.MARK_SCRIPT,
        job['filepath']
    ]

    try:
        # 使用 run 并捕获输出，指定编码
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace'
        )
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode == 0:
            job['status'] = 'completed'
            job['message'] = f'标刻完成\n{stdout[:200]}'
        else:
            job['status'] = 'failed'
            job['message'] = f'标刻失败 (返回码 {result.returncode})\n{stderr[:500]}'
    except subprocess.TimeoutExpired:
        job['status'] = 'failed'
        job['message'] = '标刻超时（300秒）'
    except Exception as e:
        job['status'] = 'failed'
        job['message'] = f'执行异常: {str(e)}'

    if gui_callback:
        gui_callback(job)

def run_subprocess(cmd):
    """封装 subprocess.run，统一处理编码和环境"""
    sdk_dir = config.SDK_DIR
    env = os.environ.copy()
    env['PATH'] = sdk_dir + os.pathsep + env.get('PATH', '')
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace',
            cwd=sdk_dir,
            env=env
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': '标刻超时（300秒）'
        }
    except Exception as e:
        return {
            'returncode': -2,
            'stdout': '',
            'stderr': f'执行异常: {str(e)}'
        }

def preview_job(job, gui_callback=None):
    """执行红光预览"""
    job['status'] = 'previewing'
    job['message'] = '正在进行红光预览...'
    if gui_callback:
        gui_callback(job)

    cmd = [config.PYTHON32_EXE, config.MARK_SCRIPT]

    if job.get('mode') == 'template':
        # 模板模式：添加 --template 和 --replace
        cmd.extend(['--template', job['template_path'], '--replace', job.get('placeholder', 'PLACEHOLDER')])
        cmd.append('--preview')
        cmd.append(job['filepath'])
    else:
        # 直接模式
        cmd.extend(['--preview', job['filepath']])

    result = run_subprocess(cmd)
    if result['returncode'] == 0:
        job['status'] = 'preview_done'
        job['message'] = '红光预览完成，请确认是否加工'
    else:
        job['status'] = 'failed'
        job['message'] = f'红光预览失败 (返回码 {result["returncode"]})\n{result["stderr"]}'
    if gui_callback:
        gui_callback(job)

def mark_job(job, gui_callback=None):
    """执行实际标刻"""
    job['status'] = 'processing'
    job['message'] = '正在标刻...'
    if gui_callback:
        gui_callback(job)

    cmd = [config.PYTHON32_EXE, config.MARK_SCRIPT]

    if job.get('mode') == 'template':
        cmd.extend(['--template', job['template_path'], '--replace', job.get('placeholder', 'PLACEHOLDER')])
        cmd.append('--mark')
        cmd.append(job['filepath'])
    else:
        cmd.extend(['--mark', job['filepath']])

    result = run_subprocess(cmd)
    if result['returncode'] == 0:
        job['status'] = 'completed'
        job['message'] = f'标刻完成\n{result["stdout"][:500]}'
    else:
        job['status'] = 'failed'
        job['message'] = f'标刻失败 (返回码 {result["returncode"]})\n{result["stderr"][:500]}'
    if gui_callback:
        gui_callback(job)

def process_queue(queue, gui_callback):
    """处理队列：pending（预览）和 pending_mark（实际加工）"""
    while True:
        try:
            # 优先处理实际加工（pending_mark）
            mark_jobs = [j for j in queue if j['status'] == 'pending_mark']
            if mark_jobs:
                job = mark_jobs[0]
                mark_job(job, gui_callback)
                continue

            # 处理红光预览（pending）
            pending_jobs = [j for j in queue if j['status'] == 'pending']
            if pending_jobs:
                job = pending_jobs[0]
                preview_job(job, gui_callback)

        except Exception as e:
            if gui_callback:
                fake_job = {
                    'id': 'system_error',
                    'status': 'failed',
                    'message': f'Worker 内部错误: {str(e)}'
                }
                gui_callback(fake_job)
        time.sleep(1)