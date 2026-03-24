# worker.py
import subprocess
import time
import config
import os

def execute_full_job(job, gui_callback=None):
    """
    执行完整流程：红光预览（preview_count次） + 实际标刻（mark_count次）
    """
    job['message'] = '正在执行预览+标刻...'
    if gui_callback:
        gui_callback(job)

    cmd = [
        config.PYTHON32_EXE,
        config.MARK_SCRIPT,
        '--preview_count', str(config.PREVIEW_COUNT),
        '--mark_count', str(config.MARK_COUNT),
        job['filepath']
    ]

    stop_event = job.get('stop_event')
    result = run_subprocess_with_stop(cmd, stop_event)

    if result['returncode'] == 0:
        if stop_event and stop_event.is_set():
            job['status'] = 'failed'
            job['message'] = '任务被用户终止'
        else:
            job['status'] = 'completed'
            job['message'] = f'任务完成\n{result["stdout"][:500]}'
    else:
        if stop_event and stop_event.is_set():
            job['status'] = 'failed'
            job['message'] = '任务被用户终止'
        else:
            job['status'] = 'failed'
            job['message'] = f'任务失败 (返回码 {result["returncode"]})\n{result["stderr"][:500]}'

    if gui_callback:
        gui_callback(job)

def run_subprocess_with_stop(cmd, stop_event, timeout=600):
    """
    运行子进程，支持通过 stop_event 提前终止
    """
    sdk_dir = config.SDK_DIR
    env = os.environ.copy()
    env['PATH'] = sdk_dir + os.pathsep + env.get('PATH', '')

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=sdk_dir,
            env=env
        )

        start_time = time.time()
        while True:
            if stop_event and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return {
                    'returncode': -1,
                    'stdout': proc.stdout.read() if proc.stdout else '',
                    'stderr': '用户停止'
                }

            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                return {
                    'returncode': proc.returncode,
                    'stdout': stdout,
                    'stderr': stderr
                }

            if time.time() - start_time > timeout:
                proc.terminate()
                proc.wait(timeout=2)
                if proc.poll() is None:
                    proc.kill()
                return {
                    'returncode': -1,
                    'stdout': proc.stdout.read() if proc.stdout else '',
                    'stderr': f'标刻超时（{timeout}秒）'
                }

            time.sleep(0.1)

    except Exception as e:
        return {
            'returncode': -2,
            'stdout': '',
            'stderr': f'执行异常: {str(e)}'
        }

def process_queue(queue, gui_callback):
    """处理队列中的 processing 任务"""
    while True:
        try:
            processing_jobs = [j for j in queue if j['status'] == 'processing']
            if processing_jobs:
                job = processing_jobs[0]
                execute_full_job(job, gui_callback)
        except Exception as e:
            if gui_callback:
                fake_job = {
                    'id': 'system_error',
                    'status': 'failed',
                    'message': f'Worker 内部错误: {str(e)}'
                }
                gui_callback(fake_job)
        time.sleep(1)