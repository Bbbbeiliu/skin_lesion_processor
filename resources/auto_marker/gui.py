# gui.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinter.messagebox
import threading


class ExecutionDialog(tk.Toplevel):
    """正在执行的对话框，带有停止按钮"""
    def __init__(self, title, message, parent=None, stop_callback=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x100")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.grab_set()

        self.stop_callback = stop_callback
        self._stopped = False

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        self.label = ttk.Label(frame, text=message)
        self.label.pack(pady=(0, 10))

        self.stop_btn = ttk.Button(frame, text="停止", command=self._stop)
        self.stop_btn.pack()

    def _stop(self):
        self._stopped = True
        if self.stop_callback:
            self.stop_callback()
        self.destroy()

    def _on_close(self):
        self._stop()


class AppGUI:
    def __init__(self, task_queue, start_receiver_func):
        self.task_queue = task_queue
        self.start_receiver = start_receiver_func

        self.root = tk.Tk()
        self.root.title("激光加工自动控制 - 被控端")
        self.root.geometry("800x500")

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 任务列表（Treeview）
        columns = ('ID', '文件名', '状态', '消息')
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # 按钮区域
        btn_frame = ttk.Frame(self.root, padding="10")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(btn_frame, text="执行选中任务", command=self.execute_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="拒绝选中任务", command=self.reject_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空已完成", command=self.clear_completed).pack(side=tk.LEFT, padx=5)

        # 日志区域
        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding="5")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 启动接收线程
        threading.Thread(target=self.start_receiver, daemon=True).start()

        # 定时刷新 GUI
        self.refresh_gui()

    def reject_selected(self):
        """拒绝选中的任务，直接从队列删除"""
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        job_id = self.tree.item(item, 'values')[0]
        for i, job in enumerate(self.task_queue):
            if job['id'] == job_id and job['status'] == 'waiting':
                del self.task_queue[i]
                self.add_log(f"任务 {job_id} 已拒绝")
                break

    def clear_completed(self):
        """清空已完成或失败的任务"""
        self.task_queue[:] = [j for j in self.task_queue if j['status'] not in ('completed', 'failed')]
        self.add_log("已清空完成/失败的任务")

    def refresh_gui(self):
        """增量刷新任务列表，保留选中状态"""
        current_items = {}
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            job_id = values[0] if values else None
            current_items[job_id] = item

        for job in self.task_queue:
            job_id = job['id']
            status_display = {
                'waiting': '待执行',
                'processing': '加工中',   # 合并预览+标刻状态
                'completed': '已完成',
                'failed': '失败'
            }.get(job['status'], job['status'])
            new_values = (job_id, job['filename'], status_display, job['message'])

            if job_id in current_items:
                item = current_items.pop(job_id)
                if self.tree.item(item, 'values') != new_values:
                    self.tree.item(item, values=new_values)
            else:
                self.tree.insert('', tk.END, values=new_values)

        for item in current_items.values():
            self.tree.delete(item)

        waiting = len([j for j in self.task_queue if j['status'] == 'waiting'])
        processing = len([j for j in self.task_queue if j['status'] == 'processing'])
        completed = len([j for j in self.task_queue if j['status'] == 'completed'])
        failed = len([j for j in self.task_queue if j['status'] == 'failed'])
        self.status_var.set(f"待执行: {waiting} | 加工中: {processing} | 已完成: {completed} | 失败: {failed}")

        self.root.after(1000, self.refresh_gui)

    def add_log(self, msg):
        """添加日志到文本区域"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_job_status(self, job):
        """外部回调，用于更新任务状态（线程安全）"""
        self.root.after(0, self._update_job_status, job)

    def _update_job_status(self, job):
        # 更新任务列表中的对应项
        for i, j in enumerate(self.task_queue):
            if j['id'] == job['id']:
                self.task_queue[i] = job
                break
        self.add_log(f"任务 {job['id']} 状态: {job['status']} - {job['message']}")

        # 根据状态显示或关闭执行对话框
        if job['status'] == 'processing':
            if not hasattr(self, '_exec_dialog') or self._exec_dialog is None:
                stop_event = job.get('stop_event')
                self._exec_dialog = ExecutionDialog(
                    "激光加工", "正在执行预览+标刻...", self.root,
                    stop_callback=stop_event.set if stop_event else None
                )
                self._exec_dialog.protocol("WM_DELETE_WINDOW", self._on_exec_dialog_closed)
        elif job['status'] in ('completed', 'failed', 'waiting'):
            if hasattr(self, '_exec_dialog') and self._exec_dialog:
                self._exec_dialog.destroy()
                self._exec_dialog = None

    def _on_exec_dialog_closed(self):
        if hasattr(self, '_exec_dialog') and self._exec_dialog:
            self._exec_dialog = None

    def run(self):
        self.root.mainloop()

    def execute_selected(self):
        """执行选中任务：立即开始预览+标刻，无中间弹窗"""
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        job_id = self.tree.item(item, 'values')[0]
        for job in self.task_queue:
            if job['id'] == job_id and job['status'] == 'waiting':
                job['stop_event'] = threading.Event()
                job['status'] = 'processing'
                job['message'] = '等待执行...'
                self.add_log(f"任务 {job['id']} 开始执行（预览5次+标刻3次）")
                break