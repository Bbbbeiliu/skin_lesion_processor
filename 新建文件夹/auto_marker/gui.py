# gui.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import tkinter.messagebox

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

        # ttk.Button(btn_frame, text="确认选中任务", command=self.confirm_selected).pack(side=tk.LEFT, padx=5)
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
        # 获取当前所有任务的 ID 与 Treeview 项的映射
        current_items = {}
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            job_id = values[0] if values else None
            current_items[job_id] = item

        # 遍历任务队列，更新或插入
        for job in self.task_queue:
            job_id = job['id']
            status_display = {
                'waiting': '待执行',
                'pending': '等待预览',
                'previewing': '预览中',
                'preview_done': '预览完成，待确认',  # 新增
                'pending_mark': '等待加工',
                'processing': '加工中',
                'completed': '已完成',
                'failed': '失败'
            }.get(job['status'], job['status'])
            new_values = (job_id, job['filename'], status_display, job['message'])

            if job_id in current_items:
                # 更新现有项
                item = current_items.pop(job_id)
                if self.tree.item(item, 'values') != new_values:
                    self.tree.item(item, values=new_values)
            else:
                # 插入新项
                self.tree.insert('', tk.END, values=new_values)

        # 删除不再存在的任务项
        for item in current_items.values():
            self.tree.delete(item)

        # 更新状态栏
        waiting = len([j for j in self.task_queue if j['status'] == 'waiting'])
        processing = len([j for j in self.task_queue if j['status'] in ('processing', 'previewing', 'pending_mark')])
        completed = len([j for j in self.task_queue if j['status'] == 'completed'])
        failed = len([j for j in self.task_queue if j['status'] == 'failed'])
        self.status_var.set(f"待执行: {waiting} | 加工中: {processing} | 已完成: {completed} | 失败: {failed}")

        # 每秒刷新一次
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

        # 如果状态变为 preview_done，弹出确认窗口
        if job['status'] == 'preview_done':
            self.ask_confirm_mark(job)

    def run(self):
        self.root.mainloop()

    def execute_selected(self):
        """执行选中任务：预览 + 后续确认"""
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        job_id = self.tree.item(item, 'values')[0]
        for job in self.task_queue:
            if job['id'] == job_id and job['status'] == 'waiting':
                job['status'] = 'pending'
                self.add_log(f"任务 {job_id} 开始红光预览")
                break

    def ask_confirm_mark(self, job):
        """弹出确认加工窗口"""
        result = tk.messagebox.askyesno("加工确认",
                                        f"任务 {job['id']}\n红光预览已完成，是否开始实际加工？")
        if result:
            # 用户确认加工
            job['status'] = 'pending_mark'
            self.add_log(f"任务 {job['id']} 已确认加工，等待执行")
        else:
            # 用户取消，任务退回等待状态
            job['status'] = 'waiting'
            job['message'] = ''
            self.add_log(f"任务 {job['id']} 已取消加工，退回待执行状态")
        # 刷新界面
        self.refresh_gui()  # 这里直接刷新，因为已经通过 update_job_status 更新过，但为了立即显示可调用