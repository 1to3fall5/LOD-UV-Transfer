"""
简易 UV Transfer GUI
使用 tkinter 创建图形界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from uv_transfer import FBXHandler, TransferAlgorithm
from uv_transfer.core.transfer_engine import UVTransferEngine, TransferConfig, TransferMode


class UVTransferGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UV Transfer Tool - LOD UV 转移工具")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Arial', 10, 'bold'))
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        self.create_widgets()
        
        # 初始化 handler
        self.fbx_handler = FBXHandler()
        
    def create_widgets(self):
        """创建界面组件"""
        row = 0
        
        # 标题
        title_label = ttk.Label(
            self.main_frame, 
            text="LOD UV Transfer Tool", 
            style='Title.TLabel'
        )
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # 源文件选择
        ttk.Label(self.main_frame, text="源文件 (LOD0):", style='Subtitle.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        self.source_path = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.source_path).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5
        )
        ttk.Button(self.main_frame, text="浏览...", command=self.browse_source).grid(
            row=row, column=2, pady=5
        )
        row += 1
        
        # 目标文件选择
        ttk.Label(self.main_frame, text="目标文件 (LOD1/2/3):", style='Subtitle.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        self.target_path = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.target_path).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5
        )
        ttk.Button(self.main_frame, text="浏览...", command=self.browse_target).grid(
            row=row, column=2, pady=5
        )
        row += 1
        
        # 输出文件选择
        ttk.Label(self.main_frame, text="输出文件:", style='Subtitle.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        self.output_path = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.output_path).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5
        )
        ttk.Button(self.main_frame, text="浏览...", command=self.browse_output).grid(
            row=row, column=2, pady=5
        )
        row += 1
        
        # 分隔线
        ttk.Separator(self.main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        # UV 通道设置
        ttk.Label(self.main_frame, text="UV 通道设置", style='Subtitle.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        row += 1
        
        # 源 UV 通道
        ttk.Label(self.main_frame, text="源 UV 通道:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.source_uv = tk.IntVar(value=2)
        uv_frame = ttk.Frame(self.main_frame)
        uv_frame.grid(row=row, column=1, sticky=tk.W, pady=2)
        for i in range(4):
            ttk.Radiobutton(uv_frame, text=str(i), variable=self.source_uv, value=i).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # 目标 UV 通道
        ttk.Label(self.main_frame, text="目标 UV 通道:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.target_uv = tk.IntVar(value=2)
        uv_frame2 = ttk.Frame(self.main_frame)
        uv_frame2.grid(row=row, column=1, sticky=tk.W, pady=2)
        for i in range(4):
            ttk.Radiobutton(uv_frame2, text=str(i), variable=self.target_uv, value=i).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # 分隔线
        ttk.Separator(self.main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        # 算法选择
        ttk.Label(self.main_frame, text="转移算法", style='Subtitle.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        row += 1
        
        self.algorithm = tk.StringVar(value="triangle_center")
        
        algorithms = [
            ("triangle_center", "三角形中心 (Triangle Center) - 最佳平衡", "最适合大多数情况"),
            ("area_weighted", "面积加权 (Area Weighted) - 保留UV岛", "更好地保留UV接缝结构"),
            ("normal_aware", "法线感知 (Normal Aware) - 曲面优化", "最适合曲面模型"),
        ]
        
        for alg_id, alg_name, alg_desc in algorithms:
            rb = ttk.Radiobutton(
                self.main_frame, 
                text=alg_name, 
                variable=self.algorithm, 
                value=alg_id
            )
            rb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            ttk.Label(self.main_frame, text=alg_desc, foreground='gray').grid(
                row=row, column=1, sticky=tk.W, padx=20, pady=2
            )
            row += 1
        
        # 分隔线
        ttk.Separator(self.main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        # 执行按钮
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        self.transfer_btn = ttk.Button(
            button_frame, 
            text="开始转移", 
            command=self.start_transfer,
            width=20
        )
        self.transfer_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="退出", 
            command=self.root.quit,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(
            self.main_frame, 
            textvariable=self.status_var,
            foreground='blue'
        )
        status_label.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # 进度条
        self.progress = ttk.Progressbar(
            self.main_frame, 
            mode='indeterminate',
            length=400
        )
        self.progress.grid(row=row, column=0, columnspan=3, pady=5)
        
    def browse_source(self):
        """选择源文件"""
        filename = filedialog.askopenfilename(
            title="选择源 FBX 文件 (LOD0)",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")]
        )
        if filename:
            self.source_path.set(filename)
            # 自动设置输出文件名
            if not self.output_path.get():
                base = Path(filename).stem
                output = Path(filename).parent / f"{base}_transferred.fbx"
                self.output_path.set(str(output))
    
    def browse_target(self):
        """选择目标文件"""
        filename = filedialog.askopenfilename(
            title="选择目标 FBX 文件 (LOD1/2/3)",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")]
        )
        if filename:
            self.target_path.set(filename)
    
    def browse_output(self):
        """选择输出文件"""
        filename = filedialog.asksaveasfilename(
            title="保存输出文件",
            defaultextension=".fbx",
            filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def start_transfer(self):
        """开始 UV 转移"""
        # 验证输入
        source = self.source_path.get()
        target = self.target_path.get()
        output = self.output_path.get()
        
        if not source or not os.path.exists(source):
            messagebox.showerror("错误", "请选择有效的源文件")
            return
        
        if not target or not os.path.exists(target):
            messagebox.showerror("错误", "请选择有效的目标文件")
            return
        
        if not output:
            messagebox.showerror("错误", "请指定输出文件路径")
            return
        
        # 禁用按钮
        self.transfer_btn.config(state='disabled')
        self.status_var.set("正在加载文件...")
        self.progress.start()
        self.root.update()
        
        try:
            # 创建配置
            algorithm_map = {
                'triangle_center': TransferAlgorithm.TRIANGLE_CENTER,
                'area_weighted': TransferAlgorithm.AREA_WEIGHTED,
                'normal_aware': TransferAlgorithm.NORMAL_AWARE,
            }
            
            config = TransferConfig(
                source_uv_channel=self.source_uv.get(),
                target_uv_channel=self.target_uv.get(),
                mode=TransferMode.SPATIAL,
                algorithm=algorithm_map[self.algorithm.get()],
                validate_source=True,
                validate_result=True
            )
            
            # 执行转移
            self.status_var.set("正在执行 UV 转移...")
            self.root.update()
            
            engine = UVTransferEngine(config=config)
            
            # 加载源文件
            self.status_var.set("加载源文件...")
            self.root.update()
            engine.load_source(source)
            
            # 加载目标文件
            self.status_var.set("加载目标文件...")
            self.root.update()
            engine.load_target(target)
            
            # 执行转移
            self.status_var.set("正在转移 UV 数据...")
            self.root.update()
            result = engine.transfer()
            
            if result.success:
                # 保存结果
                self.status_var.set("保存结果...")
                self.root.update()
                engine.save_result(output)
                
                # 显示成功信息
                msg = f"转移成功！\n\n"
                msg += f"源顶点数: {result.source_vertices}\n"
                msg += f"目标顶点数: {result.target_vertices}\n"
                msg += f"匹配率: {result.match_rate:.1%}\n"
                msg += f"精度: {result.accuracy:.6f}\n\n"
                msg += f"输出文件: {output}"
                
                messagebox.showinfo("成功", msg)
                self.status_var.set("转移完成")
            else:
                error_msg = "转移失败:\n" + "\n".join(result.errors)
                messagebox.showerror("错误", error_msg)
                self.status_var.set("转移失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"执行过程中出现错误:\n{str(e)}")
            self.status_var.set(f"错误: {str(e)}")
        finally:
            self.progress.stop()
            self.transfer_btn.config(state='normal')


def main():
    root = tk.Tk()
    app = UVTransferGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
