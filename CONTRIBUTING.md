# 贡献指南

感谢您对皮肤病灶轮廓处理系统的关注！

## 如何贡献

### 报告 Bug

如果您发现了 Bug，请：

1. 在 [Issues](https://github.com/Bbbbeiliu/skin_lesion_processor/issues) 中搜索是否已有相同问题
2. 如果没有，创建新的 Issue，包含以下信息：
   - Python 版本
   - 操作系统版本
   - 详细的问题描述
   - 复现步骤
   - 相关的错误日志或截图

### 提交新功能建议

如果您有新功能建议，请：

1. 在 [Issues](https://github.com/Bbbbeiliu/skin_lesion_processor/issues) 中描述您的想法
2. 说明该功能的用途和预期行为
3. 等待维护者讨论确认

### 提交代码

#### 开发流程

1. **Fork 仓库**
   ```bash
   # 在 GitHub 上点击 Fork 按钮
   ```

2. **克隆您的 Fork**
   ```bash
   git clone https://github.com/您的用户名/skin_lesion_processor.git
   cd skin_lesion_processor
   ```

3. **创建虚拟环境**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # 或
   source venv/bin/activate  # Linux/Mac
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

5. **创建特性分支**
   ```bash
   git checkout -b feature/您的功能名称
   # 或
   git checkout -b fix/您要修复的问题
   ```

6. **编写代码**
   - 遵循现有代码风格
   - 添加必要的注释
   - 确保代码通过测试

7. **提交更改**
   ```bash
   git add .
   git commit -m "简要描述您的更改"
   ```

8. **推送到 GitHub**
   ```bash
   git push origin feature/您的功能名称
   ```

9. **提交 Pull Request**
   - 在 GitHub 上创建 Pull Request
   - 详细描述您的更改
   - 等待代码审查

#### 代码规范

- **Python**: 遵循 [PEP 8](https://peps.python.org/pep-0008/) 风格指南
- **注释**: 重要逻辑添加注释说明
- **命名**: 使用清晰的变量和函数名
- **文档**: 更新相关文档

#### Commit 消息规范

使用清晰的 Commit 消息：

```
[新增] 添加XXX功能
[修复] 修复XXX问题
[优化] 优化XXX性能
[文档] 更新XXX文档
```

---

## 行为准则

- 尊重所有贡献者
- 建设性讨论问题
- 接受不同观点
- 专注于项目改进

---

## 许可证

通过贡献代码，您同意您的贡献将采用 MIT 许可证进行授权。
