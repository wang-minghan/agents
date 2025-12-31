import ast
from typing import Optional

class CodeSummarizer:
    @staticmethod
    def summarize_python(code: str) -> str:
        """
        使用 AST 解析 Python 代码，只保留类、函数签名和 Docstring，
        隐藏具体实现细节，大幅减少 Token 占用。
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # 如果代码语法错误（如生成了一半），回退到简单截断
            return code[:500] + "\n...[Syntax Error, Raw Content Truncated]...\n" + code[-500:]

        summary_lines = []

        for node in tree.body:
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                summary_lines.append(ast.unparse(node))
            
            elif isinstance(node, ast.ClassDef):
                summary_lines.append(f"\nclass {node.name}({', '.join(base.id for base in node.bases if isinstance(base, ast.Name))}):")
                if ast.get_docstring(node):
                    summary_lines.append(f"    \"\"\"{ast.get_docstring(node)}\"\"\"")
                
                # 遍历类的方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [arg.arg for arg in item.args.args]
                        if 'self' in args: args.remove('self')
                        args_str = ", ".join(args)
                        
                        # 处理返回值注解
                        returns = ""
                        if item.returns:
                            returns = f" -> {ast.unparse(item.returns)}"
                            
                        summary_lines.append(f"    def {item.name}(self, {args_str}){returns}: ...")
                
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                args_str = ", ".join(args)
                returns = ""
                if node.returns:
                    returns = f" -> {ast.unparse(node.returns)}"
                summary_lines.append(f"\ndef {node.name}({args_str}){returns}: ...")
                if ast.get_docstring(node):
                    summary_lines.append(f"    \"\"\"{ast.get_docstring(node)}\"\"\"")

            elif isinstance(node, ast.Assign):
                # 保留全局变量定义，但不保留具体值（如果太长）
                try:
                    line = ast.unparse(node)
                    if len(line) < 100:
                        summary_lines.append(line)
                    else:
                        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                        summary_lines.append(f"{', '.join(targets)} = ...")
                except:
                    pass

        return "\n".join(summary_lines)
