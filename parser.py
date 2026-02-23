import ast
from structures import Document

def extract_symbols_with_lines(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()

    def block(s, e):
        return "\n".join(lines[s-1:e])

    items = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
            items.append(("function", node.name, s, e, block(s, e)))

        elif isinstance(node, ast.ClassDef):
            s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
            items.append(("class", node.name, s, e, block(s, e)))

            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qs, qe = sub.lineno, getattr(sub, "end_lineno", sub.lineno)
                    name = f"{node.name}.{sub.name}"
                    items.append(("method", name, qs, qe, block(qs, qe)))

    return items
