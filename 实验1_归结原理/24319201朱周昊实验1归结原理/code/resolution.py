def ResolutionProp(KB):
    """
    实现命题逻辑的归结推理
    :param KB: 子句集，每个子句是一个元组，包含原子命题或其否定
    :return: 归结推理的过程，每个步骤保存为字符串，按顺序存到列表中返回
    """
    # 初始化推理过程列表
    steps = []
    
    # 将KB转换为列表以保持顺序
    kb_list = list(KB)
    
    # 首先添加原始子句到步骤中
    for i, clause in enumerate(kb_list, 1):
        # 格式化子句为字符串，确保没有空格
        clause_str = str(clause).replace('\'', '').replace(' ', '')
        steps.append(f"{i} {clause_str}")
    
    # 子句集的副本，用于存储所有子句（包括原始子句和新生成的子句）
    clauses = kb_list.copy()
    
    # 步骤编号，从原始子句的数量开始
    step_num = len(KB) + 1
    
    # 标记是否找到空集
    found_empty = False
    
    # 开始归结过程
    while not found_empty:
        # 遍历所有子句对
        new_clause_generated = False
        
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                # 尝试对这对子句进行归结
                resolved = resolve(clauses[i], clauses[j])
                
                # 如果可以归结
                if resolved is not None:
                    new_clause = resolved[0]
                    used_clauses = resolved[1]
                    
                    # 检查是否生成了空集
                    if new_clause == ():
                        found_empty = True
                    
                    # 格式化归结步骤
                    if len(clauses[j]) > 1:
                        # 如果第二个子句有多个文字，需要标记使用的文字
                        step_str = f"{step_num} r[{i+1},{j+1}{chr(97 + used_clauses[1])}] = {str(new_clause).replace('\'', '').replace(' ', '')}"
                    else:
                        # 否则不需要标记
                        step_str = f"{step_num} r[{i+1},{j+1}] = {str(new_clause).replace('\'', '').replace(' ', '')}"
                    
                    # 添加到步骤列表
                    steps.append(step_str)
                    
                    # 添加新子句到子句集
                    clauses.append(new_clause)
                    
                    # 增加步骤编号
                    step_num += 1
                    
                    # 标记生成了新子句
                    new_clause_generated = True
                    
                    # 如果找到空集，退出循环
                    if found_empty:
                        break
            if found_empty:
                break
        
        # 如果没有生成新子句，说明无法继续归结，退出循环
        if not new_clause_generated:
            break
    
    return steps

def resolve(clause1, clause2):
    """
    尝试对两个子句进行归结
    :param clause1: 第一个子句
    :param clause2: 第二个子句
    :return: 如果可以归结，返回(新子句, (clause1中使用的文字索引, clause2中使用的文字索引))；否则返回None
    """
    # 遍历第一个子句的所有文字
    for i, lit1 in enumerate(clause1):
        # 遍历第二个子句的所有文字
        for j, lit2 in enumerate(clause2):
            # 检查是否是互补文字
            if lit1.startswith('~') and lit1[1:] == lit2:
                # 生成新子句
                new_clause = tuple(sorted(set(clause1[:i] + clause1[i+1:] + clause2[:j] + clause2[j+1:])))
                return (new_clause, (i, j))
            elif lit2.startswith('~') and lit2[1:] == lit1:
                # 生成新子句
                new_clause = tuple(sorted(set(clause1[:i] + clause1[i+1:] + clause2[:j] + clause2[j+1:])))
                return (new_clause, (i, j))
    
    # 没有找到互补文字，无法归结
    return None


def MGU(expr1, expr2):
    """
    实现最一般合一（Most General Unifier）算法
    :param expr1: 第一个原子公式，字符串类型，如 'P(xx,a)'
    :param expr2: 第二个原子公式，字符串类型，如 'P(b,yy)'
    :return: 合一替换字典，如 {'xx':'b', 'yy':'a'}；若无法合一，返回空字典
    """
    # 解析两个表达式
    pred1, args1 = parse_expr(expr1)
    pred2, args2 = parse_expr(expr2)
    
    # 谓词必须相同
    if pred1 != pred2:
        return {}
    
    # 参数数量必须相同
    if len(args1) != len(args2):
        return {}
    
    # 初始化替换字典
    subst = {}
    
    # 对每个参数对进行合一
    for arg1, arg2 in zip(args1, args2):
        # 应用当前替换到参数
        arg1 = apply_subst_to_term(arg1, subst)
        arg2 = apply_subst_to_term(arg2, subst)
        
        # 尝试合一这两个参数
        new_subst = unify_terms(arg1, arg2)
        
        # 如果无法合一，返回空字典
        if new_subst is None:
            return {}
        
        # 合并替换
        subst.update(new_subst)
    
    return subst


def parse_expr(expr):
    """
    解析表达式，提取谓词名和参数列表
    :param expr: 表达式字符串，如 'P(xx,a)' 或 'f(g(yy))'
    :return: (谓词名, [参数列表])
    """
    # 找到左括号位置
    if '(' not in expr:
        # 没有括号，可能是常量或变量
        return (expr, [])
    
    pred_end = expr.find('(')
    pred_name = expr[:pred_end]
    
    # 提取括号内的内容
    args_str = expr[pred_end + 1:-1]  # 去掉最后的右括号
    
    # 解析参数列表（处理嵌套括号）
    args = parse_args(args_str)
    
    return (pred_name, args)


def parse_args(args_str):
    """
    解析参数列表，处理嵌套函数
    :param args_str: 参数字符串，如 'xx,a' 或 'f(g(yy)),a'
    :return: 参数列表
    """
    args = []
    current = ''
    depth = 0
    
    for char in args_str:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            # 顶层逗号，分隔参数
            if current:
                args.append(current.strip())
                current = ''
        else:
            current += char
    
    # 添加最后一个参数
    if current:
        args.append(current.strip())
    
    return args


def is_variable(term):
    """
    判断是否为变量
    根据示例推断：
    - 变量：x, y, z, xx, yy, zz, uu 等（通常用于表示变量的命名）
    - 常量：a, b, c, P, Q 等（通常用于表示具体值的命名）
    """
    if not term:
        return False
    
    # 去除可能的函数调用
    if '(' in term:
        return False
    
    # 单个小写字母（除了 a, b, c 等常见常量）是变量
    if len(term) == 1 and term.islower():
        # a, b, c 通常作为常量
        if term in 'abc':
            return False
        return True
    
    # 多字母变量：xx, yy, zz, uu 等重复字母
    if len(term) == 2 and term[0] == term[1] and term.islower():
        return True
    
    return False


def apply_subst_to_term(term, subst):
    """
    将替换应用到项
    :param term: 项字符串
    :param subst: 替换字典
    :return: 应用替换后的项
    """
    # 如果项是变量且在替换中，直接替换
    if is_variable(term) and term in subst:
        return subst[term]
    
    # 如果是函数，递归应用替换
    if '(' in term:
        pred, args = parse_expr(term)
        new_args = [apply_subst_to_term(arg, subst) for arg in args]
        return f"{pred}({','.join(new_args)})"
    
    return term


def unify_terms(term1, term2):
    """
    合一两个项
    :param term1: 第一个项
    :param term2: 第二个项
    :return: 替换字典或None（如果无法合一）
    """
    # 如果相同，无需替换
    if term1 == term2:
        return {}
    
    # 如果term1是变量
    if is_variable(term1):
        # occur检查：变量不能出现在要替换的项中
        if occurs_in(term1, term2):
            return None
        return {term1: term2}
    
    # 如果term2是变量
    if is_variable(term2):
        # occur检查
        if occurs_in(term2, term1):
            return None
        return {term2: term1}
    
    # 如果都是函数
    if '(' in term1 and '(' in term2:
        pred1, args1 = parse_expr(term1)
        pred2, args2 = parse_expr(term2)
        
        # 函数名必须相同
        if pred1 != pred2:
            return None
        
        # 参数数量必须相同
        if len(args1) != len(args2):
            return None
        
        # 合一所有参数
        subst = {}
        for arg1, arg2 in zip(args1, args2):
            # 应用当前替换
            arg1 = apply_subst_to_term(arg1, subst)
            arg2 = apply_subst_to_term(arg2, subst)
            
            # 合一这两个参数
            new_subst = unify_terms(arg1, arg2)
            if new_subst is None:
                return None
            
            subst.update(new_subst)
        
        return subst
    
    # 一个是函数，一个是常量/变量，无法合一
    return None


def occurs_in(var, term):
    """
    occur检查：检查变量是否出现在项中
    防止循环替换如 x = f(x)
    """
    if var == term:
        return True
    
    if '(' in term:
        _, args = parse_expr(term)
        for arg in args:
            if occurs_in(var, arg):
                return True
    
    return False

# # 测试示例
# if __name__ == "__main__":
#     # 示例输入，使用列表而不是集合以保持顺序
#     KB = [('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)]
#     # 调用函数
#     steps = ResolutionProp(KB)
#     # 打印结果
#     for step in steps:
#         print(step)



test1_KB = [
    ('P',),
    ('~P', 'Q'),
    ('~Q',)
]
test4_KB = [
    ('P(a)',),
    ('~P(x)', 'Q(x)'),
    ('~Q(a)',)
]
# 测试 MGU 函数
print("测试 MGU 函数:")
print(f"MGU('P(xx,a)', 'P(b,yy)') = {MGU('P(xx,a)', 'P(b,yy)')}")
print(f"MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))') = {MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))')}")
print()

# 测试 ResolutionProp
steps = ResolutionProp(test1_KB)
print("ResolutionProp 测试结果:")
for step in steps:
    print(step)