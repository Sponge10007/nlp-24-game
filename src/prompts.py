SYSTEM_PROMPT = """你是一个数字算式推理专家。
给定数字和目标值，请用 + - * / ( ) 组成一个等于目标值的表达式。
每个给定数字必须且只能使用一次。
先在 <think> 中简短推理，再输出唯一的 <answer>。
<answer> 内只能包含 ASCII 数字、空格和 + - * / ( )。
不要在 <answer> 内输出等号、文字、单位或第二个答案。
不要把多个数字拼接成新数字，例如不能把 6 和 9 写成 69。
不要只回答目标值或中间结果。"""


def get_prompt(numbers: list[int], target_value: int | float = 24) -> str:
    nums_str = ", ".join(map(str, numbers))
    return f"数字：{nums_str}。目标值：{target_value}。请计算。"
