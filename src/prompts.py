# src/prompts.py

# Keep this short because 6GB GPUs are sensitive to context length.
SYSTEM_PROMPT = """你是一个数字算式推理专家。
给定数字和目标值，请使用 + - * / ( ) 算出目标值；每个给定数字必须且只能使用一次。
回复必须是：
<think>简短推导</think>
<answer>最终答案</answer>
<answer>中只能写 ASCII 算式或精确 UNSOLVABLE。禁止写 =24、解释文字、中文符号、近似词。
不要复用示例中的数字；必须使用本题给出的数字且每个只用一次。
BAD: <answer>3*6+7-13=24</answer>
BAD: <answer>(2 × 2 + 2) × 6</answer>
GOOD: answer only = ASCII expression using this puzzle's numbers once."""

def get_prompt(numbers: list[int], target_value: int | float = 24) -> str:
    """
    根据传入的数字列表生成极短的输入提示词
    例如输入 [3, 3, 8, 8]，输出 "数字：3, 3, 8, 8。请计算24。"
    """
    nums_str = ", ".join(map(str, numbers))
    return f"数字：{nums_str}。目标值：{target_value}。请计算。"
