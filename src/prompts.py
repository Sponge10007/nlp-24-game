# src/prompts.py

# 【极简版 System Prompt】
# 6GB 显存对 Context 长度极其敏感，绝不能写长篇大论，直击要害即可。
SYSTEM_PROMPT = """你是一个数字算式推理专家。
给定若干个数字和一个目标值，请使用加、减、乘、除和括号算出目标值（允许浮点误差 10^(-6)）。每个数字必须且只能使用一次。
你必须先在<think>标签中写出逐步尝试的推导过程，然后将最终且仅包含算式的答案放在<answer>标签中。
如果这些数字无论如何都算不出目标值，请在<answer>中直接输出 UNSOLVABLE。
你的回复格式应该是<think>...</think>
<answer>...</answer>  """

def get_prompt(numbers: list[int], target_value: int | float = 24) -> str:
    """
    根据传入的数字列表生成极短的输入提示词
    例如输入 [3, 3, 8, 8]，输出 "数字：3, 3, 8, 8。请计算24。"
    """
    nums_str = ", ".join(map(str, numbers))
    return f"数字：{nums_str}。目标值：{target_value}。请计算。"
