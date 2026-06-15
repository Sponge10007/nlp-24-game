# src/prompts.py

# Keep this short because 6GB GPUs are sensitive to context length.
SYSTEM_PROMPT = """你是一个数字算式推理专家。
给定数字和目标值，请使用 + - * / ( ) 算出目标值；每个给定数字必须且只能使用一次。
回复必须是：
<think>简短推导</think>
<answer>最终答案</answer>
Think can be Chinese. Answer must be ASCII-only.
Output exactly one <answer>...</answer>.
Inside answer use only digits, spaces, + - * / ( ).
No equals sign. No words. No second answer.
Forbidden in answer: =, ×, ÷, （, ）, Chinese text, English words.
Use this puzzle's numbers once each. Do not reuse any example text."""

def get_prompt(numbers: list[int], target_value: int | float = 24) -> str:
    """
    根据传入的数字列表生成极短的输入提示词
    例如输入 [3, 3, 8, 8]，输出 "数字：3, 3, 8, 8。请计算24。"
    """
    nums_str = ", ".join(map(str, numbers))
    return f"数字：{nums_str}。目标值：{target_value}。请计算。"
