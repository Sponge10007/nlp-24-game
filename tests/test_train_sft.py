import unittest

from train_sft import render_text


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "<prompt>"


class TrainSftTest(unittest.TestCase):
    def test_render_text_builds_single_training_sequence(self):
        tokenizer = FakeTokenizer()
        example = {
            "prompt": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "completion": "<think>x</think><answer>1*2*3*4</answer>",
        }

        rendered = render_text(tokenizer, example)

        self.assertEqual(
            rendered["text"],
            "<prompt><think>x</think><answer>1*2*3*4</answer><eos>",
        )
        self.assertTrue(tokenizer.add_generation_prompt)


if __name__ == "__main__":
    unittest.main()
