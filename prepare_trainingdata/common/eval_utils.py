import ast
import re
from collections.abc import Sequence


_PAT_LAST_DIGIT = re.compile(
    r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
)


def extract_gsm8k_answer(text: str) -> float | None:
    boxed = re.search(r"\\boxed\{(.*)\}", text)
    search_text = boxed.group(1) if boxed else text
    matches = list(_PAT_LAST_DIGIT.finditer(search_text.replace(",", "").replace("+", "")))
    if not matches:
        return None
    try:
        return float(matches[-1].group())
    except ValueError:
        return None


def gsm8k_is_correct(generated_text: str, answer: float | int | str | None) -> bool:
    if answer is None:
        return False
    predicted = extract_gsm8k_answer(generated_text)
    if predicted is None:
        return False
    try:
        return abs(predicted - float(answer)) < 1e-5
    except (TypeError, ValueError):
        return False


def last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def remove_boxed(text: str | None) -> str:
    if not text:
        return ""
    if text.startswith("\\boxed "):
        return text[len("\\boxed ") :]
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text[len("\\boxed{") : -1]
    return text


def extract_countdown_answer(generated_text: str) -> str:
    boxed = last_boxed_only_string(generated_text)
    if boxed is not None:
        equation = remove_boxed(boxed)
    else:
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        equation = answer_match.group(1).strip() if answer_match else generated_text

    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    equation_match = re.search(r"([0-9+\-*/() .]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()
    return equation.strip()


def countdown_is_correct(equation: str, target: int | float | None, numbers: Sequence[int]) -> bool:
    if target is None or not _countdown_validate_equation(equation, numbers):
        return False
    result = _countdown_evaluate_equation(equation)
    return abs(result - float(target)) < 1e-5


def _countdown_validate_equation(equation: str, numbers: Sequence[int]) -> bool:
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r"\d+", equation))
        return numbers_in_eq == sorted(int(n) for n in numbers)
    except Exception:
        return False


def _countdown_evaluate_equation(equation: str) -> float:
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation):
            raise ValueError("Invalid characters in equation.")
        node = ast.parse(equation.strip(), mode="eval")
        for child in ast.walk(node):
            if not isinstance(
                child,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Num,
                    ast.Constant,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.USub,
                    ast.UAdd,
                    ast.Load,
                ),
            ):
                raise ValueError("Unsupported expression node.")
        return float(eval(compile(node, "<countdown>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return float("inf")


def count_operations(equation: str) -> int:
    return len(re.findall(r"[+\-*/^%]", equation or ""))


def parse_iconqa_problem(problem_text: str) -> tuple[str, list[str]]:
    problem_text = problem_text.replace("<image>", "").strip()
    question_lines = []
    options = []
    for line in problem_text.splitlines():
        match = re.match(r"^([A-Z])\.\s+(.*)$", line.strip())
        if match:
            options.append(match.group(2))
        elif line.strip():
            question_lines.append(line.strip())
    return "\n".join(question_lines).strip(), options


def format_iconqa_prompt(question: str, options: Sequence[str]) -> str:
    if not options:
        return f"{question}\n"
    choices = "\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(options))
    return f"{question}\nChoices: {choices}\n"


def iconqa_rule_is_correct(generated_text: str, answer: str, options: Sequence[str]) -> bool:
    prediction = _extract_iconqa_prediction(generated_text, options)
    answer = str(answer).strip()
    if not prediction or not answer:
        return False

    if _same_text(prediction, answer):
        return True
    if len(answer) == 1 and answer.upper().isalpha():
        idx = ord(answer.upper()) - ord("A")
        if 0 <= idx < len(options):
            return _same_text(prediction, options[idx]) or prediction.strip().upper() == answer.upper()
    if options and answer in options:
        return _same_text(prediction, answer)
    return False


def _extract_iconqa_prediction(generated_text: str, options: Sequence[str]) -> str:
    text = generated_text.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        text = remove_boxed(boxed).strip()

    letter_match = re.search(r"(?:answer is|answer:|option)\s*\(?([A-Z])\)?", text, re.IGNORECASE)
    if letter_match:
        letter = letter_match.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]
        return letter

    standalone = re.findall(r"\b([A-E])\b", text.upper())
    if standalone:
        letter = standalone[-1]
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]

    normalized_text = _normalize_text(text)
    for option in options:
        normalized_option = _normalize_text(option)
        if normalized_option and (normalized_option == normalized_text or normalized_option in normalized_text):
            return option
    return text


def _same_text(left: str, right: str) -> bool:
    return _normalize_text(left) == _normalize_text(right)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().rstrip("."))

