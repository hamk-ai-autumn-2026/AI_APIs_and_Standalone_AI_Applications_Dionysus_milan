"""Human-editable persistent preferences saved in memory.md."""
import os
from pathlib import Path
from config import MEMORY_PATH

DEFAULTS = {
    "Name": "Guest",
    "Preferred language": "English",
    "Explanation style": "Step by step",
    "Maximum answer lines": "8",
    "Topics of interest": "Robotics and artificial intelligence",
}
ALLOWED_LANGUAGES = ("English", "Nepali", "Finnish", "Hindi", "Spanish", "French", "Auto-detect")
ALLOWED_STYLES = ("Step by step", "Concise", "Beginner-friendly", "Detailed", "Technical")


def load_memory(path: Path = MEMORY_PATH) -> dict[str, str]:
    result = DEFAULTS.copy()
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- ") and ":" in line:
            key, _, value = line[2:].partition(":")
            if key.strip() in result:
                result[key.strip()] = value.strip()
    return normalize_preferences(result)


def normalize_preferences(values: dict) -> dict[str, str]:
    result = {}
    for key in DEFAULTS:
        value = str(values.get(key, DEFAULTS[key])).replace("\r", " ").replace("\n", " ").strip()
        result[key] = value[:120]
    result["Name"] = result["Name"] or "Guest"
    result["Preferred language"] = result["Preferred language"] if result["Preferred language"] in ALLOWED_LANGUAGES else "English"
    result["Explanation style"] = result["Explanation style"] if result["Explanation style"] in ALLOWED_STYLES else "Step by step"
    try:
        limit = int(result["Maximum answer lines"])
    except ValueError:
        limit = 8
    result["Maximum answer lines"] = str(max(1, min(limit, 30)))
    return result


def save_memory(values: dict, path: Path = MEMORY_PATH) -> dict[str, str]:
    result = normalize_preferences(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        "# The Honest Assistant — User Preferences\n\n" +
        "\n".join(f"- {key}: {value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)
    return result


def memory_for_prompt(preferences: dict | None = None) -> str:
    prefs = normalize_preferences(preferences or load_memory())
    return "\n".join(f"{k}: {v}" for k, v in prefs.items())


def personal_memory_reply(question: str, preferences: dict | None = None) -> str | None:
    """Answer a few personal-memory questions from the *saved* profile, without RAG.

    This is deliberately limited: it does not pretend unrelated facts are known.
    """
    prefs = normalize_preferences(preferences or load_memory())
    q = question.casefold().strip().rstrip("?.!। ")
    language = prefs["Preferred language"]
    name = prefs["Name"]
    if q in {"what is my name", "do you know my name", "do you remember my name",
             "मेरो नाम के हो", "मेरा नाम क्या है", "mikä on nimeni", "what's my name"}:
        return {
            "Nepali": f"तपाईंले सुरक्षित गर्नुभएको नाम {name} हो।",
            "Finnish": f"Tallennettu nimesi on {name}.",
            "Hindi": f"आपका सहेजा हुआ नाम {name} है।",
            "Spanish": f"Tu nombre guardado es {name}.",
            "French": f"Votre nom enregistré est {name}.",
        }.get(language, f"Your saved name is {name}.")
    if q in {"what language do i prefer", "what is my preferred language", "which language did i choose"}:
        return f"Your saved language preference is {prefs['Preferred language']}."
    if q in {"what is my explanation style", "how do i like explanations"}:
        return f"Your explanation style is {prefs['Explanation style']}."
    if q in {"what are my preferences", "show my preferences", "what do you remember about me"}:
        limit = int(prefs["Maximum answer lines"])
        lines = [f"{label}: {value}" for label, value in prefs.items()]
        return "\n".join(lines[:limit])
    return None
