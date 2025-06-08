import random
import json
import os

def load_imagenet_labels(filepath):
    """
    Load 1000 ImageNet class labels from a .txt file.
    Each line should contain one label, e.g., 'zebra', 'airplane', etc.

    Args:
        filepath (str): Path to the file containing ImageNet labels.

    Returns:
        List[str]: A list of class labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Label file not found: {filepath}")

    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def generate_descriptive_labels(labels, templates):
    """
    Generate natural language descriptions for each class label.

    Args:
        labels (List[str]): Original class labels.
        templates (List[str]): List of sentence templates.

    Returns:
        Dict[str, str]: Mapping from original label to descriptive phrase.
    """
    label_description_map = {}
    for label in labels:
        template = random.choice(templates)
        phrase = template.format(label=label.replace("_", " "))
        label_description_map[label] = phrase
    return label_description_map


if __name__ == "__main__":
    # ====== Configuration ======
    # 👇 Change this to your label file path
    label_file_path = "imagenet_class_labels.txt"  # <-- one label per line (e.g., zebra)
    output_json_path = "imagenet_phrase_labels.json"

    # Templates for generating descriptive labels
    description_templates = [
    "a photo of a {label}",
    "an image of the {label}",
    "a detailed depiction of a {label}",
    "a scene containing a {label}",
    "a close-up view of the {label}",
    "a snapshot showing a {label}",
    "a typical representation of a {label}",
    "a natural picture featuring a {label}",
    "a visual example of a {label}",
    "a realistic image that includes a {label}"
]

    # ====== Execution ======
    try:
        labels = load_imagenet_labels(label_file_path)
        label_map = generate_descriptive_labels(labels, description_templates)

        # Save as JSON
        with open(output_json_path, 'w') as f:
            json.dump(label_map, f, indent=4)
        print(f"✅ Saved descriptive labels to: {output_json_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
