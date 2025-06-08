import torch
from PIL import Image
import open_clip
import random
import os

# Load labels dynamically from folder names (use a list, not a dictionary)
def load_labels_from_directory(directory):
    label_list = []
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        # Ensure it's a directory and not a file
        if os.path.isdir(folder_path):
            # Add the folder name (label) to the list
            label_list.append(folder_name)
    print(label_list)
    return label_list

total_tests_done = 0

def test_image_text_similarity(model, preprocess, tokenizer, image_files, label_list, num_tests=100, num_iterations=3):
    """Test image-text similarity for a single dictionary."""
    results = []
    global total_tests_done
    success_rates_top_2 = []
    success_rates_top_3 = []

    for _ in range(num_iterations):
        success_top_2 = 0
        success_top_3 = 0

        for _ in range(num_tests):
            # Select a random image file
            image_fname = random.choice(image_files)
            
            # Extract the true label from the folder name
            true_label = os.path.basename(os.path.dirname(image_fname))
            print(true_label)
           
            # Tokenize the labels dynamically from the loaded label list
            text = tokenizer(label_list)

            # Preprocess the image
            image = preprocess(Image.open(image_fname)).unsqueeze(0)

            # Encode image and text features
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Calculate probabilities
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                # Get top matches
                top_2_indices = text_probs.topk(2, dim=-1).indices.squeeze().tolist()
                top_3_indices = text_probs.topk(3, dim=-1).indices.squeeze().tolist()

                # Check if true label is in the top 2 or top 3
                if true_label in label_list and label_list.index(true_label) in top_2_indices:
                    success_top_2 += 1
                if true_label in label_list and label_list.index(true_label) in top_3_indices:
                    success_top_3 += 1

            total_tests_done += 1
            print(f"Total tests done so far: {total_tests_done}")

        # Record success rates for this iteration
        success_rates_top_2.append(success_top_2 / num_tests)
        success_rates_top_3.append(success_top_3 / num_tests)

    # Calculate average success rates
    avg_success_top_2 = sum(success_rates_top_2) / num_iterations
    avg_success_top_3 = sum(success_rates_top_3) / num_iterations

     # Format to 2 decimal places and append the results
    formatted_results = {
        "avg_success_top_2": f"{avg_success_top_2 * 100:.2f}%",  # Convert to percentage and format
        "avg_success_top_3": f"{avg_success_top_3 * 100:.2f}%"   # Convert to percentage and format
    }

    results.append(formatted_results)

    print(f"Test results: {results}")
    return results

def main():
    # Path to the directory containing image files
    image_dir = 'image_path' # Change this to your directory path
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            image_files.append(os.path.join(root, file))

    # Path to the directory containing the folders (labels)
    label_directory = 'label_path'  # Change this to your directory path

    # Load labels from the directory
    label_list = load_labels_from_directory(label_directory)

    # Initialize model, tokenizer, and preprocess
    arch = '12-imgL-12-textL-ViT'
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAIONYFCC400M')
    tokenizer = open_clip.get_tokenizer(arch)

    # Run the image-text similarity test
    results = test_image_text_similarity(model, preprocess, tokenizer, image_files, label_list)

if __name__ == "__main__":
    main()
