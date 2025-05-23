import base64
import os
from PIL import Image
from ais_utils.Model_from_LC_Ollama import get_chatOpenAI
from ais_utils.Model_from_LC_Groq import get_chatGroq
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)  # Setting the working directory to the script's location
data_path = Path("data")  # Defining the data path
#==============================================================
llm = get_chatOpenAI("granite3.2-vision", temperature=0.2)
llm2 = get_chatGroq()
# Encoding an image into a base64 string for sending as inline data
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Preparing a chat message payload with a base64 image and user prompt
def build_vision_chat_messages(prompt: str, image_base64: str):
    return [
        {"role": "system", "content": "You are a creative storyteller."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]

# Combining the workflow: encoding, building messages, and generating a story
def generate_story_from_image(image_path: str, prompt: str = "Tell a short story based on this image.") -> None:
    if not os.path.exists(image_path):
        print("❌ Image file not found:", image_path)
        return

    print("🖼 Encoding the image to base64...")
    image_base64 = encode_image_to_base64(image_path)

    print("🧱 Building the message payload...")
    messages = build_vision_chat_messages(prompt, image_base64)

    print("🧠 Calling the model to generate response")
    story = llm.invoke(messages).content
 
    print("\n📖 Generated Response:\n")
    print(story)

# Running the example from main
if __name__ == "__main__":
    os.system('clear')
    #print("📸 Image to Story Generation Example")
    #image_file_path = data_path / "image2.jpeg"  # 👈 Replace with your image path
    #user_prompt = "Write a short (250 words) magical fairy tale about this scene."
    #generate_story_from_image(image_file_path, user_prompt)

    print("📸 Image to OCR Example")
    image_file_path = data_path/"Untitled.jpeg"  # 👈 Replace with your image path
    user_prompt = "Transcribe the English Text"
    generate_story_from_image(image_file_path, user_prompt)
