
import gradio as gr
from requests import head
from transformer_vectorizer import TransformerVectorizer
from concrete.ml.deployment import FHEModelClient
import numpy
import os
from pathlib import Path
import requests
import json
import base64
import subprocess
import shutil
import time

# This repository's directory
REPO_DIR = Path(__file__).parent

subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)

# Wait 5 sec for the server to start
time.sleep(5)

# Encrypted data limit for the browser to display
# (encrypted data is too large to display in the browser)
ENCRYPTED_DATA_BROWSER_LIMIT = 500
N_USER_KEY_STORED = 20
FHE_MODEL_PATH = "deployment/sentiment_fhe_model"

print("Loading the transformer model...")

# Initialize the transformer vectorizer
transformer_vectorizer = TransformerVectorizer()

def clean_tmp_directory():
    # Allow 20 user keys to be stored.
    # Once that limitation is reached, deleted the oldest.
    path_sub_directories = sorted([f for f in Path(".fhe_keys/").iterdir() if f.is_dir()], key=os.path.getmtime)

    user_ids = []
    if len(path_sub_directories) > N_USER_KEY_STORED:
        n_files_to_delete = len(path_sub_directories) - N_USER_KEY_STORED
        for p in path_sub_directories[:n_files_to_delete]:
            user_ids.append(p.name)
            shutil.rmtree(p)

    list_files_tmp = Path("tmp/").iterdir()
    # Delete all files related to user_id
    for file in list_files_tmp:
        for user_id in user_ids:
            if file.name.endswith(f"{user_id}.npy"):
                file.unlink()


def keygen():
    # Clean tmp directory if needed
    clean_tmp_directory()

    print("Initializing FHEModelClient...")

    #  create a user_id
    user_id = numpy.random.randint(0, 2**32)
    fhe_api = FHEModelClient(FHE_MODEL_PATH, f".fhe_keys/{user_id}")
    fhe_api.load()

    fhe_api.generate_private_and_evaluation_keys(force=True)
    evaluation_key = fhe_api.get_serialized_evaluation_keys()

    
    numpy.save(f"tmp/tmp_evaluation_key_{user_id}.npy", evaluation_key)

    return [list(evaluation_key)[:ENCRYPTED_DATA_BROWSER_LIMIT], user_id]


def encode_quantize_encrypt(text, user_id):
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")

    fhe_api = FHEModelClient(FHE_MODEL_PATH, f".fhe_keys/{user_id}")
    fhe_api.load()
    encodings = transformer_vectorizer.transform([text])
    quantized_encodings = fhe_api.model.quantize_input(encodings).astype(numpy.uint8)
    encrypted_quantized_encoding = fhe_api.quantize_encrypt_serialize(encodings)

    
    numpy.save(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy", encrypted_quantized_encoding)

    # Compute size
    encrypted_quantized_encoding_shorten = list(encrypted_quantized_encoding)[:ENCRYPTED_DATA_BROWSER_LIMIT]
    encrypted_quantized_encoding_shorten_hex = ''.join(f'{i:02x}' for i in encrypted_quantized_encoding_shorten)
    return (
        encodings[0],
        quantized_encodings[0],
        encrypted_quantized_encoding_shorten_hex,
    )


def run_fhe(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy")
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")
    if not encoded_data_path.is_file():
        raise gr.Error("No encrypted data was found. Encrypt the data before trying to predict.")

    # Read encrypted_quantized_encoding from the file
    encrypted_quantized_encoding = numpy.load(encoded_data_path)

    # Read evaluation_key from the file
    evaluation_key = numpy.load(f"tmp/tmp_evaluation_key_{user_id}.npy")

    # Use base64 to encode the encodings and evaluation key
    encrypted_quantized_encoding = base64.b64encode(encrypted_quantized_encoding).decode()
    encoded_evaluation_key = base64.b64encode(evaluation_key).decode()

    query = {}
    query["evaluation_key"] = encoded_evaluation_key
    query["encrypted_encoding"] = encrypted_quantized_encoding
    headers = {"Content-type": "application/json"}
    response = requests.post(
        "http://localhost:8000/predict_sentiment", data=json.dumps(query), headers=headers
    )
    encrypted_prediction = base64.b64decode(response.json()["encrypted_prediction"])
    print("Encrypted prediction is received")
    # Save encrypted_prediction in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_encrypted_prediction_{user_id}.npy", encrypted_prediction)
    encrypted_prediction_shorten = list(encrypted_prediction)[:ENCRYPTED_DATA_BROWSER_LIMIT]
    encrypted_prediction_shorten_hex = ''.join(f'{i:02x}' for i in encrypted_prediction_shorten)
    return encrypted_prediction_shorten_hex


def decrypt_prediction(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_prediction_{user_id}.npy")
    if not user_id:
        raise gr.Error("You need to generate FHE keys first.")
    if not encoded_data_path.is_file():
        raise gr.Error("No encrypted prediction was found. Run the prediction over the encrypted data first.")

    # Read encrypted_prediction from the file
    encrypted_prediction = numpy.load(encoded_data_path).tobytes()

    fhe_api = FHEModelClient(FHE_MODEL_PATH, f".fhe_keys/{user_id}")
    fhe_api.load()

    # We need to retrieve the private key that matches the client specs (see issue #18)
    fhe_api.generate_private_and_evaluation_keys(force=False)

    predictions = fhe_api.deserialize_decrypt_dequantize(encrypted_prediction)
    return {
        "negative": predictions[0][0],
        "neutral": predictions[0][1],
        "positive": predictions[0][2],
    }


demo = gr.Blocks(theme=gr.themes.Default(primary_hue="yellow", secondary_hue="red"))


print("Starting the demo...")
print(demo)
with demo:
    print("entered demo")
    with gr.Row():
        gr.Markdown(
            """
            <nav style="background-color: #333; padding: 1em;">
                <h1 style="color: white; text-align: center;">Encrypted Data on LLMs</h1>
            </nav>
            """
        )


    gr.Markdown(
        """
        <p align="center">
        </p>
        <p align="center">
        </p>
        """
    )

    
    gr.Markdown(
    """
# Welcome, Gulshan Kumar!!
"""
    )

    gr.Markdown("# Generate Your Keys")
    b_gen_key_and_install = gr.Button("Generate the keys and send public part to server")
    evaluation_key = gr.Textbox(
        label="Evaluation key (truncated):",
        max_lines=4,
        interactive=False,
    )

    user_id = gr.Textbox(
        label="",
        max_lines=4,
        interactive=False,
        visible=False
    )

    gr.Markdown("# Input Text(Client Side)")
    
    gr.Markdown(
        "Input a feedback to get sentiment."
    )
    text = gr.Textbox(label="Enter a message:", value="Enter feedback here")

    gr.Markdown("# Encode the data with the private key")
    b_encode_quantize_text = gr.Button(
        "Encode, quantize and encrypt the text with transformer vectorizer, and send to server"
    )

    with gr.Row():
        encoding = gr.Textbox(
            label="Transformer representation:",
            max_lines=4,
            interactive=False,
        )
        quantized_encoding = gr.Textbox(
            label="Quantized encrypted data representation:", max_lines=4, interactive=False
        )
        encrypted_quantized_encoding = gr.Textbox(
            label="Quantized Encrypted input data for model:",
            max_lines=4,
            interactive=False,
        )

    gr.Markdown("# Running FHE Evaluation(Server Side)")
    gr.Markdown(
        "The encrypted value is received by the server which will predict output on encrypted key, and then will send the encrypted prediction to client"
    )

    b_run_fhe = gr.Button("Run FHE execution:")
    encrypted_prediction = gr.Textbox(
        label="Encrypted prediction (truncated):",
        max_lines=4,
        interactive=False,
    )

    gr.Markdown("# Decrypt the sentiment(Client Side)")
    gr.Markdown(
        "The encrypted sentiment is sent back to client, who can finally decrypt it with its private key. Only the client is aware of the original tweet and the prediction."
    )
    b_decrypt_prediction = gr.Button("Decrypt prediction")

    labels_sentiment = gr.Label(label="Sentiment:")

    # Button for key generation
    b_gen_key_and_install.click(keygen, inputs=[], outputs=[evaluation_key, user_id])

    # Button to quantize and encrypt
    b_encode_quantize_text.click(
        encode_quantize_encrypt,
        inputs=[text, user_id],
        outputs=[
            encoding,
            quantized_encoding,
            encrypted_quantized_encoding,
        ],
    )

    # Button to send the encodings to the server using post at (localhost:8000/predict_sentiment)
    b_run_fhe.click(run_fhe, inputs=[user_id], outputs=[encrypted_prediction])

    # Button to decrypt the prediction on the client
    b_decrypt_prediction.click(decrypt_prediction, inputs=[user_id], outputs=[labels_sentiment])
    
demo.launch(share=True)
