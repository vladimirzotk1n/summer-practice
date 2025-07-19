import base64
from openai import OpenAI


"""
Начальная версия без батчей:
"""
def ollama_client(model, api_key, base_url, **kwargs):
    client = OpenAI(api_key=api_key, base_url=base_url)

    def query_ollama(prompt, image=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if image is not None:
            base64_image = base64.b64encode(image).decode("utf-8")
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=400,
        )

        return response

    return query_ollama


api_url = "http://qwenvl.ml.n19.int.norsi-trans.ru/v1/"
api_key = "pee-pee-poo-poo"
model = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
prompt = (
    "Extract all text from the image exactly as it appears visually, "
    "following strict reading order: top-to-bottom, left-to-right. "
    "Preserve line breaks, spacing, and punctuation. Do not modify, "
    "reorder, or summarize the text—output only the raw extracted "
    "content in its original form"
)
client = ollama_client(
        base_url=api_url,
        api_key=api_key,
        model=model,
    )

def predict(image_path):
    print("Sending request")

    with open(image_path, "rb") as f:
        image = f.read()


    response = client(
        prompt=prompt,
        image=image,
    )
    predictions = response.choices[0].message.content
    return ' '.join(predictions.split())











"""
def ollama_client(model, api_key, base_url, **kwargs):
    client = OpenAI(api_key=api_key, base_url=base_url)

    def query_ollama(prompt, images=None, image_paths=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images:
            if image is not None:
                base64_image = base64.b64encode(image).decode("utf-8")
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                )

        responses = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=400,
        )

        return responses

    return query_ollama


api_url = "http://qwenvl.ml.n19.int.norsi-trans.ru/v1/"
api_key = "pee-pee-poo-poo"
model = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

prompt = (
    "You will receive multiple images. For each image, extract all visible text.\n\n"
    "Return the result as a list. Each element must be a string on image, so result[i] = text on image[i] on request:\n\n"
)




client = ollama_client(
        base_url=api_url,
        api_key=api_key,
        model=model,
    )

def predict(image_paths):
    print("Sending request")

    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image = f.read()
            images.append(image)


    response = client(
        prompt=prompt,
        images=images,
    )
    predictions = response.choices[0].message.content

    # results = [line.strip() for line in predictions.strip().split("\n") if line.strip()]

    return predictions
    """

