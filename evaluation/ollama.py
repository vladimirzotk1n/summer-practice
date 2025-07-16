import base64
from openai import OpenAI


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
# api_key = ?
model = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
prompt = "Extract all text from the given image. Write only text from image with no additional information"

client = ollama_client(
        base_url=api_url,
        api_key=api_key,
        model=model,
    )

def predict(image_path: str) -> str:
    print("Sending request")

    with open(image_path, "rb") as f:
        image = f.read()


    response = client(
        prompt=prompt,
        image=image,
    )
    predictions = response.choices[0].message.content
    return ' '.join(predictions.split())