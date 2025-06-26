import openai

# LM Studio endpoint
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"

def generate_response(query: str, context: str, model="local-model"):
    prompt = f"""Você é um assistente. Com as informações fornecidas, responda a pergunta do usuário.

### Context
{context}

### Question
{query}

### Answer
"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )

        # ✅ Safer access via dictionary
        return response["choices"][0]["message"]["content"].strip() # type: ignore

    except Exception as e:
        print("❌ Failed to get response from LM Studio:")
        print(f"{type(e).__name__}: {e}")
        return "[ERROR: No response generated]"
