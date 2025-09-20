# app_gradio_api.py
import gradio as gr
import requests
import traceback

API_URL = "http://127.0.0.1:8000/chat"  # make sure FastAPI is running on this URL

def chat_api(user_input, history):
    history = history or []
    try:
        resp = requests.post(API_URL, json={"user_input": user_input}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "(no reply)")
        docs = data.get("docs", [])
        # show citations below the reply
        cited = ""
        if docs:
            cited = "\n\nCited docs:\n" + "\n".join([f"- {d}" for d in docs])
        history.append({"role":"user","content":user_input})
        history.append({"role":"assistant","content": reply + cited})
        return history, history
    except requests.exceptions.RequestException as e:
        err = f"API request failed: {str(e)}"
        # include full traceback in assistant reply for debugging (dev only)
        tb = traceback.format_exc()
        history.append({"role":"user","content":user_input})
        history.append({"role":"assistant","content": err + "\n\n" + tb})
        return history, history

with gr.Blocks() as demo:
    gr.Markdown("## WeRize POC Chat (API mode)")
    chatbot = gr.Chatbot(type="messages")
    txt = gr.Textbox(placeholder="Ask about loan repayment, KYC, claims...")
    clear = gr.Button("Clear")

    txt.submit(chat_api, [txt, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
