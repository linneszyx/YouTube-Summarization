import gradio as gr
from modules.summarize import summarize_video
from modules.qa import generate_answer
from modules.ui_custom import custom_css

with gr.Blocks(css=custom_css, title="🎬 YouTube Video Summarizer & Q&A") as interface:
    gr.Markdown("# 🎬 YouTube Video Summarizer \nSummarize YouTube videos and ask questions easily!")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 🔗 Video Details")
            video_url = gr.Textbox(label="YouTube Video URL", placeholder="Paste your YouTube video link...")
            language_dropdown = gr.Dropdown(choices=["en", "es", "fr", "de", "hi", "zh"], label="Transcript Language", value="en")
            
            gr.Markdown("---")
            gr.Markdown("### 🎛️ Actions")
            summarize_btn = gr.Button("📄 Generate Summary", variant="primary")
            question_btn = gr.Button("❓ Ask a Question", variant="secondary")
            gr.Markdown("---")
            gr.Markdown("### 💬 Ask a Question")
            question_input = gr.Textbox(label="Your Question", placeholder="E.g., What is the key takeaway?")

        with gr.Column(scale=2):
            gr.Markdown("### 📝 Video Summary")
            summary_output = gr.Textbox(placeholder="Summary will appear here...", lines=8, interactive=False)
            gr.Markdown("### 💬 Answer to Your Question")
            answer_output = gr.Textbox(placeholder="Answer will appear here...", lines=6, interactive=False)
            transcript_status = gr.Textbox(label="Status", visible=False, interactive=False)

    summarize_btn.click(
        summarize_video,
        inputs=[video_url, language_dropdown],
        outputs=summary_output
    )

    question_btn.click(
        generate_answer,
        inputs=[video_url, question_input, language_dropdown],
        outputs=answer_output
    )

interface.launch(server_name="0.0.0.0", server_port=8900, share=True)
