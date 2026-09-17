import tkinter as tk
from tkinter import ttk, scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_data = [
    {
        "question": "What is CodeAlpha?",
        "answer": "CodeAlpha is a software development and technology learning platform offering internships and real-world tech experience."
    },
    {
        "question": "What is the duration of this AI internship?",
        "answer": "The CodeAlpha AI internship typically lasts for 1 month."
    },
    {
        "question": "How many tasks are required to get the internship certificate?",
        "answer": "You need to complete and submit at least 2 or 3 tasks to be eligible for the internship certificate."
    },
    {
        "question": "Where should I submit my completed project code?",
        "answer": "You must push your code to a public GitHub repository named 'CodeAlpha_ProjectName' and submit via the provided Google Form."
    },
    {
        "question": "Is posting a video on LinkedIn mandatory?",
        "answer": "Yes, you must share a video explanation of your project on LinkedIn, tag @CodeAlpha, and include the GitHub repository link."
    },
    {
        "question": "What happens if I submit only one task?",
        "answer": "Submitting only one task is considered incomplete, and no certificate will be issued."
    },
    {
        "question": "Who can I contact for help regarding the internship?",
        "answer": "You can reach out via WhatsApp at +91 9336576683 or email services@codealpha.tech."
    }
]

faq_questions = [item["question"] for item in faq_data]
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
tfidf_matrix = vectorizer.fit_transform(faq_questions)

def get_bot_response(user_query):
    query_vec = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]

    if highest_score < 0.25:
        return "I'm sorry, I couldn't find a matching FAQ. Please ask questions related to the CodeAlpha internship."
    
    return faq_data[best_match_idx]["answer"]

def send_message():
    user_text = user_input.get().strip()
    if not user_text:
        return

    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, f"You: {user_text}\n\n", "user")
    
    response = get_bot_response(user_text)
    chat_box.insert(tk.END, f"Bot: {response}\n\n", "bot")
    
    chat_box.config(state=tk.DISABLED)
    chat_box.yview(tk.END)
    user_input.delete(0, tk.END)

root = tk.Tk()
root.title("CodeAlpha - FAQ Chatbot")
root.geometry("520x560")
root.resizable(False, False)

root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)

tk.Label(root, text="FAQ Chatbot (NLP Powered)", font=("Arial", 14, "bold")).pack(pady=10)

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=55, height=20, font=("Arial", 10))
chat_box.tag_config("user", foreground="#1565C0", font=("Arial", 10, "bold"))
chat_box.tag_config("bot", foreground="#2E7D32")
chat_box.pack(padx=15, pady=5)

chat_box.insert(tk.END, "Bot: Hello! I am your CodeAlpha FAQ Assistant. Ask me anything about the internship.\n\n", "bot")
chat_box.config(state=tk.DISABLED)

bottom_frame = tk.Frame(root)
bottom_frame.pack(padx=15, pady=10, fill=tk.X)

user_input = tk.Entry(bottom_frame, font=("Arial", 11), width=38)
user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
user_input.bind("<Return>", lambda event: send_message())

send_btn = tk.Button(bottom_frame, text="Send", command=send_message, bg="#1976D2", fg="white", font=("Arial", 10, "bold"), padx=12)
send_btn.pack(side=tk.RIGHT)

root.mainloop()
