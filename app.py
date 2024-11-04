import tkinter as tk
from tkinter import messagebox
from chatbot import get_chatbot_response
from ml_model import predict, train_model
from web_scraper import run_scraper

class SportsBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sports Prediction Chatbot")
        self.root.geometry("500x600")

        # Chat Frame
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(pady=10)

        self.chat_log = tk.Text(self.chat_frame, height=15, width=60, state="disabled", bg="#f4f4f4")
        self.chat_log.pack()

        self.user_message = tk.Entry(self.chat_frame, width=50)
        self.user_message.pack(side=tk.LEFT, padx=5)
        
        send_button = tk.Button(self.chat_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.LEFT)

        # Prediction Frame
        self.prediction_frame = tk.Frame(self.root)
        self.prediction_frame.pack(pady=10)

        tk.Label(self.prediction_frame, text="Prediction Section").pack()
        
        tk.Label(self.prediction_frame, text="Feature 1:").pack()
        self.feature1 = tk.Entry(self.prediction_frame, width=20)
        self.feature1.pack()

        tk.Label(self.prediction_frame, text="Feature 2:").pack()
        self.feature2 = tk.Entry(self.prediction_frame, width=20)
        self.feature2.pack()

        tk.Label(self.prediction_frame, text="Feature 3:").pack()
        self.feature3 = tk.Entry(self.prediction_frame, width=20)
        self.feature3.pack()

        predict_button = tk.Button(self.prediction_frame, text="Predict", command=self.make_prediction)
        predict_button.pack(pady=5)

        self.prediction_result = tk.Label(self.prediction_frame, text="", font=("Arial", 10))
        self.prediction_result.pack()

        # Data and Training Frame
        self.data_frame = tk.Frame(self.root)
        self.data_frame.pack(pady=10)

        scrape_button = tk.Button(self.data_frame, text="Scrape Latest Data", command=self.scrape_data)
        scrape_button.pack(side=tk.LEFT, padx=5)

        train_button = tk.Button(self.data_frame, text="Retrain Model", command=self.retrain_model)
        train_button.pack(side=tk.LEFT, padx=5)

    def send_message(self):
        user_message = self.user_message.get()
        if user_message:
            # Display user message
            self.chat_log.config(state="normal")
            self.chat_log.insert(tk.END, f"You: {user_message}\n")
            self.chat_log.config(state="disabled")
            self.user_message.delete(0, tk.END)
            
            # Get bot response
            bot_response = get_chatbot_response(user_message)
            self.chat_log.config(state="normal")
            self.chat_log.insert(tk.END, f"Bot: {bot_response}\n")
            self.chat_log.config(state="disabled")

    def make_prediction(self):
        try:
            features = [
                float(self.feature1.get()),
                float(self.feature2.get()),
                float(self.feature3.get())
            ]
            prediction = predict(features)
            self.prediction_result.config(text=f"Prediction: {prediction}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for features.")

    def scrape_data(self):
        run_scraper()
        messagebox.showinfo("Data Update", "Data scraped and updated successfully.")

    def retrain_model(self):
        train_model()
        messagebox.showinfo("Model Training", "Model retrained successfully.")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SportsBotApp(root)
    root.mainloop()
