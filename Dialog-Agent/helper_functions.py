import sqlite3

# Setup and DB functions
def init_db():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY,
        user_message TEXT,
        ai_response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def log_conversation(user_message, ai_response):
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO conversations (user_message, ai_response) VALUES (?, ?)
    """, (user_message, ai_response))
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs

# Conversation Buffer
class ConversationSummaryBuffer:
    def __init__(self, max_length=5):
        self.buffer = []
        self.max_length = max_length

    def add(self, message):
        if len(self.buffer) >= self.max_length:
            self.buffer.pop(0)
        self.buffer.append(message)

    def get_last_five(self):
        return self.buffer[-5:]

    def get(self):
        return self.buffer
