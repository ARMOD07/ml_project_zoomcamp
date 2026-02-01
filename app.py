import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Railway 🚀"

# نقطة تشغيل التطبيق
if __name__ == "__main__":
    # Railway يعطي PORT في متغير البيئة
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)