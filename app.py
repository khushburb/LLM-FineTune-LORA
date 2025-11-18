from flask import Flask, render_template_string, request
from infer_phi2_with_roles import generate_answer

app = Flask(__name__)

# Simple inline HTML template for easy portability
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM Role-based Q&A</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
        .container { max-width: 700px; margin: auto; background: white; padding: 30px;
                     border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        select, textarea, button { width: 100%; margin-top: 10px; padding: 10px;
                                   border-radius: 8px; border: 1px solid #ccc; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { background: #f1f1f1; padding: 15px; border-radius: 8px; margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Role-based Q&A Demo</h1>
        <form method="POST">
            <label for="role">Select Role:</label>
            <select name="role" id="role">
                <option value="TPM">Technical Program Manager</option>
                <option value="PM">Product Manager</option>
                <option value="Software Engineer">Software Engineer</option>
                <option value="Data Scientist">Data Scientist</option>
                <option value="ML Engineer">Machine Learning Engineer</option>
            </select>
            <label for="prompt">Enter your question:</label>
            <textarea name="prompt" id="prompt" rows="5" placeholder="Ask something...">{{ prompt or '' }}</textarea>
            <button type="submit">Generate Answer</button>
        </form>
        {% if answer %}
        <div class="result">
            <strong>Answer ({{ role }}):</strong><br>
            {{ answer }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    prompt = None
    role = None
    if request.method == "POST":
        prompt = request.form.get("prompt")
        role = request.form.get("role")
        answer = generate_answer(prompt, role)
    return render_template_string(HTML_TEMPLATE, answer=answer, prompt=prompt, role=role)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)