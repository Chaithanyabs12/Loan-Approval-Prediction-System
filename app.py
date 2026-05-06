from flask import Flask, render_template, request, redirect, session
import sqlite3
import pickle
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)
app.secret_key = "secret123"

# =========================
# DB SETUP
# =========================
def create_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

create_db()

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("loan_model.pkl", "rb"))

# =========================
# LOGIN
# =========================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = u
            return redirect("/dashboard")
        else:
            return "Invalid Login ❌"

    return render_template("login.html")


# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username,password) VALUES (?,?)",(u,p))
            conn.commit()
        except:
            return "User already exists ❌"
        conn.close()
        return redirect("/")

    return render_template("register.html")


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")


# =========================
# INPUT PAGE
# =========================
@app.route("/input")
def input():
    if "user" not in session:
        return redirect("/")
    return render_template("input.html")


# =========================
# PREDICT (UPDATED 🔥)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    try:
        gender = int(request.form.get("gender", 1))
        married = int(request.form.get("married", 1))
        education = int(request.form.get("education", 1))
        income = float(request.form.get("income", 0))
        loan = float(request.form.get("loan", 0))
        credit = int(request.form.get("credit", 1))
    except:
        return "Invalid Input ❌"

    # =========================
    # MODEL
    # =========================
    result = model.predict([[gender, married, education, income, loan, credit]])
    prob = model.predict_proba([[gender, married, education, income, loan, credit]])[0]

    approval = round(prob[1]*100,2)
    rejection = round(prob[0]*100,2)

    output = "Loan Approved" if result[0]==1 else "Loan Rejected"

    # =========================
    # SCATTER GRAPH
    # =========================
    plt.figure(figsize=(5,4))
    plt.scatter(income, loan, color="#22c55e", s=120)
    plt.xlabel("Income")
    plt.ylabel("Loan Amount")
    plt.title("Loan Visualization")

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # =========================
    # DONUT CHART (REAL %)
    # =========================
    plt.figure(figsize=(4,4))

    plt.pie(
        [approval, rejection],
        colors=["#22c55e","#ef4444"],
        startangle=90,
        wedgeprops=dict(width=0.4),
        autopct='%1.1f%%'
    )

    plt.title("Approval Probability")

    img2 = io.BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    pie_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    return render_template(
        "result.html",
        result=output,
        plot_url=plot_url,
        pie_url=pie_url,
        approval=approval,
        rejection=rejection
    )


# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)