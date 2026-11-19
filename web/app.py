from flask import Flask, render_template
from web.services import WEB_DIR
from web.routes.dataset import dataset_bp
from web.routes.upload import upload_bp

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "assets"),
)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

app.register_blueprint(dataset_bp)
app.register_blueprint(upload_bp)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)