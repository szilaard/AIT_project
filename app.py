from flask import Flask, render_template, request
from celery.result import AsyncResult
from utils.model import conv_classifier
from utils.predict import predict_from_youtube_link
from celery import Celery
import time

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    #celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)

conv_model = conv_classifier()
conv_model.load_weights("models/conv3.h5")
mapping = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']



@celery.task(name='prediction')
def processing(url):
    global genre
    genre, title = predict_from_youtube_link(conv_model, mapping, url)
    return genre, title


@app.route('/', methods=['GET'])
def index():          
    return render_template('index.html')

@app.route('/load_video', methods=['POST'])
def load_image():
    url = request.form['video-url']
    task = processing.delay(url)
    async_result = AsyncResult(id=task.task_id, app=celery)
    processing_result1, processing_result2 = async_result.get()
    return render_template('index.html', genre=processing_result1, title=processing_result2)
    

if __name__ == '__main__':
    app.run(host="192.168.1.208", port=5000)
