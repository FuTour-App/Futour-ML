# Image Genre Classisification

## Training & Deployment
See image_genre_classifier.ipynb
The resulting model is saved on model/image_genre_classifier_v3.keras

## Deployment:

- Install dependencies (tensorflow, pillow, flask)
```
pip install -r requirements.txt
```
- If you encounter any error, try to install:
```
pip install tensorflow pillow flask gunicorn
```
- Run the Flask web server:
```
flask --app app run
```
- Try it out:
```
http://127.0.0.1:5000
```
