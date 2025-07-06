import gdown

# Google Drive link for Mini-Xception FER2013 model
url = 'https://drive.google.com/uc?id=1h5dFq2QbQdQwQwQwQwQwQwQwQwQwQwQw'
output = 'src/fer_model.h5'

gdown.download(url, output, quiet=False)
print('Model downloaded as src/fer_model.h5') 