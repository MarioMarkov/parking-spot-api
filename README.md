# parking-spot-api

### Parking spot detection API build with FAST API 

#### deployed on: https://parking-spot-api-2.onrender.com

The api has a UI on the / route on which you can downlaod exmplae image and aanotation that can be uploaded again to get a prediction. 
The prediction will take about 30 seconds due to the fact that the model is deployed on a server with very weak ressources.

There are three models that one can use:
 - m_alex_net.pth normal float precission model
 - model_qunatized_static.pt statically quantized model
 - onxx_malex_net.onnx model designed to run on onxx runtime

To install packages:
```
pip install -r requirements.txt
```

To start it: 
```
uvicorn main:app
```
