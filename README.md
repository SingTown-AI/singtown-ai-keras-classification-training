# SingTown AI Keras Classification

## Support Models

- MobileNetV1 0.25 128
- MobileNetV1 0.25 160
- MobileNetV1 0.25 192
- MobileNetV1 0.25 224
- MobileNetV1 0.50 128
- MobileNetV1 0.50 160
- MobileNetV1 0.50 192
- MobileNetV1 0.50 224
- MobileNetV1 0.75 128
- MobileNetV1 0.75 160
- MobileNetV1 0.75 192
- MobileNetV1 0.75 224
- MobileNetV1 1.0 128
- MobileNetV1 1.0 160
- MobileNetV1 1.0 192
- MobileNetV1 1.0 224
- MobileNetV2 0.35 96
- MobileNetV2 0.35 128
- MobileNetV2 0.35 160
- MobileNetV2 0.35 192
- MobileNetV2 0.35 224
- MobileNetV2 0.50 96
- MobileNetV2 0.50 128
- MobileNetV2 0.50 160
- MobileNetV2 0.50 192
- MobileNetV2 0.50 224
- MobileNetV2 0.75 96
- MobileNetV2 0.75 128
- MobileNetV2 0.75 160
- MobileNetV2 0.75 192
- MobileNetV2 0.75 224
- MobileNetV2 1.0 96
- MobileNetV2 1.0 128
- MobileNetV2 1.0 160
- MobileNetV2 1.0 192
- MobileNetV2 1.0 224
- MobileNetV2 1.3 224
- MobileNetV2 1.4 224

## Support Deploys

- LiteRT for Microcontrollers(tflite micro)

## Test

```
# train
python -m singtown_ai.runner --host http://127.0.0.1:8000 --task 6ba7b810-9dad-11d1-80b4-00c04fd430c8 --token 1234567890

# deploy
python -m singtown_ai.runner --host http://127.0.0.1:8000 --task 68c01fc6-589f-4f56-a923-580bdad7e02d --token 1234567890
```

## Docker

```
docker build -t keras .

# train
docker run -it --rm --network="host" --gpus all keras:latest python -m singtown_ai.runner --host http://127.0.0.1:8000 --task 6ba7b810-9dad-11d1-80b4-00c04fd430c8 --token 1234567890

# deploy
docker run -it --rm --network="host" --gpus all keras:latest python -m singtown_ai.runner --host http://127.0.0.1:8000 --task 68c01fc6-589f-4f56-a923-580bdad7e02d --token 1234567890
```