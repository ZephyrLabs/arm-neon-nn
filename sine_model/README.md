# Sine approximation model 

* Precision: FP32
* Input: [0, 2π]
* Output: [-1, 1]

Run:
```
$ make
$ ./sine_model
```

### How was this model trained and inferenced ?
1. Train model on tensorflow, and export the tflite model.
2. Using netron to extract the weights and biases as .npy files
3. Use the npy_convert script to convert the npy files into raw bin files.
4. Load the weights into memory allocated during runtime.
5. run the necessary microkernel operations and inference.
