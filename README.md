# Arm Neon NN 

Simple neural network microkernels in C accelerated with ARMv8.2-a Neon vector intrinsics.

Examples:
* [sine_model](./src/sine_model/), multiple dense layers with ReLU activation layers in between, approximates the value of sin(x).

### Why these microkernels ?
If you are looking to deploy lightweight ML models, or maybe just need optimized ways to process large 
data operations on embedded targets without the weighted size of existing libraries, these microkernels 
offer a stripped down, simple way to implement them with simple modifications as per your platform.

With modfications it can target other architectures like:
* ARMv7-a (eg. Cortex-A7)
* Armv8-a (eg. Cortex-A53)

(they don't support the f16 data type, f32 microkernels work fine)

With much heavier modification for architectures like:
* ARMv7-m (eg. Cortex-M4 with DSP instructions)
* ARMv8-m (eg. Cortex-M55 with Helium instructions)

(they have a slightly different instruction set, but the principle is the same)

### TBD:
* Suport for quantized data types like, s8, u8, s16, u16 etc.
* More microkernels like Convolution 1D, 2D, Maxpooling etc.
* More runtime examples.
* SVE2 (with ARMv9) and Helium (ARMv8.1-m) in the future ?
