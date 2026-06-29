# Fourier Embeddings

## Background Information
- **Implicit Neural Representations (INRs)**: method of parameterizing signals, such as images, 3D shapes, etc, as continuous mathematical functions inside a neural network. Take an input coordinate and output the exact value at that location.
    - because data is usually discretized, we are unable to scale its resolution (results in bluriness). additionally, it is indifferentiable, restricting its application in optimization and machine learning in general.
    - when we treat data as a continuous function f(x) = y,we can circumvent this paradigm. 
        - x: spatial or temporal coordinates 
        - y: value at said coordinate
        - MLPs are used to represent the data in an INR
    - *Example Applications*:
        - Neural Radiance Fields (NeRFs): used to represent complex 3D scenes. Learns to predict the color and density of space by feeding 3D coordinates and viewing directions into the network.
        - Super-Resolution and Denoising: because of the continuity of data, it enables seamless upscaling to unseen resolutions and zero-shot noise reduction
        - 3D Surface Reconstruction: used to represented 3D surfaces with compplex topology instead of relying on rigid point clouds and meshes.

- **Fourier Feature Mapping**: 