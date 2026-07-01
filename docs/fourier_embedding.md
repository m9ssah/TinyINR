# Fourier Embeddings

## Background Information
- **Implicit Neural Representations (INRs)**: method of parameterizing signals, such as images, 3D shapes, etc, as continuous mathematical functions inside a neural network. Take an input coordinate and output the exact value at that location.
    - because data is usually discretized, we are unable to scale its resolution (results in bluriness). additionally, it is indifferentiable, restricting its application in optimization and machine learning in general.
    - when we treat data as a continuous function f(x) = y,we can circumvent this paradigm. 
        - x: spatial or temporal coordinates 
        - y: value at said coordinate
        - Generally, MLPs are used to represent the data in an INR
    - *Example Applications*:
        - Neural Radiance Fields (NeRFs): used to represent complex 3D scenes. Learns to predict the color and density of space by feeding 3D coordinates and viewing directions into the network.
        - Super-Resolution and Denoising: because of the continuity of data, it enables seamless upscaling to unseen resolutions and zero-shot noise reduction
        - 3D Surface Reconstruction: used to represented 3D surfaces with compplex topology instead of relying on rigid point clouds and meshes.

- **Fourier Feature Mapping**: (specific method preprocessing technqiue) method of encoding input coordinates into a higher-dimensional space by way of sinusoidal functions before passing them to a neural network
    - Its connected to INRs because it enables standard MLPs to learn fine spatial detail and high-frequency patterns. 
        - MLPs tend to exhibit **spectral bias**, i.e. learns low-frequency smooth functions at a more rapid rate as opposed to other functions. FFM overcome this by making high-frequency information easier for the network to model.
    - A point x is transformed into a vector of sine and cosine values at multiple frequencies. 
        - A common formulation: $$γ(x)=[sin(2πBx),cos(2πBx)]$$
        where B is the matrix of frequencies.

- **Neural Radiance Fields (NeRFs)**: method for representing 3D scenes as a continuous function that can generate realistic images from new viewpoints.
    - It represents a scene using a neural network instead of a voxel grid or traditional mesh. Given a 3D position and viewing direction, the network predicts the density at a location alongside the light emitted toward the camera from that location.
        - To render an image, the rays are traced from each pixel into the scene. The predicted density and radiance are combined using volume rendering to synthesize the final pixel. 
        - During training, the network compares rendered images against others which have been taken from a known camera angle. 
    - Its importance lies in its demonstrated ability of leveraging a compact neural network to capture fine geometric detail, complex lighting, and view-dependent effects like glossiness and reflections. 

- **Sinusoidal Representation Networks (SIREN)**: A neural network architecture that uses *periodic activation functions*, specifically sine functions, instead of the more traditional ReLU. 
    - It replaces standard non-linearitiess with the sine function throughout the network, giving the network the ability to capture high-frequency variations.
    - Additionally, it uses a specialized weight initialization technique that allows for stable gradients.
    - Consider using when the neural network is considered as an implicit continuous function rather than a classifier. 