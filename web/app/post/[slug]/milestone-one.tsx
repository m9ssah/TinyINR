import { ArrowDown } from "lucide-react";

function Keyword({ children }: { children: React.ReactNode }) {
  return <span className="font-semibold text-accent">{children}</span>;
}

function PipelineStep({
  title,
  detail,
}: {
  title: string;
  detail?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface px-5 py-4 text-center">
      <div className="font-mono text-sm font-semibold text-primary">{title}</div>
      {detail ? (
        <div className="mt-1 font-mono text-xs text-secondary">{detail}</div>
      ) : null}
    </div>
  );
}

function PipelineArrow() {
  return (
    <div className="flex justify-center py-2 text-accent">
      <ArrowDown className="h-5 w-5" />
    </div>
  );
}

function PipelineDiagram() {
  return (
    <div className="my-10 rounded-2xl border border-border bg-background p-5">
      <div className="mb-4 font-mono text-xs uppercase tracking-widest text-secondary">
        Milestone 1 data path
      </div>
      <PipelineStep title="Image" />
      <PipelineArrow />
      <PipelineStep title="CoordinateBatch" />
      <PipelineArrow />
      <PipelineStep
        title="Coordinates + RGB targets"
        detail="coordinates [B, N, 2] / values [B, N, 3]"
      />
      <PipelineArrow />
      <PipelineStep title="FourierEmbedding" />
      <PipelineArrow />
      <PipelineStep title="Embedded coordinate features" />
      <PipelineArrow />
      <PipelineStep title="Neural network" />
      <PipelineArrow />
      <PipelineStep title="Predicted RGB values" />
    </div>
  );
}

function FitTogetherDiagram() {
  return (
    <div className="my-8 grid gap-4 md:grid-cols-3">
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="font-mono text-sm font-semibold text-accent">Tensor</div>
        <p className="mt-2 text-sm text-secondary">
          Stores numbers, shape metadata, strides, and contiguous CPU memory.
        </p>
      </div>
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="font-mono text-sm font-semibold text-accent">
          CoordinateBatch
        </div>
        <p className="mt-2 text-sm text-secondary">
          Gives numbers spatial meaning as coordinate-value training pairs.
        </p>
      </div>
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="font-mono text-sm font-semibold text-accent">
          FourierEmbedding
        </div>
        <p className="mt-2 text-sm text-secondary">
          Transforms coordinates into richer inputs for an INR model.
        </p>
      </div>
    </div>
  );
}

export function MilestoneOnePostBody() {
  return (
    <>
      <p>
        This week was focused on building the{" "}
        <Keyword>CPU-side data path</Keyword> for TinyINR. Before implementing
        neural-network layers or CUDA kernels, we needed a reliable way to store
        tensors, represent image pixels as coordinate-value pairs, and enrich
        those coordinates with Fourier features.
      </p>

      <p>
        By the end of the week, the pipeline could take an image, convert it
        into coordinates and RGB targets, and transform each coordinate into
        features suitable for an <Keyword>implicit neural representation</Keyword>
        model.
      </p>

      <PipelineDiagram />

      <p>
        The main idea behind TinyINR is that an image can be represented as a
        function. Instead of storing an image only as a grid of pixels, we can
        train a model to learn a mapping from location to color:
      </p>

      <pre>
        <code>{`f(x, y) -> [R, G, B]`}</code>
      </pre>

      <h2>Tensor Core: the storage layer</h2>

      <p>
        The lowest layer of TinyINR is the <Keyword>Tensor</Keyword> class. A
        tensor is essentially a multidimensional array of numbers. Our
        implementation is a small CPU-first <Keyword>float32</Keyword> tensor
        built around three private fields:
      </p>

      <pre>
        <code>{`std::vector<float> data_;
std::vector<int64_t> shape_;
std::vector<int64_t> strides_;`}</code>
      </pre>

      <p>
        <Keyword>data_</Keyword> is the actual memory buffer. It stores all
        tensor values in one contiguous one-dimensional array.{" "}
        <Keyword>shape_</Keyword> records the logical dimensions of the tensor.
        For example:
      </p>

      <pre>
        <code>{`Tensor x({2, 3});              // shape [2, 3]
Tensor coords({1, 1024, 2});   // shape [batch, points, coordinate_dim]`}</code>
      </pre>

      <p>
        The third field, <Keyword>strides_</Keyword>, explains how to convert a
        multidimensional index into a flat index inside <Keyword>data_</Keyword>.
        For a tensor with shape <code>[2, 3]</code>, the row-major strides are:
      </p>

      <pre>
        <code>{`shape:   [2, 3]
strides: [3, 1]`}</code>
      </pre>

      <p>This means that accessing <code>x[1, 2]</code> becomes:</p>

      <pre>
        <code>{`flat index = 1 * 3 + 2 * 1 = 5`}</code>
      </pre>

      <p>
        So the value is stored at <code>data_[5]</code>. This stride system is
        important because it gives the rest of the project a consistent way to
        think about multidimensional data while still keeping the actual memory
        layout simple and contiguous.
      </p>

      <h2>Coordinate Batch</h2>

      <p>
        Once we had a tensor container, the next step was giving the numbers
        meaning. That is the role of <Keyword>CoordinateBatch</Keyword>.
      </p>

      <p>
        A normal image is usually represented as a grid of pixel values. For
        example, a pixel at row 10 and column 20 might have an RGB value like
        <code>[0.8, 0.3, 0.1]</code>. In an implicit neural representation, we
        instead treat that pixel as a training example:
      </p>

      <pre>
        <code>{`input coordinate: [x, y]
target value:     [R, G, B]`}</code>
      </pre>

      <p>
        So the image becomes a dataset of coordinate-value pairs. The model is
        trained to answer the question: given this location, what color should
        be there?
      </p>

      <p>
        For an image with height <Keyword>H</Keyword>, width <Keyword>W</Keyword>,
        and three color channels, CoordinateBatch produces:
      </p>

      <pre>
        <code>{`coordinates: [B, H * W, 2]
values:      [B, H * W, 3]`}</code>
      </pre>

      <p>For example, a single 32 by 32 RGB image becomes:</p>

      <pre>
        <code>{`coordinates: [1, 1024, 2]
values:      [1, 1024, 3]`}</code>
      </pre>

      <p>
        Each coordinate corresponds to one pixel, and each value stores that
        pixel&apos;s RGB target. A simplified example looks like:
      </p>

      <pre>
        <code>{`pixel (0, 0)   -> coordinate [-1.0, -1.0] -> RGB [r, g, b]
pixel (31, 0)  -> coordinate [ 1.0, -1.0] -> RGB [r, g, b]
pixel (0, 31)  -> coordinate [-1.0,  1.0] -> RGB [r, g, b]`}</code>
      </pre>

      <p>
        We normalize coordinates into a predictable range, such as{" "}
        <Keyword>[-1, 1]</Keyword>, instead of using raw pixel positions like 0
        to 511. This makes the input scale consistent across image sizes and
        also works better with the Fourier embedding step that comes next.
      </p>

      <h2>Fourier Embedding</h2>

      <p>
        The third module we built was <Keyword>FourierEmbedding</Keyword>. Its
        job is to make coordinates more expressive before they are passed into a
        neural network.
      </p>

      <p>
        Raw coordinates like <code>[x, y]</code> tell the model where a point is,
        but they are often too simple for representing fine image details.
        Neural networks tend to learn smoother, lower-frequency patterns more
        easily than sharp edges or textures. <Keyword>Fourier features</Keyword>{" "}
        help by expanding each coordinate into multiple sine and cosine values
        at different frequencies.
      </p>

      <p>
        For a single coordinate value <code>x</code>, a Fourier embedding might
        produce:
      </p>

      <pre>
        <code>{`sin(pi * x), cos(pi * x),
sin(2 * pi * x), cos(2 * pi * x),
sin(4 * pi * x), cos(4 * pi * x),
sin(8 * pi * x), cos(8 * pi * x)`}</code>
      </pre>

      <p>
        For a 2D coordinate <code>[x, y]</code>, the embedding is applied across
        both coordinate dimensions. So instead of passing only two values into
        the model, we pass a larger feature vector containing sinusoidal
        information about the position.
      </p>

      <p>If the input coordinate tensor has shape:</p>

      <pre>
        <code>{`[B, N, D]`}</code>
      </pre>

      <p>
        where <Keyword>B</Keyword> is batch size, <Keyword>N</Keyword> is the
        number of points, and <Keyword>D</Keyword> is the coordinate dimension,
        then the embedded output has shape:
      </p>

      <pre>
        <code>{`[B, N, 2 * D * F]`}</code>
      </pre>

      <p>
        where <Keyword>F</Keyword> is the number of frequency bands. The factor
        of 2 comes from using both sine and cosine.
      </p>

      <p>
        Fourier embedding does not add new image information by itself. Instead,
        it changes the representation of position so that the later MLP has an
        easier time fitting high-frequency detail. In the context of TinyINR,
        this is what allows a coordinate-based model to represent more than just
        smooth gradients. It gives the network a richer basis for reconstructing
        edges, patterns, and texture.
      </p>

      <h2>How the pieces fit together</h2>

      <FitTogetherDiagram />

      <p>In code, the intended flow looks like this:</p>

      <pre>
        <code>{`Tensor image = load_image(...);

CoordinateBatch batch = CoordinateBatch::from_image(image);

Tensor coordinates = batch.coordinates();  // [B, N, 2]
Tensor targets = batch.values();           // [B, N, 3]

FourierEmbedding embedding(num_frequencies);
Tensor features = embedding.forward(coordinates);

// Later:
// Tensor predictions = mlp.forward(features);`}</code>
      </pre>

      <h2>Upcoming</h2>

      <p>
        Next week, we will use the Fourier embedding module as our first{" "}
        <Keyword>CUDA</Keyword> target. The goal is to implement a GPU
        coordinate embedding kernel, compare its output against the CPU version,
        and benchmark CPU runtime versus CUDA runtime.
      </p>
    </>
  );
}
