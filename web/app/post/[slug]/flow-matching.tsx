import { MathDisclosure } from "@/components/MathDisclosure";
import { KatexBlock } from "@/components/KatexBlock";
import { KatexInline } from "@/components/KatexInline";

export function FlowMatchingPostBody() {
    return (
        <>
            <p>
                Generative models often seem like black boxes. How is it that sampling
                pure noise transforms into sensical images, blocks of text, or simulations?
                Somewhere between the randomness and the structure, these systems learn
                the shape of reality itself. To close this gap, we started tinyINR: we're
                chasing a holistic understanding of how generative models behave. We
                want to build explanations that preserve the underlying mathematical
                ideas while making them intuitive to visualize, reason about, and manipulate.
                Flow matching is an ideal topic to begin with.

            </p>

            <p>
                Briefly and most intuitively put, flow matching attempts to answer the following question:
            </p>

            <p className="text-center block italic text-primary text-lg">
                How can we learn a continuous process that transforms pure randomness
                into meaningful structure?
            </p>

            <p>
                More concretely, it's a method for training a model to transform noise into
                data: we define a path that interpolates between a random noise sample and a
                real training example, then teach a neural network to predict the direction of
                travel along that path at any point in time. Once trained, we can generate a
                coherent sample by starting from noise and repeatedly following the model's
                predicted direction, step by step, until we arrive at something that looks
                like it came from our data.
            </p>

            <p>
                One way to picture this is a hidden system of currents beneath a river.
                If we release objects into the water at random, each object feels a
                local push determined by the current. Yet over time, coherent motion
                emerges, and randomness resolves into structure. Flow matching gets quite
                mathematical past this point. <em>Let's break things down.</em>
            </p>


            <h2>Theory & Derivations </h2>

            <p>
                Mentions of "randomness" and "noise" keep appearing — but what do they
                actually mean, and where do they come from? We obtain them by sampling a
                point from a Gaussian noise distribution, <strong>p</strong>. This distribution is necessary
                because sampling directly from the distribution that contains our data, <strong>q</strong>,
                is intractable. To get around this, we lean on p instead, and learn the
                transformation from p to q using a neural network.

            </p>

            <p>
                Both diffusion models and flow matching models describe this transformation using
                differential equations: diffusion models are typically framed in terms of stochastic
                differential equations, while flow matching uses ordinary differential equations <strong>(ODEs)</strong>,
                whose vector field is time-dependent. Because of this time dependence, flow matching
                transforms a simple noise distribution <em>p</em> at time <em>t=0</em> into the
                complex data distribution <em>q</em> at time <em>t=1</em>.
            </p>

            <p>
                A direct transformation from <em>p</em> to <em>q</em> is intractable, because we lack global
                knowledge of the data. Instead, we sample a point from <em>p</em> and travel across
                time toward <em>q</em> by following a vector field — think of it as a current that
                pushes each point, step by step, toward the data manifold. Sampling from <em>p </em>
                makes this transformation possible precisely because the process only ever
                needs local information about the vector field at each moment; it doesn't need
                to know the full path in advance, only where to go next, until it reaches <em>q</em> at <em>t=1</em>.
            </p>

            <p>
                This takes us back to our river analogy: the vector field is the river's current,
                and our sampled point is a random object thrown into the river. The object appears
                to move randomly at first. In reality, though, it's being driven by the current
                the whole time, which eventually carries it to the ocean — or, in our case, to the
                distribution <em>q</em>.
            </p>

            <p>
                More formally, an ODE is defined by a time-dependent vector field.
            </p>

            <KatexBlock expression="u : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d" />

            <p>
                In our context, that vector field is represented by a neural network. This velocity field determines a time-dependent flow.

            </p>

            <KatexBlock expression="\psi : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d, \quad \frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)), \quad \psi_0(x) = x," />

            <p>
                where <KatexInline expression="\psi_t := \psi(t,x)" />. The velocity field <KatexInline expression="u_t" /> <strong>generates </strong>
                the probability path <KatexInline expression="p_t" /> if its flow <KatexInline expression="psi_t" /> satisfies
            </p>

            <KatexBlock expression="X_t := \psi_t(X_0) \sim p_t \quad \text{for } X_0 \sim p_0" />

            <p>
                So the goal of a flow matching model is to learn a vector field <KatexInline expression="u_t^\theta" /> whose
                flow <KatexInline expression="psi_t" /> generates a probability path <KatexInline expression="p_t" /> — one that carries a point from our
                noise distribution, <KatexInline expression="p_0 = p" />, to our data distribution, <KatexInline expression="p_1 = q" />, thereby generating
                a sample. More precisely, flow matching aims to learn the parameters <KatexInline expression="\theta" /> of a velocity
                field <KatexInline expression="u_t^\theta" /> by way of a neural network.
            </p>

            <p>
                Flow matching, then, comes down to two fundamental steps:
            </p>

            <p className="text-center block italic text-primary text-lg">
                1. Design a probability path <KatexInline expression="p_t" /> that interpolates between p and q.
            </p>
            <p className="text-center block italic text-primary text-lg">
                2. Train a velocity field <KatexInline expression="u_t^\theta" /> that generates <KatexInline expression="p_t" />, by regression.
            </p >

            <p>
                <strong>Step 1: designing the probability path.</strong> To interpolate between <em>p</em> and <em>q</em>, we lean on the marginal law of total probability:
            </p>

            <KatexBlock expression="f_X(x) = \int f_{X|Y}(x \mid y)\, f_Y(y)\, dy," />

            <p>
                where <em>X</em> and <em>Y</em> are continuous random variables. Applying this, we can define our probability path as
            </p>

            <KatexBlock expression="p_t(x) = \int p_{t|1}(x \mid x_1)\, q(x_1)\, dx_1," />

            <p>with</p>

            <KatexBlock expression="p := p_0 = \mathcal{N}(x \mid 0, I), \quad t \in [0,1]." />

            <p>
                This lets us interpolate over <em>t </em>
                between <KatexInline expression="X_0" /> and <KatexInline expression="X_1" /> by defining a random variable along the probability path:
            </p>

            <KatexBlock expression="X_t = t X_1 + (1-t) X_0 \sim p_t." />

            <p>
                <strong>Step 2: training the velocity field.</strong> We train a velocity field <KatexInline expression="u_t^\theta" /> that generates <KatexInline expression="p_t" /> by regression:
            </p>

            <KatexBlock expression="L_{FM}(\theta) = \mathbb{E}_{t, X_t} \left\| u_t^\theta(X_t) - u_t(X_t) \right\|^2." />


            <p>
                Here, the loss minimizes the expected L2 error between the predicted and true velocity
                fields, across all time values <em>t</em> and all points <KatexInline expression="X_t" /> drawn from <KatexInline expression="p_t" />.
            </p>

            <p>The problem is that this isn't realistic to compute directly: <KatexInline expression="u_t" /> depends on the full
                marginal path <KatexInline expression="p_t" />, which requires integrating over the entire data distribution <em>q</em>,
                — something we simply don't have access to in closed form. In short, we don't know the
                ground truth value of <KatexInline expression="u_t" />.
            </p>

            <p>To get around this, we condition <KatexInline expression="u_t" /> on a single data point <KatexInline expression="x_1" /> sampled from <em>q</em>:</p>

            <KatexBlock expression="X_{t|1} = t x_1 + (1-t) X_0 \sim p_{t|1}(\cdot \mid x_1) = \mathcal{N}(\cdot \mid t x_1, (1-t)^2 I)." />

            <p>Taking the derivative:</p>

            <KatexBlock expression="\frac{d}{dt} X_{t|1} = u_t(X_{t|1} \mid x_1) = \frac{x_1 - x}{1-t}." />

            <p>
                In other words, we've found the conditional velocity field <KatexInline expression="u_t(x \mid x_1)" />. Putting it all together,
                we can reformulate the flow matching loss as a conditional flow matching loss:
            </p>

            <KatexBlock expression="L_{CFM}(\theta) = \mathbb{E}_{t, X_t, X_1} \left\| u_t^\theta(X_t) - u_t(X_t \mid X_1) \right\|^2, \quad t \sim U[0,1],\ X_0 \sim p,\ X_1 \sim q." />

            <p>As wanted.</p>


            {/* <MathDisclosure title="A simple flow matching view">
                <p>
                    One useful way to think about the target is a point moving from x to
                    y over time t:
                </p>
                <pre>
                    <code>{`x(t) = (1 - t) * x0 + t * x1
v(t) = x1 - x0`}</code>
                </pre>
                <p>
                    The exact parameterization can change, but the implementation goal is
                    the same: produce a stable vector field that is easy to verify before
                    any performance work starts.
                </p>
            </MathDisclosure> */}
        </>
    );
}
