import { KatexBlock } from "@/components/KatexBlock";
import { KatexInline } from "@/components/KatexInline";

export function CICFMBody() {
    return (
        <>
            <p>
                If you are unsure of what flow matching really is, I recommend checking out our last blog: <a href="https://tiny-inr.vercel.app/post/what-is-flow-matching">"What is Flow Matching?"</a>
                This post assumes the reader knows that a flow matching model's learning objective is to nudge noise toward data, step by step.
            </p>

            <h2>
                What is a Loss Function?
            </h2>

            <p>
                Before diving into the nitty-gritty of why CICFM is the optimal loss function choice in flow matching, let's wind back and understand what a loss function is.
                Simply put, a loss function is a number that tells us how wrong our model's guess is, thus allowing us to look for ways of minimizing that number to get more accurate results.
                Essentially, training a model is nothing but nudging its knobs to make that number smaller, recursively, over lots of examples.
            </p>

            <h2>
                The Flow Matching Objective
            </h2>


            <p>
                The model's main goal is to guess a "velocity," i.e., which direction to nudge the current guess toward a more accurate one, at each step. For single numbers, we merely need one guess.
                But in our case, we're dealing with whole functions, such as images (modelled from pixel coordinate to color) or a protein's 3D shape (modelled from atom index to 3D position).
            </p>

            <h3>
                We are faced with two extremes: we need to decide whether we're predicting velocity for the whole image or per pixel
            </h3>

            <p>
                If we treat the entire image/shape as one giant vector and predict one big velocity for all of it at once, we'd be replicating traditional flow-matching models.
                But because this project is focused on making flow matching domain-agnostic, this approach doesn't fit; it chains us to a fixed resolution/size, and it requires a separate compression stage
                to precede it.
            </p>

            <p>
                On the other hand, if we predict the velocity for each pixel/atom independently, with no knowledge of what any other pixel is doing, we get a cheaper and more flexible solution
                as a direct result of that naive independence assumption. We quickly realize, however, that this is useless in practice, since it loses all coherence with the rest of the shape.
                To put this into perspective, it's like trying to color in a picture while only ever seeing a tiny subsection of it at once.
            </p>

            <h2>
                CICFM, The Middle State
            </h2>

            <p>
                To mitigate this blindness, CICFM hands each point-wise guess a compressed idea of what the full shape looks like.
                This gives every point context while keeping its prediction independent of the others. We end up with the best of both worlds: we perserve both accuracy and performance.
            </p>

            <p>
                More formally, <strong>Conditionally Indepedendent Continuous Flow Matching</strong> loss offers a specialized training objective used in continuous normalizing flows and generative models
                where spatial, temporal, or coordinate-wise components are modelled as conditionally independent given a latent context variable <em>z</em>.
            </p>
        </>
    )

}