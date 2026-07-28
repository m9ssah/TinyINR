import { ArrowRight } from "lucide-react";

function Keyword({ children }: { children: React.ReactNode }) {
  return <span className="font-semibold text-accent">{children}</span>;
}

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="font-mono text-xs uppercase tracking-widest text-secondary">
        {label}
      </div>
      <div className="mt-3 text-3xl font-bold text-primary">{value}</div>
      <p className="mt-2 text-sm leading-6 text-secondary">{detail}</p>
    </div>
  );
}

function TimingStep({
  title,
  detail,
}: {
  title: string;
  detail: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-background px-5 py-4">
      <div className="font-mono text-sm font-semibold text-primary">{title}</div>
      <p className="mt-2 text-sm leading-6 text-secondary">{detail}</p>
    </div>
  );
}

function TimingFlow() {
  return (
    <div className="my-10 rounded-2xl border border-border bg-surface p-5">
      <div className="mb-5 font-mono text-xs uppercase tracking-widest text-secondary">
        End-to-end CUDA timing
      </div>
      <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">
        <TimingStep
          title="H2D copy"
          detail="Move coordinates from host memory to device memory."
        />
        <ArrowRight className="mx-auto hidden h-5 w-5 text-accent md:block" />
        <TimingStep
          title="CUDA kernel"
          detail="Compute sine and cosine bands in parallel."
        />
        <ArrowRight className="mx-auto hidden h-5 w-5 text-accent md:block" />
        <TimingStep
          title="D2H copy"
          detail="Move embedded features back to host memory."
        />
      </div>
    </div>
  );
}

const benchmarkSpeedups = [
  {
    coordinates: "1,024",
    results: [
      { frequencies: 4, speedup: 7.21754 },
      { frequencies: 8, speedup: 10.2635 },
      { frequencies: 16, speedup: 12.6378 },
    ],
  },
  {
    coordinates: "16,384",
    results: [
      { frequencies: 4, speedup: 11.8262 },
      { frequencies: 8, speedup: 16.9418 },
      { frequencies: 16, speedup: 21.3239 },
    ],
  },
  {
    coordinates: "65,536",
    results: [
      { frequencies: 4, speedup: 16.7509 },
      { frequencies: 8, speedup: 21.2391 },
      { frequencies: 16, speedup: 17.1991 },
    ],
  },
  {
    coordinates: "262,144",
    results: [
      { frequencies: 4, speedup: 19.3634 },
      { frequencies: 8, speedup: 23.1688 },
      { frequencies: 16, speedup: 18.3053 },
    ],
  },
];

const frequencyStyles = {
  4: { color: "#39d3bb", label: "F = 4" },
  8: { color: "#f2c14e", label: "F = 8" },
  16: { color: "#f97068", label: "F = 16" },
};

function BenchmarkSpeedupPlot() {
  const chartWidth = 720;
  const chartHeight = 320;
  const margin = { top: 26, right: 20, bottom: 62, left: 58 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const yMax = 25;
  const groupWidth = plotWidth / benchmarkSpeedups.length;
  const barWidth = 22;
  const barGap = 8;
  const yTicks = [0, 5, 10, 15, 20, 25];

  return (
    <figure className="my-10 rounded-xl border border-border bg-surface p-5">
      <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <figcaption className="font-mono text-xs uppercase tracking-widest text-secondary">
            End-to-end CUDA speedup
          </figcaption>
          <p className="mt-2 max-w-xl text-sm leading-6 text-secondary">
            CPU time divided by full GPU path time: host-to-device copy, kernel,
            and device-to-host copy.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {Object.entries(frequencyStyles).map(([frequency, style]) => (
            <div
              key={frequency}
              className="flex items-center gap-2 font-mono text-xs text-secondary"
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: style.color }}
              />
              {style.label}
            </div>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg
          role="img"
          aria-labelledby="benchmark-speedup-title benchmark-speedup-description"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="h-auto min-w-[680px]"
        >
          <title id="benchmark-speedup-title">
            End-to-end CUDA speedup by coordinate count and frequency bands
          </title>
          <desc id="benchmark-speedup-description">
            A grouped bar chart showing end-to-end CUDA speedup over CPU for
            coordinate counts from 1024 to 262144 and frequency band counts of
            4, 8, and 16.
          </desc>

          <line
            x1={margin.left}
            y1={margin.top}
            x2={margin.left}
            y2={margin.top + plotHeight}
            stroke="#2d3748"
          />
          <line
            x1={margin.left}
            y1={margin.top + plotHeight}
            x2={margin.left + plotWidth}
            y2={margin.top + plotHeight}
            stroke="#2d3748"
          />

          {yTicks.map((tick) => {
            const y = margin.top + plotHeight - (tick / yMax) * plotHeight;

            return (
              <g key={tick}>
                <line
                  x1={margin.left}
                  y1={y}
                  x2={margin.left + plotWidth}
                  y2={y}
                  stroke="#2d3748"
                  strokeDasharray={tick === 0 ? undefined : "4 6"}
                  opacity={tick === 0 ? 1 : 0.55}
                />
                <text
                  x={margin.left - 12}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-secondary font-mono text-[11px]"
                >
                  {tick}x
                </text>
              </g>
            );
          })}

          {benchmarkSpeedups.map((group, groupIndex) => {
            const groupCenter = margin.left + groupWidth * groupIndex + groupWidth / 2;
            const totalBarWidth = group.results.length * barWidth + 2 * barGap;

            return (
              <g key={group.coordinates}>
                {group.results.map((result, resultIndex) => {
                  const x =
                    groupCenter -
                    totalBarWidth / 2 +
                    resultIndex * (barWidth + barGap);
                  const barHeight = (result.speedup / yMax) * plotHeight;
                  const y = margin.top + plotHeight - barHeight;
                  const style =
                    frequencyStyles[
                      result.frequencies as keyof typeof frequencyStyles
                    ];

                  return (
                    <g key={result.frequencies}>
                      <rect
                        x={x}
                        y={y}
                        width={barWidth}
                        height={barHeight}
                        rx="4"
                        fill={style.color}
                      />
                      <text
                        x={x + barWidth / 2}
                        y={y - 6}
                        textAnchor="middle"
                        className="fill-primary font-mono text-[10px]"
                      >
                        {result.speedup.toFixed(1)}x
                      </text>
                    </g>
                  );
                })}
                <text
                  x={groupCenter}
                  y={margin.top + plotHeight + 28}
                  textAnchor="middle"
                  className="fill-secondary font-mono text-xs"
                >
                  {group.coordinates}
                </text>
              </g>
            );
          })}

          <text
            x={margin.left + plotWidth / 2}
            y={chartHeight - 8}
            textAnchor="middle"
            className="fill-secondary font-mono text-[11px]"
          >
            Coordinates
          </text>
          <text
            x={16}
            y={margin.top + plotHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90 16 ${margin.top + plotHeight / 2})`}
            className="fill-secondary font-mono text-[11px]"
          >
            Speedup over CPU
          </text>
        </svg>
      </div>
    </figure>
  );
}

function ResultsTable() {
  const rows = [
    {
      coordinates: "1,024",
      frequencies: "4",
      cpu: "0.33 ms",
      kernel: "0.006 ms",
      endToEnd: "0.046 ms",
      speedup: "7.2x",
    },
    {
      coordinates: "16,384",
      frequencies: "16",
      cpu: "25.47 ms",
      kernel: "0.096 ms",
      endToEnd: "1.19 ms",
      speedup: "21.3x",
    },
    {
      coordinates: "65,536",
      frequencies: "8",
      cpu: "47.50 ms",
      kernel: "0.114 ms",
      endToEnd: "2.24 ms",
      speedup: "21.2x",
    },
    {
      coordinates: "262,144",
      frequencies: "8",
      cpu: "189.91 ms",
      kernel: "0.421 ms",
      endToEnd: "8.20 ms",
      speedup: "23.2x",
    },
  ];

  return (
    <div className="my-10 overflow-x-auto rounded-xl border border-border">
      <table className="w-full min-w-[680px] border-collapse text-sm">
        <thead className="bg-surface">
          <tr className="font-mono text-xs uppercase tracking-widest text-secondary">
            <th className="border-b border-border px-4 py-3 text-right">
              Coordinates
            </th>
            <th className="border-b border-border px-4 py-3 text-right">
              Frequencies
            </th>
            <th className="border-b border-border px-4 py-3 text-right">
              CPU time
            </th>
            <th className="border-b border-border px-4 py-3 text-right">
              CUDA kernel
            </th>
            <th className="border-b border-border px-4 py-3 text-right">
              CUDA end-to-end
            </th>
            <th className="border-b border-border px-4 py-3 text-right">
              Speedup
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.coordinates}-${row.frequencies}`}>
              <td className="border-b border-border/60 px-4 py-3 text-right font-mono text-primary">
                {row.coordinates}
              </td>
              <td className="border-b border-border/60 px-4 py-3 text-right font-mono text-primary">
                {row.frequencies}
              </td>
              <td className="border-b border-border/60 px-4 py-3 text-right font-mono text-secondary">
                {row.cpu}
              </td>
              <td className="border-b border-border/60 px-4 py-3 text-right font-mono text-secondary">
                {row.kernel}
              </td>
              <td className="border-b border-border/60 px-4 py-3 text-right font-mono text-secondary">
                {row.endToEnd}
              </td>
              <td className="border-b border-border/60 px-4 py-3 text-right font-mono font-semibold text-accent">
                {row.speedup}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MilestoneTwoPostBody() {
  return (
    <>
      <p>
        This week, we implemented and benchmarked the first custom{" "}
        <Keyword>CUDA kernel</Keyword> in TinyINR: a GPU version of the
        coordinate Fourier embedding module. Previously, we built the CPU-side
        foundation with <Keyword>Tensor</Keyword>, <Keyword>CoordinateBatch</Keyword>,
        and <Keyword>FourierEmbedding</Keyword>. This milestone moved one clean
        operation from that pipeline onto the GPU and measured whether CUDA
        actually helped.
      </p>

      <p>
        The operation we targeted was coordinate embedding. Given a batch of
        coordinates with shape:
      </p>

      <pre>
        <code>{`[B, N, D]`}</code>
      </pre>

      <p>
        the embedding expands each coordinate using sine and cosine features
        across multiple frequency bands. With <Keyword>F</Keyword> frequency
        bands, the logical output can be viewed as:
      </p>

      <pre>
        <code>{`[B, N, D, F, 2]`}</code>
      </pre>

      <p>
        For image coordinates, <Keyword>D = 2</Keyword>, representing{" "}
        <code>(x, y)</code>. Each coordinate scalar is transformed into sine and
        cosine values before being passed into the later INR model.
      </p>

      <h2>What we compared</h2>

      <p>
        Coordinate embedding made a good first CUDA target because each output
        value can be computed independently. One thread does not need to wait
        for another thread&apos;s result, so the work maps naturally to GPU
        parallelism.
      </p>

      <TimingFlow />

      <p>We benchmarked three timings:</p>

      <ul>
        <li>
          <Keyword>CPU time</Keyword>: the reference implementation running on
          the host.
        </li>
        <li>
          <Keyword>CUDA kernel-only time</Keyword>: just the GPU computation.
        </li>
        <li>
          <Keyword>CUDA end-to-end time</Keyword>: host-to-device copy, kernel
          execution, and device-to-host copy together.
        </li>
      </ul>

      <p>
        This split matters because a GPU kernel can be extremely fast while the
        overall GPU path is still limited by memory transfer overhead. For small
        inputs, copying data can take more time than computing the embedding.
      </p>

      <h2>Correctness first</h2>

      <p>
        Before looking at speed, we checked whether the CUDA output matched the
        CPU implementation. Across all benchmark cases, the maximum absolute
        error was:
      </p>

      <div className="my-8 grid gap-4 md:grid-cols-3">
        <MetricCard
          label="Max absolute error"
          value="5.96e-08"
          detail="Well below the 1e-5 parity threshold."
        />
        <MetricCard
          label="Benchmark cases"
          value="12"
          detail="Four coordinate counts crossed with three frequency counts."
        />
        <MetricCard
          label="Best end-to-end speedup"
          value="23.2x"
          detail="Measured at 262,144 coordinates and 8 frequency bands."
        />
      </div>

      <p>
        This confirmed that the CUDA kernel was numerically aligned with the CPU
        reference. The main debugging risk was not the sine and cosine formula;
        it was indexing. The kernel stores the logical output as a flat array,
        so the thread index has to land in the correct coordinate, frequency,
        and sine/cosine slot.
      </p>

      <pre>
        <code>{`input index  = (b * N + n) * D + d
output index = input_index * F * 2 + f * 2 + trig`}</code>
      </pre>

      <h2>Benchmark results</h2>

      <p>
        The CUDA kernel was faster than the CPU implementation in every tested
        case. We tested coordinate counts of <Keyword>1,024</Keyword>,{" "}
        <Keyword>16,384</Keyword>, <Keyword>65,536</Keyword>, and{" "}
        <Keyword>262,144</Keyword>, each with <Keyword>4</Keyword>,{" "}
        <Keyword>8</Keyword>, and <Keyword>16</Keyword> frequency bands.
      </p>

      <BenchmarkSpeedupPlot />

      <ResultsTable />

      <p>
        For the smallest benchmark, with <Keyword>1,024</Keyword> coordinates
        and <Keyword>4</Keyword> frequency bands, the CPU took about{" "}
        <Keyword>0.33 ms</Keyword>, while the CUDA kernel took only{" "}
        <Keyword>0.006 ms</Keyword>. That is a kernel-only speedup of roughly{" "}
        <Keyword>52x</Keyword>.
      </p>

      <p>
        As the workload grew, the GPU advantage became clearer. For{" "}
        <Keyword>262,144</Keyword> coordinates and <Keyword>8</Keyword>{" "}
        frequency bands, the CPU took about <Keyword>189.9 ms</Keyword>, while
        the CUDA kernel took about <Keyword>0.42 ms</Keyword>. That produced a
        kernel-only speedup of about <Keyword>451x</Keyword>.
      </p>

      <h2>Kernel speed is not the full story</h2>

      <p>
        The most important pattern was that device-to-host copy time became a
        major part of end-to-end runtime. For <Keyword>262,144</Keyword>{" "}
        coordinates and <Keyword>8</Keyword> frequency bands, the kernel itself
        took only about <Keyword>0.42 ms</Keyword>, but copying the output back
        to the CPU took about <Keyword>7.19 ms</Keyword>.
      </p>

      <p>
        That means the bottleneck shifted. Once the math moved to CUDA, memory
        movement became a bigger part of the total cost than the computation
        itself. This is why we report both kernel-only speedup and end-to-end
        speedup.
      </p>

      <pre>
        <code>{`kernel_speedup = cpu_ms / kernel_ms
e2e_speedup    = cpu_ms / (h2d_ms + kernel_ms + d2h_ms)`}</code>
      </pre>

      <h2>What we learned</h2>

      <p>
        The biggest lesson from this benchmark is that CUDA speedups need to be
        reported carefully. If we only reported kernel-only time, the results
        would look enormous: in some cases, over <Keyword>400x</Keyword> faster
        than CPU. But for the actual pipeline, end-to-end timing is more honest
        because it includes the cost of moving data to and from the GPU.
      </p>

      <p>
        From these results, the end-to-end CUDA path was at least{" "}
        <Keyword>7x</Keyword> faster in the smallest case and over{" "}
        <Keyword>20x</Keyword> faster in several larger cases. This suggests
        that coordinate embedding is a strong candidate for GPU acceleration,
        especially as image size and frequency count grow.
      </p>

      <h2>What comes next</h2>

      <p>
        For the next stage, we plan to build a minimal trainable INR baseline.
        That means implementing the MLP interface, activation, reconstruction
        loss, backward/update path, and training loop over uniformly sampled
        coordinates.
      </p>

      <p>
        After that, we can start asking the more interesting sampling question:
        which coordinates should the model spend most of its training budget on?
      </p>
    </>
  );
}
