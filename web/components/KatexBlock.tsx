import katex from "katex";

interface KatexBlockProps {
    expression: string;
}

export function KatexBlock({ expression }: KatexBlockProps) {
    const html = katex.renderToString(expression, {
        displayMode: true,
        throwOnError: false,
        output: "html",
    });

    return (
        <div
            className="my-6 overflow-x-auto rounded-xl border border-border bg-background p-4"
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
}